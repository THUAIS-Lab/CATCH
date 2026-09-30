import wrapt

_TARGET = "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd"
_TARGET_GR = "verl.workers.actor.dp_actor"


def setup():
    # ---- vLLM ZeroMQ patch (DISABLED — stale, do not activate) ----
    # @wrapt.when_imported(_TARGET)
    # def _patch(mod):
    #     ...

    # ---- Unified dispatcher (GR + chi-squared) ----
    # Replaces DataParallelPPOActor.update_policy with a dispatch function
    # that routes to the appropriate implementation based on mutually
    # exclusive config flags.
    @wrapt.when_imported(_TARGET_GR)
    def _patch_gr(mod):
        from rllm.patches.chi2_loss import chi2_update_policy
        from rllm.patches.gr_actor import gr_update_policy

        # ------------------------------------------------------------------
        # Patch FSDPActorConfig.__init__ to accept chi2 fields that are not
        # declared on the site-packages dataclass.  omega_conf_to_dataclass()
        # calls FSDPActorConfig(**dict), which rejects unknown kwargs.
        # ------------------------------------------------------------------
        from verl.workers.config.actor import FSDPActorConfig

        _CHI2_FIELDS = {"use_chi_square_loss", "chi_square_loss_coef"}
        _original_fsdp_init = FSDPActorConfig.__init__

        def _patched_fsdp_init(self, **kwargs):
            chi2_kwargs = {}
            for key in _CHI2_FIELDS:
                if key in kwargs:
                    chi2_kwargs[key] = kwargs.pop(key)
            _original_fsdp_init(self, **kwargs)
            for key, val in chi2_kwargs.items():
                object.__setattr__(self, key, val)

        FSDPActorConfig.__init__ = _patched_fsdp_init

        # Save reference to original (single-pass, no regularization)
        mod._original_update_policy = mod.DataParallelPPOActor.update_policy

        # Inject original into both patch modules so they can fall back
        import rllm.patches.chi2_loss as _chi2_mod
        import rllm.patches.gr_actor as _gr_mod

        _chi2_mod._original_update_policy = mod._original_update_policy
        _gr_mod._original_update_policy = mod._original_update_policy

        # ------------------------------------------------------------------
        # Unified dispatcher
        # ------------------------------------------------------------------
        def _dispatch_update_policy(self, data):
            # Ensure all optional config fields exist (defensive fallback
            # when site-packages FSDPActorConfig does not define them).
            from rllm.patches.chi2_loss import _ensure_chi2_config
            from rllm.patches.gr_actor import _ensure_gr_config

            _ensure_gr_config(self.config)
            _ensure_chi2_config(self.config)

            # ---- mutual exclusion validation ------------------------------
            active: list[str] = []
            if getattr(self.config, "use_gr", False):
                active.append("use_gr (gradient regularization)")
            if getattr(self.config, "use_chi_square_loss", False):
                active.append("use_chi_square_loss (chi-squared divergence)")
            if getattr(self.config, "use_kl_loss", False):
                active.append("use_kl_loss (KL divergence)")
            if len(active) > 1:
                raise RuntimeError(
                    f"Only one of {{{', '.join(active)}}} may be enabled "
                    f"at a time. Currently enabled: {active}. "
                    "Chi-squared divergence, KL divergence, and gradient "
                    "regularization are mutually exclusive."
                )

            # ---- dispatch ------------------------------------------------
            if getattr(self.config, "use_gr", False):
                return gr_update_policy(self, data)
            elif getattr(self.config, "use_chi_square_loss", False):
                return chi2_update_policy(self, data)
            else:
                return mod._original_update_policy(self, data)

        mod.DataParallelPPOActor.update_policy = _dispatch_update_policy
