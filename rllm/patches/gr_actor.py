"""
Gradient Regularization (GR) for verl's DataParallelPPOActor.

Replaces `update_policy()` with a GR variant per:
  Ackermann, Noukhovitch, Ishida, Sugiyama,
  "Gradient Regularization Prevents Reward Hacking in
   Reinforcement Learning from Human Feedback and Verifiable Rewards"
  arXiv:2602.18037, February 2026.

Algorithm (Figure 9 and Section C.1):
  1. Normal forward+backward pass → accumulate gradient g = ∇J(ϕ)
  2. Clip disturbance g to 10.0
  3. Perturb transformer block params: ϕ' = ϕ + ε·g
  4. Second forward+backward pass (same data, perturbed ϕ) → g' = ∇J(ϕ')
  5. Clip perturbed gradient g' to 10.0
  6. Combine: final_grad = g + γ · (g' − g) / ε
  7. Restore original params → normal optimizer.step() (clips to 1.0)

Key implementation details:
  - loss_scale_factor=1.0 in both passes → SUM gradients (matching paper)
  - Only transformer block params are perturbed (not embedding/output head)
  - bf16 only; fp16 scaler is not supported (raises NotImplementedError)
"""

from __future__ import annotations

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from codetiming import Timer
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch

# ---------------------------------------------------------------------------
# Module-level state (set by the monkey-patch hook in verl_patch_hook.py)
# ---------------------------------------------------------------------------
_original_update_policy = None
"""Reference to the original DataParallelPPOActor.update_policy (unbound)."""

# ---------------------------------------------------------------------------
# GR configuration defaults — injected onto ActorConfig if fields are missing
# ---------------------------------------------------------------------------
_GR_DEFAULTS = {
    "use_gr": False,
    "gr_gamma": 1e-3,               # γ: GR strength
    "gr_epsilon": 1e-3,             # ε: perturbation scale
    "gr_disturbance_clip": 10.0,    # clip for disturbance g before perturb
    "gr_perturbed_clip": 10.0,      # clip for perturbed gradient g'
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_gr_config(config):
    """Add GR fields to *config* (an ActorConfig) if they are missing.

    This avoids forcing a site-packages edit of ``verl/workers/config/actor.py``
    when the fields can simply be set via Hydra ``+`` overrides.
    """
    for key, default in _GR_DEFAULTS.items():
        if not hasattr(config, key):
            object.__setattr__(config, key, default)


def _is_transformer_block_param(name: str) -> bool:
    """Return True if *name* belongs to a transformer block.

    Paper §C.1: "perturb only the parameters of the transformer blocks,
    including attention matrices, MLP weights and layer norm parameters,
    but **not** perturb the embedding layer or final output layer."
    """
    # These substrings appear in HF-style parameter names for
    # embedding / output-head layers.
    skip = ("embed_tokens", "lm_head", "wte")
    for token in skip:
        if token in name:
            return False
    return True


def _clip_grad_norm_fsdp(actor_module, max_norm: float):
    """FSDP-aware gradient clipping returning the (scalar) global norm."""
    if isinstance(actor_module, FSDP):
        grad_norm = actor_module.clip_grad_norm_(max_norm=max_norm)
    elif isinstance(actor_module, FSDPModule):
        grad_norm = fsdp2_clip_grad_norm_(
            actor_module.parameters(), max_norm=max_norm
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_module.parameters(), max_norm=max_norm
        )

    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()  # DTensor → plain tensor
    return grad_norm


def _compute_grad_norm_fsdp(actor_module) -> float:
    """FSDP-aware global gradient norm — measurement only, NO clipping.

    Uses the same parameter-gathering logic as ``_optimizer_step()`` but
    with ``max_norm=inf`` so that gradients are left untouched.
    """
    max_norm = float("inf")
    if isinstance(actor_module, FSDP):
        grad_norm = actor_module.clip_grad_norm_(max_norm=max_norm)
    elif isinstance(actor_module, FSDPModule):
        grad_norm = fsdp2_clip_grad_norm_(
            actor_module.parameters(), max_norm=max_norm
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_module.parameters(), max_norm=max_norm
        )

    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()
    return float(grad_norm.detach().cpu().item())


# ---------------------------------------------------------------------------
# Core GR operations
# ---------------------------------------------------------------------------

def _gr_save_and_perturb(self, epsilon: float, disturbance_clip: float) -> tuple[dict, float]:
    """Save gradients and original data, then perturb transformer-block params.

    Implements paper Figure 9:
      grad1 = norm_clip(grad1)
      phi_2 = phi + varepsilon * grad1

    All parameters' pass-1 gradients are saved so that the GR combination
    step can restore correct gradients for *unperturbed* params as well
    (their g' indirectly differs from g because transformer blocks were perturbed).

    Returns
    -------
    (saved, orig_grad_norm)
        saved : dict[str, tuple[Tensor, Tensor | None, bool]]
            ``{param_name: (saved_grad, saved_data, was_perturbed)}``.
        orig_grad_norm : float
            Global gradient norm **before** disturbance clipping (i.e., the raw
            pass-1 gradient norm).  Computed with the same FSDP-aware method as
            ``_optimizer_step()``.
    """
    saved: dict[str, tuple[torch.Tensor, torch.Tensor | None, bool]] = {}

    # Compute the raw pass-1 gradient norm (before clipping), using the same
    # FSDP-aware path as _optimizer_step().
    orig_grad_norm = _compute_grad_norm_fsdp(self.actor_module)

    # Clip disturbance gradient (paper: grad1 = norm_clip(grad1))
    _clip_grad_norm_fsdp(self.actor_module, disturbance_clip)

    for name, param in self.actor_module.named_parameters():
        if param.grad is None:
            continue

        is_transformer = _is_transformer_block_param(name)

        if is_transformer:
            saved[name] = (param.grad.clone(), param.data.clone(), True)
            # ϕ' = ϕ + ε·g  (only transformer blocks)
            param.data.add_(epsilon * param.grad)
        else:
            # Save pass-1 grad for non-transformer params — we restore it
            # after pass 2 because g' ≠ g for these params (indirect effect
            # of perturbing transformer blocks on the forward pass).
            saved[name] = (param.grad.clone(), None, False)

    return saved, orig_grad_norm


def _gr_combine_and_restore(
    self,
    saved: dict,
    gamma: float,
    epsilon: float,
    perturbed_clip: float,
) -> None:
    """Clip perturbed gradient, combine per GR formula, restore original params.

    Implements paper Figure 9:
      grad2 = norm_clip(grad2)
      comb_grad = grad1 + gamma * (grad2 - grad1) / varepsilon
      model.set_state_dict(phi)
      model.grad = comb_grad

    For *perturbed* params (transformer blocks): applies the full GR formula.
    For *unperturbed* params (embedding / output head): restores the pass-1
    gradient directly, since g' differs from g only due to indirect forward-pass
    effects — there is no perturbation to base the finite-difference on.
    """
    # Clip perturbed gradient (paper: grad2 = norm_clip(grad2))
    _clip_grad_norm_fsdp(self.actor_module, perturbed_clip)

    for name, param in self.actor_module.named_parameters():
        if name not in saved:
            continue

        g_saved, original_data, was_perturbed = saved[name]

        if was_perturbed:
            g_prime = param.grad
            if g_prime is not None:
                # Paper Figure 9:
                #   comb_grad = grad1 + gamma * (grad2 - grad1) / varepsilon
                param.grad = g_saved + gamma * (g_prime - g_saved) / epsilon
            else:
                param.grad = g_saved
            # Restore original parameters (paper: model.set_state_dict(phi))
            param.data.copy_(original_data)
        else:
            # Unperturbed param (embedding / output head): simply restore
            # the pass-1 gradient.  g' was computed with perturbed transformer
            # blocks in the forward pass, so it differs from g even though this
            # param itself was never changed.
            param.grad = g_saved
            # data was never changed — no restoration needed


# ---------------------------------------------------------------------------
# Micro-batch forward+loss+backward  (extracted from original update_policy)
# ---------------------------------------------------------------------------

def _run_gr_micro_batch(
    self,
    micro_batch: DataProto,
    temperature: float,
    on_policy: bool,
    loss_agg_mode: str,
    loss_mode: str,
    entropy_coeff: float,
    metrics: dict,
    suffix: str = "",
) -> tuple[dict, float]:
    """Forward + policy-loss + backward for a single micro-batch.

    Replicates the micro-batch body of the *original* ``update_policy``
    **except** that ``loss_scale_factor`` is forced to **1.0** so that
    gradients are **summed** across micro-batches — matching the paper's
    manual accumulation loop (Figure 9).

    Returns
    -------
    (mb_metrics, orig_scale_factor)
        orig_scale_factor is the **original** verl loss_scale_factor (the one
        that would have been used without GR).  The caller accumulates these
        across micro-batches to rescale the combined gradient back to verl's
        convention after the GR finite-difference steps.
    """
    # ---- move to device ---------------------------------------------------
    micro_batch = micro_batch.to(get_device_id())
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

    response_mask = model_inputs["response_mask"]
    old_log_prob_in = model_inputs["old_log_probs"]
    advantages = model_inputs["advantages"]

    # The **original** verl loss_scale_factor — used only to accumulate a
    # total rescaling denominator so that the final combined gradient matches
    # verl's convention (average over micro-batches).
    if self.config.use_dynamic_bsz:
        orig_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
    else:
        orig_scale_factor = 1.0 / self.gradient_accumulation

    # GR: loss_scale_factor = 1.0  → sum gradients (paper Figure 9)
    loss_scale_factor = 1.0

    # ---- forward ----------------------------------------------------------
    calculate_entropy = entropy_coeff != 0
    entropy, log_prob = self._forward_micro_batch(
        model_inputs,
        temperature=temperature,
        calculate_entropy=calculate_entropy,
    )

    # ---- old_log_prob (three modes, matching original) --------------------
    if (
        hasattr(self.config, "use_rollout_log_probs")
        and self.config.use_rollout_log_probs
    ):
        old_log_prob = old_log_prob_in
    elif on_policy:
        old_log_prob = log_prob.detach()
    else:
        old_log_prob = old_log_prob_in

    # ---- policy loss ------------------------------------------------------
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=self.config,
        rollout_is_weights=rollout_is_weights,
    )
    mb_metrics: dict = {}
    mb_metrics.update(pg_metrics)

    # ---- entropy ----------------------------------------------------------
    if entropy_coeff != 0:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
        )
        policy_loss = pg_loss - entropy_loss * entropy_coeff
    else:
        policy_loss = pg_loss

    # ---- KL loss (in policy loss) -----------------------------------------
    if self.config.use_kl_loss:
        ref_log_prob = model_inputs["ref_log_prob"]
        kld = kl_penalty(
            logprob=log_prob,
            ref_logprob=ref_log_prob,
            kl_penalty=self.config.kl_loss_type,
        )
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
        )
        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
        mb_metrics[f"actor/kl_loss{suffix}"] = (
            kl_loss.detach().item() * loss_scale_factor
        )

    # ---- backward ---------------------------------------------------------
    loss = policy_loss * loss_scale_factor  # = policy_loss (scale_factor = 1.0)
    loss.backward()

    # ---- metrics (logged at original scale for comparability) -------------
    mb_metrics[f"actor/pg_loss{suffix}"] = (
        pg_loss.detach().item() * loss_scale_factor
    )

    return mb_metrics, orig_scale_factor


# ---------------------------------------------------------------------------
# Main entry-point — replacement for DataParallelPPOActor.update_policy
# ---------------------------------------------------------------------------

def gr_update_policy(self, data: DataProto) -> dict:
    """GR-augmented policy update (replaces ``DataParallelPPOActor.update_policy``).

    When ``self.config.use_gr`` is ``False`` the *original* ``update_policy`` is
    called transparently so that GR imposes zero overhead when disabled.
    """
    global _original_update_policy

    # ---- guard: GR disabled → fall back to original -----------------------
    _ensure_gr_config(self.config)
    if not self.config.use_gr:
        return _original_update_policy(self, data)

    # ---- fp16 guard -------------------------------------------------------
    if self.scaler is not None:
        raise NotImplementedError(
            "GR with fp16 mixed-precision (ShardedGradScaler) is not yet "
            "supported. The current setup uses bf16 by default, which is fine."
        )

    # ====================================================================
    # Setup  (replicates original update_policy lines 372–418)
    # ====================================================================
    self.actor_module.train()
    temperature = data.meta_info["temperature"]

    select_keys = [
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "old_log_probs",
        "advantages",
    ]
    if self.config.use_kl_loss:
        select_keys.append("ref_log_prob")
    if "rollout_is_weights" in data.batch.keys():
        select_keys.append("rollout_is_weights")
    if "rollout_log_probs" in data.batch.keys():
        select_keys.append("rollout_log_probs")

    has_mm = "multi_modal_inputs" in data.non_tensor_batch.keys()
    non_tensor_select = ["multi_modal_inputs"] if has_mm else []
    data = data.select(
        batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select
    )

    mini_batches = data.split(self.config.ppo_mini_batch_size)
    on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

    # Cached config values
    loss_agg_mode = self.config.loss_agg_mode
    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
    entropy_coeff = self.config.entropy_coeff
    gr_gamma = self.config.gr_gamma
    gr_epsilon = self.config.gr_epsilon
    gr_dist_clip = self.config.gr_disturbance_clip
    gr_pert_clip = self.config.gr_perturbed_clip

    metrics: dict = {}

    # ====================================================================
    # Main training loop (ppo_epochs × mini_batches)
    # ====================================================================
    for _epoch in range(self.config.ppo_epochs):
        for _bidx, mini_batch in enumerate(mini_batches):
            # ---- micro-batch splitting ------------------------------------
            if self.config.use_dynamic_bsz:
                max_tokens = (
                    self.config.ppo_max_token_len_per_gpu
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, _ = prepare_dynamic_batch(
                    mini_batch, max_token_len=max_tokens
                )
            else:
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size
                    // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(
                    self.config.ppo_micro_batch_size_per_gpu
                )

            # ================================================================
            #  PASS 1 — compute g = ∇J(ϕ)
            #
            #  Paper:  for idx in range(batch_size / accumulation_steps):
            #              loss = grpo_loss(...)
            #              loss.backwards()
            #              grad1 += model.grad
            # ================================================================
            total_orig_scale = 0.0
            self.actor_optimizer.zero_grad()
            with Timer(name="gr_pass1", logger=None) as t_p1:
                for mb in micro_batches:
                    mb_m, orig_s = _run_gr_micro_batch(
                        self, mb, temperature, on_policy,
                        loss_agg_mode, loss_mode, entropy_coeff,
                        metrics, suffix="_p1",
                    )
                    total_orig_scale += orig_s
                    append_to_dict(metrics, mb_m)

            # ================================================================
            #  GR STEP 1 — save g, clip disturbance, perturb
            #
            #  Paper:  grad1 = norm_clip(grad1)
            #          phi_2 = phi + varepsilon * grad1
            # ================================================================
            with Timer(name="gr_perturb", logger=None) as t_perturb:
                saved, orig_grad_norm = _gr_save_and_perturb(self, gr_epsilon, gr_dist_clip)

            # ================================================================
            #  PASS 2 — compute g' = ∇J(ϕ + ε·g)
            #
            #  Paper:  model.set_state_dict(phi_2)
            #          for idx in range(...):
            #              loss = grpo_loss(...)
            #              loss.backwards()
            #              grad2 += model.grad
            # ================================================================
            self.actor_optimizer.zero_grad()
            with Timer(name="gr_pass2", logger=None) as t_p2:
                for mb in micro_batches:
                    mb_m, _ = _run_gr_micro_batch(
                        self, mb, temperature, on_policy,
                        loss_agg_mode, loss_mode, entropy_coeff,
                        metrics, suffix="_p2",
                    )
                    append_to_dict(metrics, mb_m)

            # ================================================================
            #  GR STEP 2 — clip g', combine, restore ϕ
            #
            #  Paper:  grad2 = norm_clip(grad2)
            #          comb_grad = grad1 + gamma * (grad2 - grad1) / varepsilon
            #          model.set_state_dict(phi)
            #          model.grad = comb_grad
            # ================================================================
            with Timer(name="gr_combine_step", logger=None) as t_combine:
                _gr_combine_and_restore(self, saved, gr_gamma, gr_epsilon, gr_pert_clip)

                # ---- Rescale combined gradient back to verl's convention -----
                # GR internally sums gradients (loss_scale_factor=1.0), matching
                # paper Figure 9.  Verl averages over micro-batches.  The ratio
                # between the two conventions is approximately the number of
                # micro-batches (exact when each micro-batch contributes an equal
                # number of response tokens).
                #
                # total_orig_scale = Σ (response_mask.shape[0] / ppo_mini_batch_size)
                # is always ≈ 1.0 because Σ response_mask.shape[0] ≈ ppo_mini_batch_size,
                # so it does NOT capture the sum→average correction.  Use
                # len(micro_batches) instead.
                n_mb = max(len(micro_batches), 1)
                append_to_dict(metrics, {
                    "actor/gr_total_orig_scale": total_orig_scale,
                    "actor/gr_n_micro_batches": float(n_mb),
                })
                for _name, _param in self.actor_module.named_parameters():
                    if _param.grad is not None:
                        _param.grad.div_(n_mb)

                # Normal optimizer step (paper: optimizer.step())
                grad_norm = self._optimizer_step()

            # ---- GR timing metrics (milliseconds) --------------------------
            append_to_dict(metrics, {
                "timing_s/gr_t_p1": t_p1.last,
                "timing_s/gr_t_perturb": t_perturb.last,
                "timing_s/gr_t_p2": t_p2.last,
                "timing_s/gr_t_combine": t_combine.last,
                "actor/gr_orig_grad_norm": orig_grad_norm,
                "actor/grad_norm": grad_norm.detach().item(),
            })

    self.actor_optimizer.zero_grad()
    return metrics
