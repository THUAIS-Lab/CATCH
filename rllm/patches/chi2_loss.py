"""
Chi-Squared Divergence Regularization for verl's DataParallelPPOActor.

Implements the chi-squared divergence penalty from:
  Laidlaw, Singhal, Dragan,
  "Correlated Proxies: A New Definition and Improved Mitigation for
   Reward Hacking"
  arXiv:2403.03185, March 2024.

Follows the official RLHF implementation in:
  github.com/cassidylaidlaw/llm_optimization
  src/ppo/custom_trlx_trainers/modeling.py  (PPOWithRefPolicyConfig.loss)

Per-sample estimator (Appendix D.2, Schulman 2020 low-variance form)::

    chi2 = mean_i [ratio_i + inv_ratio_i - 2]

where ``ratio_i = exp(log_ratio_i)`` is the per-sequence probability ratio
(see Lemma A.6 — contextual-bandit / γ=0 reduction).

Loss term added to the PPO objective::

    λ · sqrt(chi2.clamp(min=1e-4))

Numerical stability: log-ratio clipped at ±10 with first-order Taylor
expansion for the clipped tail (matching the official code exactly).

Mutual exclusion: chi-squared, KL, and GR are incompatible — at most one
may be active.  Validation happens in the unified dispatcher in
``verl_patch_hook.py``.
"""

from __future__ import annotations

import math
import time

import torch
from codetiming import Timer
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch

# ---------------------------------------------------------------------------
# Module-level state (set by the monkey-patch hook in verl_patch_hook.py)
# ---------------------------------------------------------------------------
_original_update_policy = None
"""Reference to the original DataParallelPPOActor.update_policy (unbound)."""

# ---------------------------------------------------------------------------
# Chi-squared configuration defaults
# ---------------------------------------------------------------------------
_CHI2_DEFAULTS = {
    "use_chi_square_loss": False,
    "chi_square_loss_coef": 0.0008,
}
"""Default values injected onto the config object when fields are missing.

The default coefficient 0.0008 is the best value reported in Table 4
of the paper (52.83% median win rate, stable across seeds).
"""

_LOG_RATIO_CLIP: float = 10.0
"""Symmetric clip threshold for per-sequence log-ratio sums."""

_EXP_LOG_RATIO_CLIP: float = math.exp(_LOG_RATIO_CLIP)
"""Pre-computed exp(10) ≈ 22026.465 for the linear Taylor expansion branch."""

_CHI2_CLAMP_MIN: float = 1e-4
"""Minimum chi2 value before sqrt, prevents sqrt(0) gradient singularity."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_chi2_config(config):
    """Add chi-squared fields to *config* if they are missing.

    This is a defensive fallback for cases where the Hydra ``+`` override
    did not reach the actor's ``self.config`` DictConfig.  Under normal
    operation the fields are present because OmegaConf DictConfig accepts
    arbitrary keys added via Hydra ``+`` syntax.
    """
    for key, default in _CHI2_DEFAULTS.items():
        if not hasattr(config, key):
            object.__setattr__(config, key, default)


# ---------------------------------------------------------------------------
# Core chi-squared computation
# ---------------------------------------------------------------------------

def compute_chi_square_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    log_ratio_clip: float = _LOG_RATIO_CLIP,
) -> tuple[torch.Tensor, dict]:
    """Compute the chi-squared divergence between policy and reference.

    Strictly follows the official Laidlaw et al. implementation in
    ``src/ppo/custom_trlx_trainers/modeling.py``,
    ``PPOWithRefPolicyConfig.loss()``.

    **Sequence-level formulation**: the per-token log-ratio is summed over
    all response tokens, yielding the per-sequence probability ratio
    :math:`\\prod_t \\pi_\\theta(a_t|s_t) / \\pi_{ref}(a_t|s_t)`.
    This matches the contextual-bandit (:math:`\\gamma=0`) reduction
    (Lemma A.6 / A.7 — OM divergence = AD divergence for tree-structured MDPs).

    **Numerical clipping** (matching official code): when
    :math:`|\\text{log\\_ratio}| \\ge 10`, a first-order Taylor expansion
    replaces ``expm1`` to prevent overflow.  Tracked via ``*_clipfrac``
    monitoring metrics.

    Args:
        log_prob: Current policy log-probabilities.
            Shape ``[batch_size, response_length]``.
        ref_log_prob: Reference (SFT) policy log-probabilities.
            Shape ``[batch_size, response_length]``.
        response_mask: Mask for real vs padding tokens (1 = real).
            Shape ``[batch_size, response_length]``.
        log_ratio_clip: Symmetric clip threshold for per-sequence log-ratio.

    Returns:
        chi2: Scalar tensor — the raw chi2 estimate (BEFORE sqrt), mean
            over batch of ``ratio + inv_ratio − 2``.
        metrics: Dict with ``actor/chi2_value``,
            ``actor/chi2_ratio_clipfrac``, ``actor/chi2_inv_ratio_clipfrac``.
    """
    # ------------------------------------------------------------------
    # Step 1: per-sequence log-ratio sum
    #   log_ratio_i = Σ_t (log π_θ(a_t) − log π_ref(a_t)) over response tokens
    # Shape: [batch_size]
    # ------------------------------------------------------------------
    log_ratio = ((log_prob - ref_log_prob) * response_mask).sum(dim=-1)

    # ------------------------------------------------------------------
    # Step 2: ratio − 1 = exp(log_ratio) − 1  (clip at +10)
    #   Linear Taylor expansion for log_ratio >= clip:
    #     exp(x) ≈ exp(clip) · (1 + x − clip)
    #     ratio − 1 = exp(clip) · (1 + x − clip) − 1
    # ------------------------------------------------------------------
    ratio_m_1 = torch.where(
        log_ratio >= log_ratio_clip,
        _EXP_LOG_RATIO_CLIP * (1.0 + log_ratio - log_ratio_clip) - 1.0,
        torch.special.expm1(log_ratio.clamp(max=log_ratio_clip)),
    )

    # ------------------------------------------------------------------
    # Step 3: inverse ratio − 1 = exp(−log_ratio) − 1  (clip at −10)
    #   Linear Taylor expansion for log_ratio <= -clip:
    #     exp(−x) ≈ exp(clip) · (1 − x − clip)
    #     inv_ratio − 1 = exp(clip) · (1 − x − clip) − 1
    # ------------------------------------------------------------------
    inv_ratio_m_1 = torch.where(
        log_ratio <= -log_ratio_clip,
        _EXP_LOG_RATIO_CLIP * (1.0 - log_ratio - log_ratio_clip) - 1.0,
        torch.special.expm1((-log_ratio).clamp(max=log_ratio_clip)),
    )

    # ------------------------------------------------------------------
    # Step 4: chi2 = mean_i [ratio_i + 1/ratio_i − 2]
    #   = mean_i [ (ratio_i − 1) + (1/ratio_i − 1) ]
    #   = mean_i [ ratio_m_1 + inv_ratio_m_1 ]
    # This is the Schulman 2020 low-variance estimator for χ²(π_θ ∥ π_ref).
    # ------------------------------------------------------------------
    chi2 = (ratio_m_1 + inv_ratio_m_1).mean()

    # ------------------------------------------------------------------
    # Step 5: monitoring metrics
    # ------------------------------------------------------------------
    metrics = {
        "actor/chi2_value": chi2.detach().item(),
        "actor/chi2_ratio_clipfrac": (
            (log_ratio >= log_ratio_clip).float().mean().detach().item()
        ),
        "actor/chi2_inv_ratio_clipfrac": (
            (log_ratio <= -log_ratio_clip).float().mean().detach().item()
        ),
    }

    return chi2, metrics


# ---------------------------------------------------------------------------
# Main entry-point — replacement for DataParallelPPOActor.update_policy
# ---------------------------------------------------------------------------

def chi2_update_policy(self, data: DataProto) -> dict:
    """Chi-squared-regularized policy update.

    Replaces ``DataParallelPPOActor.update_policy`` when
    ``self.config.use_chi_square_loss`` is ``True``.

    This is a single-pass update that mirrors the original
    ``update_policy`` exactly, with one addition: after the KL penalty
    block (if any), a chi-squared divergence penalty is added to
    ``policy_loss`` before ``backward()``.

    Wandb metrics (under ``actor/`` and ``timing_s/``):
    - ``actor/chi2_value`` — raw chi2 estimate before sqrt
    - ``actor/chi2_loss`` — final penalty term (after sqrt × loss_scale_factor)
    - ``actor/chi2_coef`` — the λ coefficient
    - ``actor/chi2_ratio_clipfrac`` — fraction of samples clipped at +10
    - ``actor/chi2_inv_ratio_clipfrac`` — fraction of samples clipped at −10
    - ``timing_s/chi2_t`` — wall time for chi2 computation block
    """
    global _original_update_policy

    # ---- guard: chi2 disabled → fall back to original -----------------------
    _ensure_chi2_config(self.config)
    if not self.config.use_chi_square_loss:
        return _original_update_policy(self, data)

    # ====================================================================
    # Setup (mirrors original update_policy lines 372–418)
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
    # chi2 requires ref_log_prob — always select it
    if self.config.use_chi_square_loss:
        select_keys.append("ref_log_prob")
    if "rollout_is_weights" in data.batch.keys():
        select_keys.append("rollout_is_weights")
    if "rollout_log_probs" in data.batch.keys():
        select_keys.append("rollout_log_probs")

    has_mm = "multi_modal_inputs" in data.non_tensor_batch.keys()
    non_tensor_select = ["multi_modal_inputs"] if has_mm else []
    data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select)

    mini_batches = data.split(self.config.ppo_mini_batch_size)
    on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

    # Cached config values
    loss_agg_mode = self.config.loss_agg_mode
    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
    entropy_coeff = self.config.entropy_coeff
    chi2_coef = self.config.chi_square_loss_coef

    metrics: dict = {}

    # ====================================================================
    # Training loop (ppo_epochs × mini_batches)
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

            self.actor_optimizer.zero_grad()

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                micro_batch_metrics: dict = {}
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                response_mask = model_inputs["response_mask"]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                if self.config.use_dynamic_bsz:
                    loss_scale_factor = (
                        response_mask.shape[0] / self.config.ppo_mini_batch_size
                    )
                else:
                    loss_scale_factor = 1.0 / self.gradient_accumulation

                # ---- forward --------------------------------------------------
                calculate_entropy = entropy_coeff != 0
                entropy, log_prob = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )

                # ---- old_log_prob (three modes, matching original) ------------
                if (
                    hasattr(self.config, "use_rollout_log_probs")
                    and self.config.use_rollout_log_probs
                ):
                    old_log_prob = model_inputs["old_log_probs"]
                elif on_policy:
                    old_log_prob = log_prob.detach()
                else:
                    old_log_prob = model_inputs["old_log_probs"]

                # ---- policy loss ----------------------------------------------
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
                micro_batch_metrics.update(pg_metrics)

                # ---- rollout correction metrics (bypass mode) -----------------
                rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                    from verl.trainer.ppo.rollout_corr_helper import (
                        compute_rollout_corr_metrics_from_logprobs,
                    )

                    rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=response_mask,
                    )
                    micro_batch_metrics.update(rollout_corr_metrics)

                # ---- entropy --------------------------------------------------
                if entropy_coeff != 0:
                    entropy_loss = agg_loss(
                        loss_mat=entropy,
                        loss_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                    )
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss

                # ================================================================
                #  CHI-SQUARED DIVERGENCE REGULARIZATION
                #
                #  Paper Appendix D.2 / official code:
                #    loss += λ · sqrt( mean_i [π_θ/π_ref + π_ref/π_θ − 2] )
                #
                #  sqrt is applied per the official code (`--use_sqrt_chi2`
                #  flag and ``torch.sqrt(chi2.clip(min=1e-4))`` in
                #  PPOWithRefPolicyConfig.loss()).
                # ================================================================
                if self.config.use_chi_square_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    t0 = time.time()
                    chi2, chi2_metrics = compute_chi_square_loss(
                        log_prob=log_prob,
                        ref_log_prob=ref_log_prob,
                        response_mask=response_mask,
                    )
                    chi2_penalty = torch.sqrt(chi2.clamp(min=_CHI2_CLAMP_MIN))
                    policy_loss = policy_loss + chi2_penalty * chi2_coef
                    t_chi2 = time.time() - t0

                    micro_batch_metrics.update(chi2_metrics)
                    micro_batch_metrics["actor/chi2_loss"] = (
                        chi2_penalty.detach().item() * loss_scale_factor
                    )
                    micro_batch_metrics["actor/chi2_coef"] = chi2_coef
                    micro_batch_metrics["timing_s/chi2_t"] = t_chi2

                # ---- backward -------------------------------------------------
                if self.config.use_dynamic_bsz:
                    loss = policy_loss * loss_scale_factor
                else:
                    loss = policy_loss * loss_scale_factor
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_batch_metrics["actor/pg_loss"] = (
                    pg_loss.detach().item() * loss_scale_factor
                )
                append_to_dict(metrics, micro_batch_metrics)

            # ---- optimizer step -------------------------------------------
            grad_norm = self._optimizer_step()
            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)

    self.actor_optimizer.zero_grad()
    return metrics
