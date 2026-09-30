from rllm.trainer.verl.agent_ppo_trainer import (
    _summarize_reward_hack_metrics,
    _summarize_rule_monitor_metrics,
)


def test_summarize_reward_hack_metrics_omits_pass_rate_metrics():
    reward_outputs = [
        {
            "reward": 1.0,
            "metadata": {
                "passed_tests": 2,
                "total_tests": 2,
            },
        },
        {
            "reward": 0.0,
            "metadata": {
                "passed_tests": 1,
                "total_tests": 4,
            },
        },
    ]

    metrics = _summarize_reward_hack_metrics(reward_outputs, prefix="critic")

    assert "critic/mean_pass_rate" not in metrics


def test_summarize_reward_hack_metrics_reports_hack_metrics_when_available():
    reward_outputs = [
        {
            "reward": 1.0,
            "metadata": {
                "passed_tests": 1,
                "total_tests": 1,
                "reward_wo_hack": 0.0,
                "reward_w_hack": 1.0,
                "reward_w_hack_time_s": 0.3,
                "reward_wo_hack_time_s": 0.5,
                "is_hack": True,
                "trivial_hack": False,
                "nontrivial_hack": True,
            },
        },
        {
            "reward": 1.0,
            "metadata": {
                "passed_tests": 1,
                "total_tests": 1,
                "reward_wo_hack": 1.0,
                "reward_w_hack": 1.0,
                "reward_w_hack_time_s": 0.7,
                "reward_wo_hack_time_s": 0.9,
                "is_hack": False,
                "trivial_hack": False,
                "nontrivial_hack": False,
            },
        },
    ]

    metrics = _summarize_reward_hack_metrics(reward_outputs, prefix="val")

    assert metrics["hack/val/reward_wo_hack"] == 0.5
    assert metrics["hack/val/hack_rate"] == 0.5
    assert metrics["hack/val/trivial_hack_rate"] == 0.0
    assert metrics["hack/val/nontrivial_hack_rate"] == 0.5
    assert "timing_s/reward_w_hack" not in metrics
    assert "timing_s/reward_wo_hack" not in metrics


def test_summarize_reward_hack_metrics_reports_train_timing_means_when_available():
    reward_outputs = [
        {
            "reward": 1.0,
            "metadata": {
                "passed_tests": 1,
                "total_tests": 1,
                "reward_wo_hack": 0.0,
                "reward_w_hack": 1.0,
                "reward_w_hack_time_s": 0.3,
                "reward_wo_hack_time_s": 0.5,
                "is_hack": True,
                "trivial_hack": True,
                "nontrivial_hack": False,
            },
        },
        {
            "reward": 0.0,
            "metadata": {
                "passed_tests": 0,
                "total_tests": 1,
                "reward_wo_hack": 0.0,
                "reward_w_hack": 0.0,
                "reward_w_hack_time_s": 0.7,
                "reward_wo_hack_time_s": 0.9,
                "is_hack": False,
                "trivial_hack": False,
                "nontrivial_hack": False,
            },
        },
    ]

    metrics = _summarize_reward_hack_metrics(reward_outputs, prefix="critic")

    assert metrics["timing_s/reward_w_hack"] == 0.5
    assert metrics["timing_s/reward_w_hack_min"] == 0.3
    assert metrics["timing_s/reward_w_hack_max"] == 0.7
    assert metrics["timing_s/reward_wo_hack"] == 0.7
    assert metrics["timing_s/reward_wo_hack_min"] == 0.5
    assert metrics["timing_s/reward_wo_hack_max"] == 0.9
    assert metrics["hack/train/hack_rate"] == 0.5
    assert metrics["hack/train/trivial_hack_rate"] == 0.5
    assert metrics["hack/train/nontrivial_hack_rate"] == 0.0


def test_summarize_reward_hack_metrics_reports_critic_binary_pass_means():
    reward_outputs = [
        {
            "reward": 1.0,
            "metadata": {
                "all_passed_easy": True,
                "all_passed_hard": False,
                "all_passed": False,
                "reward_cache_bonus": 0.2,
            },
        },
        {
            "reward": 1.0,
            "metadata": {
                "all_passed_easy": 1,
                "all_passed_hard": 1,
                "all_passed": 1,
                "reward_cache_bonus": 0.1,
            },
        },
        {
            "reward": 0.0,
            "metadata": {
                "all_passed_easy": 0,
                "all_passed_hard": 0,
                "all_passed": 0,
                "reward_cache_bonus": 0.0,
            },
        },
    ]

    metrics = _summarize_reward_hack_metrics(reward_outputs, prefix="critic")

    assert metrics["critic/all_passed_easy/mean"] == 2 / 3
    assert metrics["critic/all_passed_hard/mean"] == 1 / 3
    assert metrics["critic/all_passed/mean"] == 1 / 3
    assert metrics["critic/with_cache_bonus/mean"] == 2 / 3


# ---------------------------------------------------------------------------
# _summarize_rule_monitor_metrics
# ---------------------------------------------------------------------------


def test_rule_monitor_empty():
    assert _summarize_rule_monitor_metrics([], prefix="critic") == {}


def test_rule_monitor_hack_only():
    """Hack samples only: each rate uses total hacks as denominator."""
    reward_outputs = [
        {"metadata": {"hack_method": "eq"}},
        {"metadata": {"hack_method": "eq"}},
        {"metadata": {"hack_method": "exit0"}},
        {"metadata": {"hack_method": "unknown"}},
    ]
    metrics = _summarize_rule_monitor_metrics(reward_outputs, prefix="critic")

    # 4 hacks total: eq=2, exit0=1, unknown=1
    assert metrics["monitor/train/rule_eq_rate"] == 2 / 4
    assert metrics["monitor/train/rule_exit0_rate"] == 1 / 4
    assert metrics["monitor/train/rule_unknown_rate"] == 1 / 4
    assert metrics["monitor/train/rule_xfail_rate"] == 0.0
    # No normal_* keys
    assert "monitor/train/rule_normal_rate" not in metrics
    assert "monitor/train/rule_normal_eq_rate" not in metrics


def test_rule_monitor_normal_only():
    """Clean samples only: each rate uses total clean samples as denominator."""
    reward_outputs = [
        {"metadata": {"hack_method": "normal"}},
        {"metadata": {"hack_method": "normal"}},
        {"metadata": {"hack_method": "normal"}},
        {"metadata": {"hack_method": "normal_eq"}},
    ]
    metrics = _summarize_rule_monitor_metrics(reward_outputs, prefix="val")

    # 4 clean total: normal=3, normal_eq=1
    assert metrics["monitor/val/rule_normal_rate"] == 3 / 4
    assert metrics["monitor/val/rule_normal_eq_rate"] == 1 / 4
    assert metrics["monitor/val/rule_normal_xfail_rate"] == 0.0
    # No hack keys
    assert "monitor/val/rule_eq_rate" not in metrics
    assert "monitor/val/rule_exit0_rate" not in metrics


def test_rule_monitor_mixed_separate_denominators():
    """Hack and normal rates use independent denominators — they don't mix."""
    reward_outputs = [
        # 3 hacks
        {"metadata": {"hack_method": "eq"}},
        {"metadata": {"hack_method": "exit0"}},
        {"metadata": {"hack_method": "eq"}},
        # 5 clean
        {"metadata": {"hack_method": "normal"}},
        {"metadata": {"hack_method": "normal"}},
        {"metadata": {"hack_method": "normal"}},
        {"metadata": {"hack_method": "normal_eq"}},
        {"metadata": {"hack_method": "normal_xfail"}},
    ]
    metrics = _summarize_rule_monitor_metrics(reward_outputs, prefix="critic")

    # Hack side: denominator = 3
    assert metrics["monitor/train/rule_eq_rate"] == 2 / 3
    assert metrics["monitor/train/rule_exit0_rate"] == 1 / 3
    assert metrics["monitor/train/rule_xfail_rate"] == 0.0

    # Normal side: denominator = 5
    assert metrics["monitor/train/rule_normal_rate"] == 3 / 5
    assert metrics["monitor/train/rule_normal_eq_rate"] == 1 / 5
    assert metrics["monitor/train/rule_normal_xfail_rate"] == 1 / 5


def test_rule_monitor_default_unknown():
    """Missing hack_method defaults to 'unknown' on the hack side."""
    reward_outputs = [
        {"metadata": {}},
        {"metadata": {"hack_method": "exit0"}},
    ]
    metrics = _summarize_rule_monitor_metrics(reward_outputs, prefix="critic")
    assert metrics["monitor/train/rule_unknown_rate"] == 0.5
    assert metrics["monitor/train/rule_exit0_rate"] == 0.5
