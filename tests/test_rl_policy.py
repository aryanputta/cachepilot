from cachepilot.policy.rl_policy import AdmissionPolicy, SchedulerState, TinyMLP


def test_admission_policy_decision_exposes_probability():
    policy = AdmissionPolicy(model=TinyMLP(seed=0), warmup_n=0)
    state = SchedulerState(
        vram_util=0.8,
        queue_depth=4,
        max_queue=16,
        prompt_len=512,
        max_prompt_len=4096,
        priority=2,
        est_gen_len=128,
        max_gen_len=512,
        eviction_rate=0.3,
        time_since_eviction=2.0,
    )
    decision = policy.decide(state)
    assert 0.0 <= decision.probability <= 1.0
    assert decision.features.shape == (7,)


def test_policy_gradient_step_changes_weights():
    model = TinyMLP(seed=0)
    before = model.W2.copy()
    state = SchedulerState(
        vram_util=0.8,
        queue_depth=4,
        max_queue=16,
        prompt_len=512,
        max_prompt_len=4096,
        priority=2,
        est_gen_len=128,
        max_gen_len=512,
        eviction_rate=0.3,
        time_since_eviction=2.0,
    )
    policy = AdmissionPolicy(model=model, warmup_n=0)
    decision = policy.decide(state)
    model.policy_gradient_step(decision.features, action=True, advantage=1.5, lr=1e-2)
    assert not (before == model.W2).all()
