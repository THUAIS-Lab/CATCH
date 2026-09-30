def rank_breakpoint(rank=0):
    import ray
    runtime_env_vars = ray.get_runtime_context().runtime_env['env_vars']
    if runtime_env_vars.get('RAY_DEBUG', 0) and runtime_env_vars.get('RAY_LOCAL_RANK', '0') == str(rank):
        breakpoint()