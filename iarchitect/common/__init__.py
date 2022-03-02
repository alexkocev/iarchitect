def show_policy_behaviour(environment,policy,max_iter):
    time_step = environment.current_time_step()
    iter_ = 0.0
    results = [(None,None,time_step.observation.numpy())]
    while not time_step.is_last() and iter_<=max_iter:
        iter_+=1
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        results.append((action_step.action.numpy(),time_step.reward.numpy(),time_step.observation.numpy()))
    return results