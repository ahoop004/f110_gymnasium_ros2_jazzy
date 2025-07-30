# rl_training/utils/rewards.py

def simple_pass_reward(ego_pose, opp_pose, ego_collision, step_penalty=-0.01):
    """
    Reward for staying in front of the opponent.
    Args:
        ego_pose: [x, y, theta] for ego
        opp_pose: [x, y, theta] for opponent
        ego_collision: 0 or 1
        step_penalty: float, small negative reward per step

    Returns:
        float: reward value
    """
    ego_x = ego_pose[0]
    opp_x = opp_pose[0]

    if ego_collision:
        return -10.0

    reward = step_penalty
    if ego_x > opp_x:
        reward += 1.0
    return reward
