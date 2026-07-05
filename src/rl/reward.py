def compute_reward(
    infection_ratio,
    connectivity_ratio,
    pruning_cost,
    alpha=0.5,
    beta=0.3,
    gamma=0.2,
):
    """
    Compute the RL reward.

    Parameters
    ----------
    infection_ratio : float
    connectivity_ratio : float
    pruning_cost : float

    Returns
    -------
    float
        Reward value.
    """

    reward = (
        alpha * (1 - infection_ratio)
        + beta * connectivity_ratio
        - gamma * pruning_cost
    )

    return reward