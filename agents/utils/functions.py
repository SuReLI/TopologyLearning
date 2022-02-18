import torch
from gym.spaces import Box, Discrete


def forward_kl_div(means_1, stds_1, means_2, stds_2):
    return (1 / 2) * (torch.log(stds_2 ** 2 / stds_1 ** 2) - 1 + (stds_1 ** 2 + (means_1 - means_2) ** 2) / stds_2 ** 2)


def symmetric_kl_div(means_1, stds_1, means_2, stds_2):
    kl_div = forward_kl_div(means_1, stds_1, means_2, stds_2) + forward_kl_div(means_2, stds_2, means_1, stds_1)
    return torch.mean(kl_div)


def kl_div_prior(p_means, p_stds, symmetric=True):
    zeros = torch.zeros(*p_stds.shape)
    ones = torch.ones(*p_means.shape)
    return symmetric_kl_div(p_means, p_stds, zeros, ones)


def get_state_size(state_space):
    """
    Return the size of a state using the given state space.
    """
    if isinstance(state_space, Box):
        if len(state_space.shape) > 1:
            raise NotImplementedError("Operations are not implemented to deals with matrix states.")
        return state_space.shape[0]
    elif isinstance(state_space, Discrete):
        return state_space.shape.n


def get_nb_actions(action_space):
    if isinstance(action_space, Box):
        total_nb_actions = None
        for nb_actions in action_space.shape:
            if total_nb_actions is None:
                total_nb_actions = nb_actions
            else:
                total_nb_actions *= nb_actions
        return total_nb_actions
    else:
        return 1
