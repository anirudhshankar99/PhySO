import torch
import numpy as np


def compute_rtgs(rewards, done_index, gamma):
    # rewards: (T, B) with only terminal step having non-zero reward
    # done_index: (B,) integer index of terminal step per sequence

    T, B = rewards.shape
    rtgs = torch.zeros_like(rewards)
    # Build a done mask per time-step: done_mask[t, b] == 1 if t == done_index[b]
    done_mask = torch.zeros_like(rewards, dtype=torch.bool)
    # done_mask[done_index(i), i] = True for all i
    done_mask[done_index, torch.arange(B)] = True # Boolean of shape (T, B,) for sequences that are completed

    G = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        # If terminal at t, reset return to immediate reward (no bootstrapping past terminal)
        G = torch.where(done_mask[t], rewards[t], rewards[t] + gamma * G)
        rtgs[t] = G

    return rtgs

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage
    
    m = torch.min(ratio*advantage, clipped)
    logratio = log_prob - old_log_prob
    approx_kl = ((ratio - 1) - logratio).mean()
    clipfracs = [((ratio - 1.0).abs() > eps).float().mean().item()]
    return -m, approx_kl, clipfracs

def loss_func(values, update_logprobs, advantages, rtgs, rollout_logprobs, eps):
    """
    Loss function for reinforcing symbolic programs.
    Parameters
    ----------

    Returns
    -------

    """
    actor_loss, _, clipfracs = policy_loss(rollout_logprobs, update_logprobs, advantages, eps)
    critic_loss = values - rtgs

    return actor_loss, clipfracs, critic_loss