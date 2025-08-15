import torch
import numpy as np


def safe_cross_entropy(p, logq, dim=-1):
    safe_logq = torch.where(p == 0, torch.ones_like(logq), logq)
    return -torch.sum(p * safe_logq, dim=dim)

def compute_rtgs(rewards, done_index, gamma):
    rtgs = torch.zeros_like(rewards)
    for t_ in reversed(range(done_index)):
        if t_ == done_index - 1: rtgs[t_] = rewards[t_]
        else: rtgs[t_] = rewards[t_] + rtgs[t_ + 1] * gamma
    return rtgs

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage.unsqueeze(-1)
    
    m = torch.min(ratio*advantage.unsqueeze(-1), clipped)
    logratio = log_prob - old_log_prob
    approx_kl = ((ratio - 1) - logratio).mean()
    clipfracs = [((ratio - 1.0).abs() > eps).float().mean().item()]
    return -m, approx_kl, clipfracs

def loss_func(values, update_probs, actions, advantages, rtgs, rollout_logprobs, ideal_probs_train, lengths, entropy_weight, eps):
    """
    Loss function for reinforcing symbolic programs.
    Parameters
    ----------

    Returns
    -------

    """
    update_dist = torch.distributions.Categorical(update_probs)
    update_logprobs = update_dist.log_prob(actions)
    actor_loss, _, __ = policy_loss(rollout_logprobs, update_logprobs, advantages, eps)
    actor_loss = actor_loss.mean()
    critic_loss = (values - rtgs).pow(2).mean()

    (max_time_step, n_train, n_choices,) = ideal_probs_train.shape
    mask_length_np = np.tile(np.arange(0, max_time_step), (n_train, 1)  # (n_train, max_time_step,)
                             ).astype(int) < np.tile(lengths, (max_time_step, 1)).transpose()
    mask_length_np = mask_length_np.transpose().astype(float)  # (max_time_step, n_train,)
    mask_length = torch.tensor(mask_length_np, requires_grad=False)  # (max_time_step, n_train,)

    # Sum over action dim
    entropy_per_step = safe_cross_entropy(update_probs, update_logprobs, dim=2)  # (max_time_step, n_train,)
    # Sum over sequence dim
    entropy = torch.sum(entropy_per_step * mask_length, dim=0)  # (n_train,)
    entropy_loss = -entropy_weight * torch.mean(entropy)

    return actor_loss, entropy_loss, critic_loss