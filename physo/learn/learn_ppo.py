import torch
import numpy as np
import time

# Internal imports
from . import loss_ppo

def learner ( actor,
             actor_optimiser,
             critic,
             critic_optimiser,
             n_epochs,
             num_update_interations,
             batch_reseter,
             risk_factor,
             gamma,
             entropy_weight,
             eps,
             verbose = True,
             stop_reward = 1.,
             stop_after_n_epochs = 50,
             max_n_evaluations   = None,
             run_logger     = None,
             run_visualiser = None,
            ):
    """
    Trains model to generate symbolic programs satisfying a reward by reinforcing on best candidates at each epoch.
    Parameters
    ----------
    model : torch.nn.Module
        Differentiable RNN cell.
    optimizer : torch.optim
        Optimizer to use.
    n_epochs : int
        Number of epochs.
    batch_reseter : callable
        Function returning a new empty physym.batch.Batch.
    risk_factor : float
        Fraction between 0 and 1 of elite programs to reinforce on.
    gamma_decay : float
        Weight of power law to use along program length: gamma_decay**t where t is the step in the sequence in the loss
        function (gamma_decay < 1 gives more important to first tokens and gamma_decay > 1 gives more weight to last
        tokens).
    entropy_weight : float
        Weight to give to entropy part of the loss.
    verbose : int, optional
        If verbose = False or 0, print nothing, if True or 1, prints learning time, if > 1 print epochs progression.
    stop_reward : float, optional
        Early stops if stop_reward is reached by a program (= 1 by default), use stop_reward = (1-1e-5) when using free
        constants.
    stop_after_n_epochs : int, optional
        Number of additional epochs to do after early stop condition is reached.
    max_n_evaluations : int or None, optional
        Maximum number of unique expression evaluations allowed (for benchmarking purposes). Immediately terminates
        the symbolic regression task if the limit is about to be reached. The parameter max_n_evaluations is distinct
        from batch_size * n_epochs because batch_size * n_epochs sets the number of expressions generated but a lot of
        these are not evaluated because they have inconsistent units.
    run_logger : object or None, optional
        Custom run logger to use having a run_logger.log method taking as args (epoch, batch, model, rewards, keep,
        notkept, loss_val).
    run_visualiser : object or None, optional
        Custom run visualiser to use having a run_visualiser.visualise method taking as args (run_logger, batch).
    Returns
    -------
    hall_of_fame_R, hall_of_fame : list of float, list of physym.program.Program
        hall_of_fame : history of overall best programs found.
        hall_of_fame_R : Corresponding reward values.
        Use hall_of_fame[-1] to access best model found.
    """
    t000 = time.perf_counter()

    # Basic logs
    overall_max_R_history = []
    hall_of_fame          = []
    # Nb. of expressions evaluated
    n_evaluated           = 0
    
    for epoch in range (n_epochs):

        if verbose>1: print("Epoch %i/%i"%(epoch, n_epochs))

        # -------------------------------------------------
        # --------------------- INIT  ---------------------
        # -------------------------------------------------

        # Reset new batch (embedding reset)
        batch = batch_reseter()
        batch_size    = batch.batch_size
        max_time_step = batch.max_time_step
        input_size = batch.obs_size
        output_size = batch.n_choices

        # Initial RNN cell input
        actor_states = actor.get_zeros_initial_state(batch_size)  # (n_layers, 2, batch_size, hidden_size)
        critic_states = critic.get_zeros_initial_state(batch_size)  # (n_layers, 2, batch_size, hidden_size)

        # Optimizer reset

        # Candidates
        # logits        = []
        # actions       = []

        # Number of elite candidates to keep
        n_keep = int(risk_factor*batch_size)

        # -------------------------------------------------
        # -------------------- RNN RUN  -------------------
        # -------------------------------------------------
        observations = torch.zeros(max_time_step, batch_size, input_size)
        rollout_actions = torch.zeros(max_time_step, batch_size)
        rollout_logprobs = torch.zeros(max_time_step, batch_size)
        rollout_rewards = torch.zeros(max_time_step, batch_size)
        rollout_values = torch.zeros(max_time_step, batch_size)
        rollout_prior_logprobs = torch.zeros(max_time_step, batch_size, output_size)

        pre_rollout_actor_states = actor_states.clone()
        pre_rollout_critic_states = critic_states.clone()
        # RNN run
        for i in range (max_time_step):

            # ------------ OBSERVATIONS ------------
            environment_state = torch.tensor(batch.get_obs().astype(np.float32), requires_grad=False,) # (batch_size, obs_size)

            # ------------ MODEL ------------

            # Giving up-to-date observations
            with torch.no_grad():
                policy_logits, actor_states = actor(input_tensor = environment_state,
                                      states = actor_states)    # (batch_size, output_size)
                value, critic_states = critic(input_tensor = environment_state,
                                            states = critic_states)
            value = value.flatten()

            # Getting raw prob distribution for action n°i

            # ------------ PRIOR ------------

            prior_array = batch.prior().astype(np.float32)         # (batch_size, output_size)

            # 0 protection so there is always something to sample
            epsilon = 0 #1e-14 #1e0*np.finfo(np.float32).eps
            prior_array[prior_array==0] = epsilon
            is_able_to_sample = (prior_array.sum(axis=-1)>0.)      # (batch_size,)
            assert is_able_to_sample.all(), "Prior(s) make it impossible to successfully sample expression(s) as all " \
                                            "choosable tokens have 0 prob for %i/%i programs."%(is_able_to_sample.sum(), batch_size)

            # To log
            prior    = torch.tensor(prior_array, requires_grad=False) # (batch_size, output_size)
            logprior = torch.log(prior)                               # (batch_size, output_size)

            # ------------ SAMPLING ------------

            combined_logits  = policy_logits + logprior                              # (batch_size, output_size)
            dists = torch.distributions.Categorical(logits=combined_logits)
            action = dists.sample()
            logprob = dists.log_prob(action)

            # ------------ ACTION ------------

            # Saving action n°i
            rollout_logprobs[i] = logprob
            rollout_actions[i] = action
            observations[i] = environment_state
            rollout_values[i] = value
            rollout_prior_logprobs[i] = logprior

            batch.programs.append(action.detach().cpu().numpy())
        rollout_dones = torch.tensor(batch.programs.n_lengths - 1, dtype=torch.int)

        actor_states = pre_rollout_actor_states
        critic_states = pre_rollout_critic_states
        R = batch.get_rewards() # (B,)
        keep    = R.argsort()[::-1][0:n_keep].copy()                              # (n_keep,)
        notkept = R.argsort()[::-1][n_keep: ].copy()                              # (batch_size-n_keep,)
        zero_mask = R == 0
        R[zero_mask] = -0.1
        rollout_rewards[rollout_dones, torch.arange(batch_size)] = torch.tensor(R, dtype=torch.float32)

        rtgs = loss_ppo.compute_rtgs(rollout_rewards, rollout_dones, gamma)
        advantages = rtgs - rollout_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n_train = n_keep
        lengths = batch.programs.n_lengths[keep]
        mask_length_np = np.tile(np.arange(0, max_time_step), (n_train, 1)  # (n_train, max_time_step,)
                                ).astype(int) < np.tile(lengths, (max_time_step, 1)).transpose()
        mask_length_np = mask_length_np.transpose().astype(float)  # (max_time_step, n_train,)
        mask_length = torch.tensor(mask_length_np, requires_grad=False)  # (max_time_step, n_train,)

        actor_states_init, critic_states_init = actor_states.clone(), critic_states.clone()
        for _ in range(num_update_interations):
            update_logprobs = torch.zeros(max_time_step, batch_size)
            update_values = torch.zeros(max_time_step, batch_size)
            actor_states, critic_states = actor_states_init.clone(), critic_states_init.clone()
            for j in range(max_time_step):
                update_policy_logits, actor_states = actor(input_tensor = observations[j], states = actor_states)
                values, critic_states = critic(input_tensor = observations[j], states = critic_states)
                values = values.squeeze(-1)
                update_combined_logits = update_policy_logits + rollout_prior_logprobs[j]
                dists = torch.distributions.Categorical(logits=update_combined_logits)
                logprob = dists.log_prob(rollout_actions[j])
                update_logprobs[j] = logprob
                update_values[j] = values
            actor_loss, clipfracs, critic_loss = loss_ppo.loss_func (values=update_values[:, keep], 
                                        update_logprobs=update_logprobs[:, keep],
                                        advantages=advantages[:, keep], 
                                        rtgs=rtgs[:, keep], 
                                        rollout_logprobs=rollout_logprobs[:, keep],
                                        eps=eps)
            
            actor_loss = torch.mean(actor_loss * mask_length)
            critic_loss = (critic_loss * mask_length).pow(2).mean()
            actor_optimiser.zero_grad()
            actor_loss  .backward()
            actor_optimiser .step()
            critic_optimiser.zero_grad()
            critic_loss  .backward()
            critic_optimiser .step()

        # -------------------------------------------------
        # ----------------- LOGGING VALUES ----------------
        # -------------------------------------------------

        # Basic logging (necessary for early stopper)
        if epoch == 0:
            overall_max_R_history       = [R.max()]
            hall_of_fame                = [batch.programs.get_prog(R.argmax())]
        if epoch> 0:
            if R.max() > np.max(overall_max_R_history):
                overall_max_R_history.append(R.max())
                hall_of_fame.append(batch.programs.get_prog(R.argmax()))
            else:
                overall_max_R_history.append(overall_max_R_history[-1])

        # Custom logging
        if run_logger is not None:
            run_logger.log(epoch    = epoch,
                           batch    = batch,
                           model    = actor,
                           rewards  = R,
                           keep     = keep,
                           notkept  = notkept,
                           loss_val = actor_loss)

        # -------------------------------------------------
        # ----------------- VISUALISATION -----------------
        # -------------------------------------------------

        # Custom visualisation
        if run_visualiser is not None:
            run_visualiser.visualise(run_logger = run_logger, batch = batch)

        # -------------------------------------------------
        # ----------------- EARLY STOPPER -----------------
        # -------------------------------------------------
        early_stop_reward_eps = 2*np.finfo(np.float32).eps

        # If above stop_reward (+/- eps) stop after [stop_after_n_epochs] epochs.
        if (stop_reward - overall_max_R_history[-1]) <= early_stop_reward_eps:
            if stop_after_n_epochs == 0:
                try:
                    run_visualiser.save_visualisation()
                    run_visualiser.save_data()
                    run_visualiser.save_pareto_data()
                    run_visualiser.save_pareto_fig()
                except:
                    print("Unable to save last plots and data before early stopping.")
                break
            stop_after_n_epochs -= 1

        # -------------------------------------------------
        # ------------ MAX EVALUATIONS STOPPER ------------
        # -------------------------------------------------

        # Update nb. of evaluated programs
        n_evaluated += (R > 0.).sum()

        # If max_n_evaluations mode is used and we are one batch away from reaching the limit, stop.
        if (max_n_evaluations is not None) and (n_evaluated + batch_size > max_n_evaluations):
            try:
                run_visualiser.save_visualisation()
                run_visualiser.save_data()
                run_visualiser.save_pareto_data()
                run_visualiser.save_pareto_fig()
            except:
                print("Unable to save last plots and data before stopping due to max evaluation limit.")
            break

    t111 = time.perf_counter()
    if verbose:
        print("  -> Time = %f s"%(t111-t000))

    hall_of_fame_R = np.array(overall_max_R_history)
    return hall_of_fame_R, hall_of_fame