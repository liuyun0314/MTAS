import torch
import numpy as np
import torch.nn.functional as F


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        # self.algo = args.algorithm_name
        self.num_agents = num_agents
        # self.env_name = env_name

        obs_shape = args.obs_dim  # 一个智能体的观察空间维度
        share_obs_shape = args.share_obs_dim  # 所有智能体的共享观察空间维度
        # if type(obs_shape[-1]) == list:
        #     obs_shape = obs_shape[:1]
        #
        # if type(share_obs_shape[-1]) == list:
        #     share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, share_obs_shape),
                                  dtype=np.float32)
        # self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, share_obs_shape),
        #                           dtype=np.float32)
        self.obs = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, obs_shape), dtype=np.float32)

        act_shape = args.n_action
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_shape),
                                        dtype=np.float32)
        # if act_space.__class__.__name__ == 'Discrete':
        #     self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, len(act_space)),
        #                                      dtype=np.float32)
        # else:
        #     self.available_actions = None

        # act_shape = args.n_actions

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        # self.bad_masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1))
        self.active_masks = np.ones_like(self.masks)
        # self.active_masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1))

        self.step = 0

    # def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
    #            value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
    # def insert(self, share_obs, obs, actions, action_log_probs,
    #                value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions):
    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards, available_actions):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        # self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        # self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.available_actions[self.step + 1] = available_actions.copy()

        # self.masks[self.step + 1] = masks.copy()
        # if bad_masks is not None:
        #     self.bad_masks[self.step + 1] = bad_masks.copy()
        # if active_masks is not None:
        #     self.active_masks[self.step + 1] = active_masks.copy()
        # if available_actions is not None:
        #     self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    # def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
    #                  value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
    def chooseinsert(self, share_obs, obs, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        # self.rnn_states[self.step + 1] = rnn_states.copy()
        # self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        # self.rnn_states[0] = self.rnn_states[-1].copy()
        # self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        # self.rnn_states[0] = self.rnn_states[-1].copy()
        # self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                    self.value_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                # if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                #     gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                # if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                #     gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (   # assert: 断言，如果后面的条件为假，则抛出异常
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()    # generate a random permutation of 0 to batch_size-1 and convert to numpy array
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        # rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        # rnn_states = rnn_states[rows, cols]
        # rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        # rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            # rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            # rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            # yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            #       value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            #       adv_targ, available_actions_batch
            yield share_obs_batch, obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch


class ReplayBuffer_obj:
    def __init__(self, batch_size):
        self.global_batch_size = batch_size
        self.s = []
        self.a_r = np.zeros((self.global_batch_size, 1))
        self.ar_logprob = np.zeros((self.global_batch_size, 1))
        self.r = np.zeros((self.global_batch_size, 1))
        # self.s_ = np.zeros((args.batch_size, dim))
        self.s_ = []
        self.dw = np.zeros((self.global_batch_size, 1))
        self.done = np.zeros((self.global_batch_size, 1))
        self.count = 0

    def store(self, s, a_r, ar_logprob, r, s_, dw, done):
        # self.s[self.count] = s
        self.s.append(s)
        self.a_r[self.count] = a_r
        self.ar_logprob[self.count] = ar_logprob
        self.r[self.count] = r
        self.s_.append(s_)
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def clear(self):
        self.s = []
        self.a_r = np.zeros((self.global_batch_size, 1))
        self.ar_logprob = np.zeros((self.global_batch_size, 1))
        self.r = np.zeros((self.global_batch_size, 1))
        # self.s_ = np.zeros((args.batch_size, dim))
        self.s_ = []
        self.dw = np.zeros((self.global_batch_size, 1))
        self.done = np.zeros((self.global_batch_size, 1))
        self.count = 0

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a_r = torch.tensor(self.a_r, dtype=torch.long)
        ar_logprob = torch.tensor(self.ar_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a_r, ar_logprob, r, s_, dw, done