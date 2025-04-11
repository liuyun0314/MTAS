import torch
import numpy as np
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discrete_autoregreesive_act(decoder, roleActor, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    available_actions = torch.tensor([[[1., 1., 1., 1., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 1., 1., 1., 1., 1.]]])
    # ### add by ly
    # logit = decoder(shifted_action, obs_rep, obs)
    # logit = roleActor(logit)
    # ############

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        ### add by ly
        logit_i = roleActor(logit, i)
        ####
        # logit_i = logit[:, i, :]
        logit_i[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit_i)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log

def discrete_parallel_act(decoder, roleActor, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions):

    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]

    logit = decoder(shifted_action, obs_rep, obs)
    action_logit = torch.zeros(batch_size, n_agent, action_dim).to(**tpdv)
    ### add by ly
    for i in range(n_agent):
        action_logit[:, i, :] = roleActor(logit[:, i, :], i)
    available_actions = available_actions.expand(action_logit.size())
    action_logit[available_actions == 0] = -1e10
    ####
    if torch.isnan(action_logit).any():
        print("action_logit has nan")
    distri = Categorical(logits=action_logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy

def decentralization_value(encoder, roleCritic, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv, available_actions):
    pass


def continuous_autoregreesive_act(decoder, roleActor, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        ### add by ly
        act_mean = roleActor(act_mean)[:, i, :]
        ####
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, roleActor, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    ### add by ly
    act_mean= roleActor(act_mean)
    # for i in range(n_agent):
    #     act_mean[:, i, :] = roleActor(act_mean[:, i, :], i)
    ####
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
