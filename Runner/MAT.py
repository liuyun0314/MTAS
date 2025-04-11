import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from Runner.util import check, init
from Runner.transformer_act import discrete_autoregreesive_act
from Runner.transformer_act import discrete_parallel_act
from Runner.transformer_act import continuous_autoregreesive_act
from Runner.transformer_act import continuous_parallel_act

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        # Masked Multi-Head Attention + Add & Norm
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep

class Decoder(nn.Module):

    def __init__(self, obs_dim, reasoning_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.reasoning_dim = reasoning_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())  # include a linnear layer and a GELU activation function
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, self.reasoning_dim)))


    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit

class RoleActor(nn.Module):
    def __init__(self, n_agnt, obs_dim, action_dim, hidden_dim):
        super(RoleActor, self).__init__()
        self.role = nn.Parameter(torch.randn(n_agnt, 1))
        self.n_agent = n_agnt
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.ModuleList()
        for n in range(self.n_agent):
            # if n == 3:
            #     # agent for sequening (5 sequencing rules)
            #     action_dim = 5
            # else:
            #     # agents for routing (4 routing rules)
            #     action_dim = 4
            actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                  init_(nn.Linear(obs_dim, self.hidden_dim), activate=True), nn.GELU(), nn.LayerNorm(self.hidden_dim),
                                  init_(nn.Linear(self.hidden_dim, self.hidden_dim), activate=True), nn.GELU(), nn.LayerNorm(self.hidden_dim),
                                  init_(nn.Linear(self.hidden_dim, action_dim)))
            self.mlp.append(actor)

    def forward(self, obs, n):
        logit = self.mlp[n](obs)

        # logit = []
        # for n in range(self.n_agent):
        #     input_o = obs[:, n, :]
        #     logit_n = self.mlp[n](input_o)
        #     # logit_n = self.mlp[n](obs[:, n, :])    # obs: (batch, n_agent, obs_dim)
        #     logit.append(logit_n)
        # logit = torch.stack(logit, dim=1)
        return logit

class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, reasoning_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        ### add by ly
        self.state_dim = state_dim
        #####

        # state unused
        # state_dim = 37

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, reasoning_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        ## add by ly
        self.role_actor = RoleActor(self.n_agent, reasoning_dim, action_dim, n_embd)
        ####
        self.to(device)

    def forward(self, state, obs, action, available_actions):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)
        available_actions = torch.tensor([[[1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                           [1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                           [1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 1., 1., 1., 1., 1.]]])

        # state unused
        # ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
        # state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if torch.isnan(obs_rep).any():
            print("action_logit has nan")
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, self.role_actor, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, self.role_actor, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
        # state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        available_actions = torch.tensor([[[1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                           [1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                           [1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 1., 1., 1., 1., 1.]]])
        available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, self.role_actor, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, self.role_actor, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
        state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot
class Critic(nn.Module):
    def __init__(self, state_dim, obs_dim, hidden_dim, action_dim, n_block, n_embd, n_head, n_agent, encode_state, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.n_block = n_block
        self.n_embd = n_embd
        self.n_head = n_head
        self.mlp = nn.ModuleList()
        self.device = device
        self.encode_state = encode_state
        self.encoder = Encoder(self.state_dim, self.obs_dim, self.n_block, self.n_embd, self.n_head, self.n_agent, self.encode_state)
        for i in range(self.n_agent):
            critic = nn.Sequential(nn.LayerNorm(obs_dim),
                                  init_(nn.Linear(obs_dim, self.hidden_dim), activate=True), nn.GELU(),
                                  nn.LayerNorm(self.hidden_dim),
                                  init_(nn.Linear(self.hidden_dim, 1)))
            self.mlp.append(critic)
        self.to(device)

    def get_values(self, state, obs):
        # state unused
        ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
        state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)    # obs_rep -> tensor(1, 4, 128)
        return v_tot



