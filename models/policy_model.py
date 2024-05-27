import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations and
        optionlly compute the log-likelihood of given action under
        those distributions
        :param obs:
        :param act:
        :return:
        """
        pi1, pi2 = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi1, act[0])
            logp_b = self._log_prob_from_distribution(pi2, act[1])
        return pi1, logp_a, pi2, logp_b



class MLPPolicyNet(Actor):
    # def __init__(self, config, object_num, attribute_num, object_dict, attribute_dict):
    def __init__(self, config):
        super(MLPPolicyNet, self).__init__()
        self.cfg = config

        if self.cfg.use_glove:
            self.net = nn.Sequential(
                nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories + self.cfg.glove_vector_dim, self.cfg.n_categories),
                nn.Tanh(),
                nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
                nn.Tanh(),
                nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
                nn.Tanh(),
                # nn.Softmax(dim=-1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories, self.cfg.n_categories),
                nn.Tanh(),
                nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
                nn.Tanh(),
                nn.Linear(self.cfg.n_categories, self.cfg.n_categories),
                nn.Tanh(),
                # nn.Softmax(dim=-1)
            )

        self.net_obj = nn.Linear(self.cfg.n_categories, self.cfg.n_categories)
        self.net_attr = nn.Linear(self.cfg.n_categories, self.cfg.n_property)
        self.soft = nn.Softmax(dim=-1)


    def get_logit(self, obs):
        """
        compute logit of given obs
        :param obs: [buffer_len, n_featuren_dim + n_categories], tensor
        :return:
        """
        out_bas = self.net(obs)
        obj_logit = self.soft(self.net_obj(out_bas))
        pre_logit = self.soft(self.net_attr(out_bas))
        logit1 = Categorical(probs=obj_logit)
        logit2 = Categorical(probs=pre_logit)
        return logit1, logit2

    def get_prob(self, obs):
        """
        compute prob of given obs
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        out_base = self.net(obs)
        prob1 = self.soft(self.net_obj(out_base))
        prob2 = self.soft(self.net_attr(out_base))
        return prob1, prob2



    def _distribution(self, obs):
        """
        compute distribution of actions
        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.get_logit(obs)

    def _log_prob_from_distribution(self, pi, act):
        """
        compute log prob of action
        :param pi: torch distribution
        :param act: [buffer_len, num_actions], actions
        :return:
        """
        # -> (num_actions, buffer_len)
        act = act.transpose(0, 1)
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(self, config):
        super(MLPCritic, self).__init__()
        self.cfg = config

        if self.cfg.use_glove:
            self.v_net = nn.Sequential(
                nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories + self.cfg.glove_vector_dim, self.cfg.n_feature_dim),
                nn.Tanh(),
                nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
                nn.Tanh(),
                nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
                nn.Tanh(),
                nn.Linear(self.cfg.n_feature_dim, 1)
            )
        else:
            self.v_net = nn.Sequential(
                nn.Linear(self.cfg.n_feature_dim + self.cfg.n_categories, self.cfg.n_feature_dim),
                nn.Tanh(),
                nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
                nn.Tanh(),
                nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
                nn.Tanh(),
                nn.Linear(self.cfg.n_feature_dim, 1)
            )



    def forward(self, obs):
        """
        predict value of observation
        :param obs: [batchszie, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.v_net(obs).squeeze()


class MLPActorCritic(nn.Module):
    # def __init__(self, config, object_num, attribute_num, object_dict, attribute_dict):
    def __init__(self, config):
        super(MLPActorCritic, self).__init__()
        self.cfg = config
        self.pi = MLPPolicyNet(config)
        self.v = MLPCritic(config)

    def step(self, obs):
        """

        :param obs:  [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        with torch.no_grad():
            pi_obj, pi_pre = self.pi._distribution(obs)
            a_obj = pi_obj.sample([self.cfg.ppo_num_actions])
            a_pre = pi_pre.sample([self.cfg.ppo_num_actions])
            log_a_obj = self.pi._log_prob_from_distribution(pi_obj, a_obj)
            log_a_pre = self.pi._log_prob_from_distribution(pi_pre, a_pre)
            v = self.v(obs)
            if self.cfg.cuda:
                a_obj, a_pre, v, log_a_obj, log_a_pre = a_obj.squeeze().cpu(), a_pre.squeeze().cpu(), v.cpu(), log_a_obj.squeeze().cpu(), log_a_pre.squeeze().cpu()
        return a_obj.numpy(), a_pre.numpy(), v.numpy(), log_a_obj.numpy(), log_a_pre.numpy()

    def act(self, obs):
        """

        :param obs: [buffer_len, n_feature_dim + n_categories], tensor
        :return:
        """
        return self.step(obs)[0]

