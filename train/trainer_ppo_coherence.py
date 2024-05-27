"""
PPO algorithm
Reference: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/
"""
from tkinter import W
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from time import time
import logging
from utils.C_GCN import C_GCN
from utils.utils import *
from models.environment_coherence_model import EnvCoherenceModel
from models.policy_model import MLPActorCritic



class PPOBuffer(object):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantage of state-action pairs
    """

    def __init__(self, config, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.cfg = config
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.act_attr_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.logp_attr_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.logit_buf = np.zeros(combined_shape(size, self.cfg.n_categories), dtype=np.float32)
        self.obj_logit_buf = np.zeros(combined_shape(size, self.cfg.n_categories), dtype=np.float32) 
        self.obj_attr_logit_buf = np.zeros(combined_shape(size, self.cfg.n_property), dtype=np.float32) 
        self.txt_obj = [] 
        self.txt_attr = []       
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, act_attr, rew, val, logp, logp_attr, logit, obj_logit, obj_attr_logit, txt_obj, txt_attr):
        """
        Append one timestep of agent-environment interaction to the buffer
        :param obs:
        :param act:
        :param rew:
        :param val:
        :param logp:
        :return:
        """
        assert self.ptr < self.max_size
        if self.cfg.cuda:
            obs = obs.detach().cpu().numpy()
        else:
            obs = obs.numpy()
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.act_attr_buf[self.ptr] = act_attr
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.logp_attr_buf[self.ptr] = logp_attr
        self.logit_buf[self.ptr] = logit
        self.obj_logit_buf[self.ptr] = obj_logit 
        self.obj_attr_logit_buf[self.ptr] = obj_attr_logit
        self.txt_obj.append(txt_obj)
        self.txt_attr.append(txt_attr)
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        :param last_val:
        :return:
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        print('rew: ', rews)
        print('vals: ', vals)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        print('adv_buf: ', self.adv_buf[path_slice])

        # the next line computes reward-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        print('ret_buf: ', self.ret_buf[path_slice])

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        :return:
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        lengths = [len(self.txt_obj[i]) for i in range(len(self.txt_obj))]

        lengths_attr = [len(self.txt_attr[i]) for i in range(len(self.txt_attr))]
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    act_attr=self.act_attr_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf,
                    logp_attr=self.logp_attr_buf,
                    logit=self.logit_buf,
                    obj_logit=self.obj_logit_buf,
                    obj_attr_logit=self.obj_attr_logit_buf,
                    txt_obj=self.txt_obj,
                    txt_attr=self.txt_attr)
        self.txt_attr, self.txt_obj = [], []
        return_dict = {}
        for k, v in data.items():
            if k != 'txt_obj' and k != 'txt_attr':
                return_dict[k] = torch.as_tensor(v, dtype=torch.float32)
            else:
                return_dict[k] = v
        return return_dict, lengths, lengths_attr


class PPOTrainer(object):
    def __init__(self, config, split='train'):
        self.cfg = config

        # init env and actor-critic
        self.env = EnvCoherenceModel(config, split)
        self.ac = MLPActorCritic(config)
        self.obj_gcn = C_GCN(num_classes=self.cfg.n_categories-1, \
                             in_channel=300, \
                             embed_size=300, \
                             t=0.0005, \
                             adj_array=self.env.object_y_do_x[1:,1:], \
                             glove_array=self.env.glove_word_vector[1:,:], \
                             opt=self.cfg)
        self.attr_gcn = C_GCN(num_classes=self.cfg.n_property-1, \
                             in_channel=300, \
                             embed_size=300, \
                             t=0.005, \
                             adj_array=self.env.attr_y_do_x[1:,1:], \
                             glove_array=self.env.glove_word_attribute_vector[1:,:], \
                             opt=self.cfg)
        if self.cfg.cuda:
            self.ac = self.ac.cuda()
            self.obj_gcn = self.obj_gcn.cuda()
            self.attr_gcn = self.attr_gcn.cuda()

        self.word_stat = np.load(self.cfg.data_dir + '/caches/vg_word_stat.npy')

        # count parameters
        pi_counts = count_var(self.ac.pi)
        v_counts = count_var(self.ac.v)
        logging.info('Number of Actor-Critic parameters: {}'.format(pi_counts + v_counts))

        # init PPOBuffer
        if self.cfg.use_glove:
            self.buf = PPOBuffer(config, self.cfg.n_feature_dim + self.cfg.n_categories + self.cfg.glove_vector_dim, \
                        self.cfg.ppo_num_actions, self.cfg.ppo_update_steps, self.cfg.ppo_gamma, self.cfg.ppo_lambda)
        else:
            self.buf = PPOBuffer(config, self.cfg.n_feature_dim + self.cfg.n_categories, \
                        self.cfg.ppo_num_actions, self.cfg.ppo_update_steps, self.cfg.ppo_gamma, self.cfg.ppo_lambda)


        print('-------------------')
        print('All parameters')
        for name, param in self.ac.named_parameters():
            print(name, param.size())
        print('-------------------')
        print('Trainable parameters')
        for name, param in self.ac.named_parameters():
            if param.requires_grad:
                print(name, param.size())

        # init optimizer
        if self.cfg.use_gcn:
            param = self.get_config_optm('pi')
            param1 = self.get_config_optm('v')
            param2 = self.get_config_optm('gcn')

            self.sup_optimizer = optim.Adam(param, lr=self.cfg.ppo_sup_lr)
            self.pi_optimizer = optim.Adam(param, lr=self.cfg.ppo_pi_lr)
            self.v_optimizer = optim.Adam(param1, lr=self.cfg.ppo_v_lr)
            self.obj_attr_optimizer = optim.Adam(param2, lr=self.cfg.gcn_lr)

        else:
            self.sup_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.cfg.ppo_sup_lr)
            self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.cfg.ppo_pi_lr)
            self.v_optimizer = optim.Adam(self.ac.v.parameters(), lr=self.cfg.ppo_v_lr)

        if self.cfg.ppo_pretrained is not None:
            self.load_pretrained_net(self.cfg.ppo_pretrained)


    def get_config_optm(self, type):
        if type == 'pi':
            return [
                    {'params': self.ac.pi.parameters(), 'lr': self.cfg.ppo_sup_lr},
            ]
        elif type == 'v':
            return [
                    {'params': self.ac.v.parameters(), 'lr': self.cfg.ppo_v_lr},
            ]
        
        else:
            return [
                    {'params': self.obj_gcn.parameters(), 'lr':self.cfg.gcn_lr},
                    {'params': self.attr_gcn.parameters(), 'lr':self.cfg.gcn_lr}
            ]
    def compute_loss_sup(self, data, len_obj, len_attr):
        if self.cfg.use_gcn:
            obs_raw, logits, obj_logits, attr_logits, txt_obs, txt_attr = data['obs'], data['logit'], data['obj_logit'], data['obj_attr_logit'], data['txt_obj'], data['txt_attr']
            obs = obs_raw.clone()
            self.env.glove_word_vector = torch.cat((torch.zeros(size=(1,300)), self.obj_gcn().cpu()), dim = 0)
            self.env.glove_word_attribute_vector = torch.cat((torch.zeros(size=(1,300)),self.attr_gcn().cpu()), dim = 0)
            txt_obs = [ind for inds in txt_obs for ind in inds]
            txt_attr = [ind for inds in txt_attr for ind in inds]
            obj_vector_split = self.env.glove_word_vector[txt_obs].split(len_obj)
            attr_vector_split = self.env.glove_word_attribute_vector[txt_attr].split(len_attr)
            
            obj_vector_split_mean = []
            for obj_vector_single in obj_vector_split:
                if obj_vector_single.size() == (0,300):
                    tt = torch.zeros(size=(1,300))
                else:
                    tt = torch.mean(obj_vector_single, dim=0, keepdim=True)
                obj_vector_split_mean.append(tt)           
            new_obj_vector = torch.cat(obj_vector_split_mean, dim=0)

            attr_vector_split_mean = []
            for attr_vector_single in attr_vector_split:
                if attr_vector_single.size() == (0,300):
                    tt = torch.zeros(size=(1,300))
                else:
                    tt = torch.mean(attr_vector_single, dim=0, keepdim=True)
                attr_vector_split_mean.append(tt)           
            new_attr_vector = torch.cat(attr_vector_split_mean, dim=0)
            
            new_glove_vector = ( new_obj_vector + new_attr_vector ) / 2
            obs[:, -300:] = new_glove_vector
        else:
            obs, logits, obj_logits, attr_logits = data['obs'], data['logit'], data['obj_logit'], data['obj_attr_logit']

        if self.cfg.cuda:
            obs, logits, obj_logits, attr_logits = obs.cuda(), logits.cuda(), obj_logits.cuda(), attr_logits.cuda()

        prob1, prob2 = self.ac.pi.get_prob(obs)
        criterion = nn.MSELoss()
        loss = criterion(prob1, logits)
        loss1 = criterion(prob1, obj_logits)
        loss2 = criterion(prob2, attr_logits)
        return self.cfg.ppo_coef_logit * (loss + loss1 + loss2)

    def compute_loss_pi(self, data, len_obj, len_attr):
        if self.cfg.use_gcn:
            obs_raw, act, act_attr, adv, logp_old, logp_attr_old, txt_obs, txt_attr = data['obs'], data['act'], \
                                                                                  data['act_attr'], data['adv'], \
                                                                                  data['logp'], data['logp_attr'], \
                                                                                  data['txt_obj'], data['txt_attr']
            obs = obs_raw.clone()
            self.env.glove_word_vector = torch.cat((torch.zeros(size=(1,300)), self.obj_gcn().cpu()),dim = 0)
            self.env.glove_word_attribute_vector = torch.cat((torch.zeros(size=(1,300)),self.attr_gcn().cpu()),dim = 0)
            txt_obs = [ind for inds in txt_obs for ind in inds]
            txt_attr = [ind for inds in txt_attr for ind in inds]
            obj_vector_split = self.env.glove_word_vector[txt_obs].split(len_obj)
            attr_vector_split = self.env.glove_word_attribute_vector[txt_attr].split(len_attr)
            
            obj_vector_split_mean = []
            for obj_vector_single in obj_vector_split:
                if obj_vector_single.size() == (0,300):
                    tt = torch.zeros(size=(1,300))
                else:
                    tt = torch.mean(obj_vector_single, dim=0, keepdim=True)
                obj_vector_split_mean.append(tt)           
            new_obj_vector = torch.cat(obj_vector_split_mean, dim=0)

            attr_vector_split_mean = []
            for attr_vector_single in attr_vector_split:
                if attr_vector_single.size() == (0,300):
                    tt = torch.zeros(size=(1,300))
                else:
                    tt = torch.mean(attr_vector_single, dim=0, keepdim=True)
                attr_vector_split_mean.append(tt)           
            new_attr_vector = torch.cat(attr_vector_split_mean, dim=0)
            
            new_glove_vector = ( new_obj_vector + new_attr_vector ) / 2
            obs[:, -300:] = new_glove_vector
            
        else:
            obs, act, act_attr, adv, logp_old, logp_attr_old = data['obs'], data['act'], data['act_attr'], data['adv'], data['logp'], data['logp_attr']
        act = act.type(torch.int32)
        act_attr = act_attr.type(torch.int32)
        
        if self.cfg.cuda:
            obs, act, adv, logp_old, act_attr, logp_attr_old = obs.cuda(), act.cuda(), adv.cuda(), logp_old.cuda(), act_attr.cuda(), logp_attr_old.cuda()
        act_2 = [act, act_attr]
        pi, logp, pi_attr, logp_attr = self.ac.pi(obs, act_2)
        logp = logp.transpose(0, 1)
        logp_attr = logp_attr.transpose(0, 1)
        ratio0 = torch.exp(logp - logp_old)
        ratio1 = torch.exp(logp_attr - logp_attr_old)
        ratio = ( ratio0 + ratio1 ) / 2
        clip_adv = torch.clamp(ratio, 1 - self.cfg.ppo_clip_ratio, 1 + self.cfg.ppo_clip_ratio) * adv.unsqueeze(
            1).repeat(1, ratio.shape[1])
        ent = (pi.entropy().mean() + pi_attr.entropy().mean()) / 2
        loss_pi = - self.cfg.ppo_coef_ratio * (torch.min(ratio * adv.unsqueeze(1).repeat(1, ratio.shape[1]),
                                                         clip_adv)).mean() - self.cfg.ppo_coef_ent * ent
        # useful extra info
        approx_kl = ((logp_old - logp).mean().item() + (logp_attr_old - logp_attr).mean().item()) / 2
        ent = ent.item()
        clipped = ratio.gt(1 + self.cfg.ppo_clip_ratio) | ratio.lt(1 - self.cfg.ppo_clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data, len_obj, len_attr):
        if self.cfg.use_gcn:
            obs_raw, ret, txt_obs, txt_attr = data['obs'], data['ret'], data['txt_obj'], data['txt_attr']
            obs = obs_raw.clone()
            self.env.glove_word_vector = torch.cat((torch.zeros(size=(1,300)), self.obj_gcn().cpu()),dim = 0)
            self.env.glove_word_attribute_vector = torch.cat((torch.zeros(size=(1,300)),self.attr_gcn().cpu()),dim = 0)
            txt_obs = [ind for inds in txt_obs for ind in inds]
            txt_attr = [ind for inds in txt_attr for ind in inds]
            obj_vector_split = self.env.glove_word_vector[txt_obs].split(len_obj)
            attr_vector_split = self.env.glove_word_attribute_vector[txt_attr].split(len_attr)
            
            obj_vector_split_mean = []
            for obj_vector_single in obj_vector_split:
                if obj_vector_single.size() == (0,300):
                    tt = torch.zeros(size=(1,300))
                else:
                    tt = torch.mean(obj_vector_single, dim=0, keepdim=True)
                obj_vector_split_mean.append(tt)           
            new_obj_vector = torch.cat(obj_vector_split_mean, dim=0)

            attr_vector_split_mean = []
            for attr_vector_single in attr_vector_split:
                if attr_vector_single.size() == (0,300):
                    tt = torch.zeros(size=(1,300))
                else:
                    tt = torch.mean(attr_vector_single, dim=0, keepdim=True)
                attr_vector_split_mean.append(tt)           
            new_attr_vector = torch.cat(attr_vector_split_mean, dim=0)
            
            new_glove_vector = ( new_obj_vector + new_attr_vector ) / 2
            obs[:, -300:] = new_glove_vector
            
        else:
            obs, ret = data['obs'], data['ret']
        if self.cfg.cuda:
            obs, ret = obs.cuda(), ret.cuda()
        value = self.ac.v(obs)
        return self.cfg.ppo_coef_value * ((value - ret) ** 2).mean()

    def train_epoch(self):
        data, len_obj, len_attr = self.buf.get()
        pi_loss_old, pi_info_old = self.compute_loss_pi(data, len_obj, len_attr)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = self.compute_loss_v(data, len_obj, len_attr).item()

        # train policy with multiple steps of supervised learning
        for i in range(self.cfg.ppo_train_sup_iters):
            self.sup_optimizer.zero_grad()
            sup_loss = self.compute_loss_sup(data, len_obj, len_attr)
            sup_loss.backward()
            self.sup_optimizer.step()
            if i % 10 == 0:
                logging.info('PPO PI update at step {} Sup Loss: {:.6f}'.format(i, sup_loss.item()))
        
        if self.cfg.use_gcn:
            for i in range(self.cfg.gcn_train_iters):
                self.obj_attr_optimizer.zero_grad()
                sup_loss = self.compute_loss_sup(data, len_obj, len_attr)
                sup_loss.backward()
                self.obj_attr_optimizer.step()
                if i % 10 == 0:
                    logging.info('PPO GCN update at step {} Sup Loss: {:.6f}'.format(i, sup_loss.item()))
        
        # train policy with multiple steps of gradient descent
        for i in range(self.cfg.ppo_train_pi_iters):
            self.pi_optimizer.zero_grad()
            pi_loss, pi_info = self.compute_loss_pi(data, len_obj, len_attr)
            kl = pi_info['kl']
            if kl > 1.5 * self.cfg.ppo_target_kl:
                logging.info('Early stopping at step {} due to reaching max kl.'.format(i))
                break
            pi_loss.backward()
            self.pi_optimizer.step()
            if i % 10 == 0:
                logging.info('PPO PI update at step {} PI Loss: {:.6f}, KL: {:.6f}'.format(i, pi_loss.item(), kl))


        # train value with multiple steps of gradient descent
        for i in range(self.cfg.ppo_train_v_iters):
            self.v_optimizer.zero_grad()
            v_loss = self.compute_loss_v(data,len_obj, len_attr)
            v_loss.backward()
            self.v_optimizer.step()
            if i % 10 == 0:
                logging.info('PPO Value update at step {} Value Loss: {:.6f}'.format(i, v_loss.item()))


        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info_old['cf']
        logging.info(
            'Loss PI: {:.6f}, Loss V: {:.6f}, KL: {:.6f}, Entropy: {:.6f}, ClipFrac: {:.6f} '
            'DeltaLossPi: {:.6f}, DeltaLossV: {:.6f}'.format(pi_loss_old, v_loss_old, kl, ent, cf,
                                                             pi_loss.item() - pi_loss_old,
                                                             v_loss.item() - v_loss_old))

    def get_start_obs(self):
        if self.cfg.use_glove and self.cfg.use_gcn:
            self.env.glove_word_vector = torch.cat((torch.zeros(size=(1,300)), self.obj_gcn().cpu()),dim = 0)
            self.env.glove_word_attribute_vector = torch.cat((torch.zeros(size=(1,300)),self.attr_gcn().cpu()),dim = 0)
        index = random.randint(0, len(self.env.loaddb) - 1)
        scene = self.env.db.scenedb[index]
        logging.info('Image %s' % scene['image_index'])
        all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]
        all_captions = [x['caption'] for x in all_meta_regions]
        all_captions = np.random.permutation(all_captions)
        txt = list(all_captions[:self.cfg.train_turns])

        txt_object_index = [self.env.db.class_to_ind[n] for n in txt[0].split(" ") if n in self.env.db.class_to_ind.keys()]
        txt_attribute_index = [self.env.db.attribute_to_ind[n] for n in txt[0].split(" ") if n in self.env.db.attribute_to_ind.keys()]

        if len(txt_object_index) or len(txt_attribute_index):
            if len(txt_object_index) > 0  and len(txt_attribute_index) == 0:
                glove_vector = torch.mean(self.env.glove_word_vector[txt_object_index], dim=0, keepdim=True)
            elif len(txt_object_index) and len(txt_attribute_index):
                glove_vector = torch.mean(self.env.glove_word_vector[txt_object_index], dim=0, keepdim=True)
                attribute_glove_vector = torch.mean(self.env.glove_word_attribute_vector[txt_attribute_index], dim=0, keepdim=True)
                new_vector = torch.cat((glove_vector, attribute_glove_vector), dim=0)
                glove_vector = torch.mean(new_vector,dim=0, keepdim=True)
            elif len(txt_object_index) == 0 and len(txt_attribute_index) > 0:
                glove_vector = torch.mean(self.env.glove_word_attribute_vector[txt_attribute_index], dim=0, keepdim=True)
        else:
            glove_vector = torch.zeros(size=(1,300))

        self.env.reset(txt, index)
        logging.info('first query:')
        for t in txt: 
            logging.info(t)
        if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
            r, inds, rank = self.env.retrieve_precomp(*self.env.tokenize())
        else:
            r, inds, rank = self.env.retrieve(*self.env.tokenize())
        logits_dis = torch.from_numpy(self.env.vg_all_logits_dis).unsqueeze(0)
        if self.cfg.use_glove:
            o = torch.cat([torch.mean(self.env.txt_feat, dim=0, keepdim=True),\
                                logits_dis.type_as(self.env.txt_feat), \
                                glove_vector.type_as(self.env.txt_feat)], dim=1)
        else:
             o = torch.cat([torch.mean(self.env.txt_feat, dim=0, keepdim=True),\
                                logits_dis.type_as(self.env.txt_feat)], dim=1)           
        return r, o, inds


    def train(self):
        start_time = time()
        timestep = 0
        propor_list, logits_word, logits_objects = [], [], []
        # main loop: collect experience in env and update/log each epoch
        for epoch in range(self.cfg.ppo_epochs):

            ep_ret, ep_len = 0, 0
            _, o, inds = self.get_start_obs()
            # obj_inds_label, attr_inds_label = self.get_one_hot(self.env.logit,self.env.attr_logit)
            
            for t in range(self.cfg.max_turns):
                if self.cfg.cuda:
                    o = o.cuda()
                a_obj, a_attr, v, logp_obj, logp_attr = self.ac.step(o)

                # obj_inds_pre, attr_inds_pre = self.get_one_hot(a_obj,a_attr)

                obj_logit, obj_attr_logit = self.env.gen_obj_logit() 

                txt = self.env.txt
                word_inds = []
                for tx in txt:
                    tokens = [w for w in word_tokenize(tx)]
                    word_inds.extend([self.env.db.lang_vocab(w) for w in tokens])
                word_inds = list(set(word_inds))
                logit = np.sum(self.word_stat[word_inds], axis=0)
                logit = logit / np.sum(logit)

                next_o, r, d, _, _, txt_obj, txt_attr = self.env.step(a_obj, a_attr, 'train')

                if self.cfg.use_gcn:
                    self.env.glove_word_vector = torch.cat((torch.zeros(size=(1,300)), self.obj_gcn().cpu()),dim = 0)
                    self.env.glove_word_attribute_vector = torch.cat((torch.zeros(size=(1,300)),self.attr_gcn().cpu()),dim = 0)

                logging.info(
                    'reward now: {:.6f}'.format(r))


                ep_ret += r
                ep_len += 1

                # save and log

                self.buf.store(o, a_obj, a_attr, r, v, logp_obj, logp_attr, logit, obj_logit, obj_attr_logit, txt_obj, txt_attr)
                timestep += 1

                # update obs
                o = next_o

                timeout = t == self.cfg.max_turns - 1
                logging.info('turn: {}'.format(t))

                if timeout:
                    _, _, v, _, _ = self.ac.step(o)
                    logging.info('Start a finish path')
                    self.buf.finish_path(v)

                start_update = timestep == self.cfg.ppo_update_steps
                if start_update:
                    timestep = 0
                    logging.info('Start a update')
                    self.train_epoch()
                    logging.info(
                        'Value: {:.3f}, NowEpisodeReward: {:.3f}, NowEpisodeLen: {}'.format(v, ep_ret, ep_len))

            if (epoch % self.cfg.ppo_save_freq == 0) or (epoch == self.cfg.ppo_epochs - 1):
                self.save_checkpoint(epoch, self.cfg.exp_name)

            logging.info('Episode: {}, TotalInteract: {}, Time: {:.3f}'.format(epoch, (epoch + 1) * self.cfg.max_turns,
                                                                               time() - start_time))

    def test(self):

        all_retrieve_inds = []
        db_length = len(self.env.db.scenedb)
        self.env.glove_word_vector = torch.cat((torch.zeros(size=(1,300)), self.obj_gcn().detach().cpu()), dim = 0)
        self.env.glove_word_attribute_vector = torch.cat((torch.zeros(size=(1,300)),self.attr_gcn().detach().cpu()), dim = 0)     
        self.ac.eval()
        self.obj_gcn.eval()
        self.attr_gcn.eval()

        for idx in range(db_length):

            self.env.num_attr_obj = 0
            retrieve_inds = []
            # get test text
            test_scene = self.env.db.scenedb[idx]
            logging.info('Image %s' % test_scene['image_index'])
            all_meta_regions = [test_scene['regions'][x] for x in sorted(list(test_scene['regions'].keys()))]
            all_captions = [x['caption'] for x in all_meta_regions]
            txt = all_captions[:self.cfg.test_turns]

            logging.info('first query:')   
            for t in txt: 
                logging.info(t)

            txt_object_index = list(set([self.env.db.class_to_ind[n] for m in txt for n in m.split(" ") if n in self.env.db.class_to_ind.keys()]))
            txt_attribute_index = list(set([self.env.db.attribute_to_ind[n] for m in txt for n in m.split(" ") if n in self.env.db.attribute_to_ind.keys()]))
        
            if len(txt_object_index) or len(txt_attribute_index):
                if len(txt_object_index) > 0  and len(txt_attribute_index) == 0:
                    glove_vector = torch.mean(self.env.glove_word_vector[txt_object_index].detach(), dim=0, keepdim=True)
                elif len(txt_object_index) and len(txt_attribute_index):
                    glove_vector = torch.mean(self.env.glove_word_vector[txt_object_index].detach(), dim=0, keepdim=True)
                    attribute_glove_vector = torch.mean(self.env.glove_word_attribute_vector[txt_attribute_index].detach(), dim=0, keepdim=True)
                    new_vector = torch.cat((glove_vector, attribute_glove_vector), dim=0)
                    glove_vector = torch.mean(new_vector,dim=0, keepdim=True)
                elif len(txt_object_index) == 0 and len(txt_attribute_index) > 0:
                    glove_vector = torch.mean(self.env.glove_word_attribute_vector[txt_attribute_index].detach(), dim=0, keepdim=True)
            else:
                glove_vector = torch.zeros(size=(1,300))
            
            self.env.reset(txt, idx)
            if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
                _, inds, rank = self.env.retrieve_precomp(*self.env.tokenize())
            else:
                _, inds, rank = self.env.retrieve(*self.env.tokenize())
            
            # get top10 image
            top_inds = inds[:10]
            img_inds = [self.env.db.scenedb[idx]['image_index'] for idx in top_inds]
            logging.info('Top image: ')
            for i in img_inds:
                logging.info(i)

            logits_dis = torch.from_numpy(self.env.vg_all_logits_dis).unsqueeze(0)
            if self.cfg.use_glove:
                o = torch.cat([torch.mean(self.env.txt_feat, dim=0, keepdim=True), \
                                logits_dis.type_as(self.env.txt_feat), \
                                glove_vector.type_as(self.env.txt_feat)], dim=1)
            else:
                o = torch.cat([torch.mean(self.env.txt_feat, dim=0, keepdim=True), \
                                logits_dis.type_as(self.env.txt_feat)], dim=1)


            retrieve_inds.append(inds)
            for t in range(self.cfg.max_turns):
                if rank[0] < self.cfg.ppo_stop_rank:
                    break
                if self.cfg.cuda:
                    o = o.cuda()
                dis, dis_attr = self.ac.pi.get_prob(o)
                dis = dis.squeeze(1).detach()
                dis_attr = dis_attr.squeeze(1).detach()
                a = torch.argsort(dis, descending=True).squeeze().cpu().numpy()[t * self.cfg.ppo_num_actions: (t+1) * self.cfg.ppo_num_actions]

                a_attr = torch.argsort(dis_attr, descending=True).squeeze().cpu().numpy()[t * self.cfg.ppo_num_actions: (t+1) * self.cfg.ppo_num_actions]

                next_o, r, d, inds, logit, _, _ = self.env.step(a, a_attr, 'test')
                retrieve_inds.append(inds)

                if d:
                    break

                # update obs
                o = next_o
            all_retrieve_inds.append(retrieve_inds)
            logging.info('Get %d th retrieve result' % idx)
        
        ranks = []
        for idx in range(len(all_retrieve_inds)):
            inds = np.array(all_retrieve_inds[idx])
            rank = np.where(inds == idx)[1]
            ranks.append(rank)
        metric_record = ""
        for t in range(self.cfg.max_turns + 1):
            
            logging.info("Get %d th turn result" % t)
            res = []
            for idx, rank in enumerate(ranks):
                if t < len(rank):
                    res.append(rank[t])
                else:
                    res.append(rank[-1])
            res = np.array(res)
            r1 = 100.0 * len(np.where(res < 1)[0]) / len(res)
            r5 = 100.0 * len(np.where(res < 5)[0]) / len(res)
            r10 = 100.0 * len(np.where(res < 10)[0]) / len(res)
            r20 = 100.0 * len(np.where(res < 20)[0]) / len(res)
            r50 = 100.0 * len(np.where(res < 50)[0]) / len(res)
            r100 = 100.0 * len(np.where(res < 100)[0]) / len(res)
            medr = np.floor(np.median(res)) + 1
            meanr = res.mean() + 1

            logging.info(
                "Text to image: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % (r1, r5, r10, r20, r50, r100, medr, meanr))
            
            str1 = "%.1f %.1f %.1f %.1f" % (r1, r5, r10, meanr)
            metric_record = metric_record + "\n" + str1
        
        return metric_record


    def save_checkpoint(self, epoch, model_name):
        print('Saving checkpoint...')
        checkpoint_dir = osp.join(self.cfg.model_dir, 'snapshots')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
            'state_dict': self.ac.state_dict(),
            'state_dict_obj_gcn':self.obj_gcn.state_dict(),
            'state_dict_attr_gcn':self.attr_gcn.state_dict(),
            'pi_optimizer': self.pi_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
        }
        torch.save(states, osp.join(checkpoint_dir, str(epoch) + '.pkl'))

    def load_pretrained_net(self, pretrained_path):
        assert (osp.exists(pretrained_path))
        states = torch.load(pretrained_path)
        self.ac.load_state_dict(states['state_dict'], strict=True)
        self.obj_gcn.load_state_dict(states['state_dict_obj_gcn'], strict=True)
        self.attr_gcn.load_state_dict(states['state_dict_attr_gcn'], strict=True)
        self.pi_optimizer.load_state_dict(states['pi_optimizer'])
        self.v_optimizer.load_state_dict(states['v_optimizer'])
        self.epoch = states['epoch']
