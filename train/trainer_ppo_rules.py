import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from time import time
from torch.distributions.categorical import Categorical
from scipy.special import softmax
import logging
from collections import defaultdict
from scipy.special import softmax
from utils.utils import *
from models.environment_rules_model import EnvRulesModel


class PPOTrainer(object):
    def __init__(self, config, split='train'):
        self.cfg = config

        # init env and actor-critic
        self.env = EnvRulesModel(config, split)

        if self.cfg.ppo_rule == 'coherence':
            self.stat_coherence()


    def stat_coherence(self):
        vg_img_logits = self.env.vg_img_logits
        obj_pred_row, obj_pred_col = np.where(vg_img_logits > 0.9)
        logits_coherence = np.zeros((self.cfg.n_categories, self.cfg.n_categories))

        idx2objs = {}
        for i, idx in enumerate(obj_pred_row):
            if idx not in idx2objs.keys():
                idx2objs[idx] = [obj_pred_col[i]]
            else:
                idx2objs[idx].append(obj_pred_col[i])

        for _, objs in idx2objs.items():
            for obj in objs:
                other = list(set(objs) - {obj})
                logits_coherence[obj][other] += 1

        self.logits_coherence = logits_coherence / (np.sum(logits_coherence, axis=-1) + 1e-6)


        vg_img_attr_logits = self.env.vg_img_property_logits
        attr_pred_row, attr_pred_col = np.where(vg_img_attr_logits > 0.9)
        attr_logits_coherence = np.zeros((self.cfg.n_property, self.cfg.n_property))

        idx2attrs = {}
        for i, idx in enumerate(attr_pred_row):
            if idx not in idx2attrs.keys():
                idx2objs[idx] = [attr_pred_col[i]]
            else:
                idx2objs[idx].append(attr_pred_col[i])

        for _, attrs in idx2attrs.items():
            for attr in attrs:
                other = list(set(attrs) - {attr})
                attr_logits_coherence[attr][other] += 1

        self.attr_logits_coherence = attr_logits_coherence / (np.sum(attr_logits_coherence, axis=-1) + 1e-6)

    def get_action(self, turn, inds):

        if self.cfg.ppo_rule == 'score_obj':
            logits1 = self.env.vg_img_logits[inds]
            logits_dis = softmax(logits1, axis=0)
            logits_dis = np.where(logits_dis>0.5, 1, 0)
            propor = np.mean(logits_dis[:2500,:], axis=0)
            dis_obj = Categorical(logits=torch.from_numpy(propor))
        elif self.cfg.ppo_rule == 'query_attr_sim':
            obj_sim = self.env.obj_sim
            dis_obj = Categorical(logits=obj_sim)
        elif self.cfg.ppo_rule == 'coherence':
            obj_sim = self.env.obj_sim
            top_obj = torch.argmax(obj_sim, dim=-1).item()
            logit_coherence = self.logits_coherence[top_obj]
            dis_obj = Categorical(logits=torch.from_numpy(logit_coherence))

        else:
            raise Exception('Please choose a right rule. Now is {}'.format(self.cfg.ppo_rule))
                
        a_obj = torch.argsort(dis_obj.probs, descending=True)[turn * self.cfg.ppo_num_actions:(turn + 1) * self.cfg.ppo_num_actions]

        if self.cfg.cuda:
            a_obj = a_obj.squeeze().cpu()

        return a_obj.numpy(), []

    def test(self):
        all_retrieve_inds = []
        db_length = len(self.env.db.scenedb)
        for idx in range(db_length):
            retrieve_inds = []
            # get test text
            test_scene = self.env.db.scenedb[idx]
            all_meta_regions = [test_scene['regions'][x] for x in sorted(list(test_scene['regions'].keys()))]
            all_captions = [x['caption'] for x in all_meta_regions]
            txt = all_captions[:self.cfg.test_turns]
            self.env.reset(txt, idx)
            if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
                _, inds, rank = self.env.retrieve_precomp(*self.env.tokenize())
            else:
                _, inds, rank = self.env.retrieve(*self.env.tokenize())
            retrieve_inds.append(inds)
            for t in range(self.cfg.max_turns):
                if rank[0] < self.cfg.ppo_stop_rank:
                    break
                a_obj, a_attr = self.get_action(t, inds)
                next_o, r, d, inds, logit = self.env.step(a_obj, a_attr)
                retrieve_inds.append(inds)

                if d:
                    break

            all_retrieve_inds.append(retrieve_inds)
            logging.info('Get %d th retrieve result' % idx)

        ranks = []
        for idx in range(len(all_retrieve_inds)):
            inds = np.array(all_retrieve_inds[idx])
            rank = np.where(inds == idx)[1]
            ranks.append(rank)

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


    def save_checkpoint(self, epoch, model_name):
        print('Saving checkpoint...')
        checkpoint_dir = osp.join(self.cfg.model_dir, 'snapshots')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
        }
        torch.save(states, osp.join(checkpoint_dir, str(epoch) + '.pkl'))

    def load_pretrained_net(self, pretrained_path):
        assert (osp.exists(pretrained_path))
        states = torch.load(pretrained_path)
        self.epoch = states['epoch']
