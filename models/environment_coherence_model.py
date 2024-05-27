import sys
from cv2 import log, sort

from matplotlib.pyplot import axis

sys.path.append('../')
import os.path as osp
import numpy as np
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from scipy.special import softmax

from models.text_image_matching_model import TextImageMatchingModel
from models.image_attribute_model import ImageAttributeModel
from train.trainer_txt_img_matching import TextImageMatchingTrainer
from train.trainer_img_attr import ImageAttributeTrainer
from utils.utils import *
from utils.config import get_test_config
from utils.vocab import Vocabulary
from datasets.vg import vg
from datasets.loader import region_loader, region_collate_fn


class EnvCoherenceModel(nn.Module):
    def __init__(self, config, split='train'):
        super(EnvCoherenceModel, self).__init__()
        self.cfg = config
        self.txt_img_matching_trainer = TextImageMatchingTrainer(config)
        self.set_no_grad(self.txt_img_matching_trainer.net)
        if self.cfg.vg_img_feature is None or self.cfg.vg_img_logits is None:
            self.img_attr_trainer = ImageAttributeTrainer(config)
            self.set_no_grad(self.img_attr_trainer.net)
        self.txt = None
        self.index = None
        self.mask = None
        self.txt_feat = None
        self.logit = None
        self.logit_return = None
        self.attr_logit = None
        self.attr_logit_return = None
        self.cache_dir = osp.abspath(osp.join(config.data_dir, 'caches'))
        
        if split == 'train':
            self.db = vg(config, 'train')
        elif split == 'val':
            self.db = vg(config, 'val')
        elif split == 'test':
            self.db = vg(config, 'test')
        else:
            raise Exception('Please choose a right split.')

        self.loaddb = region_loader(self.db)
        self.loader = DataLoader(self.loaddb, batch_size=self.cfg.batch_size, shuffle=False,
                                 num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)
        
        self.prepare_feature_logits(self.cfg)
        self.prepare_feature_logits_gt(self.cfg)

        self.glove_word_vector = torch.from_numpy(np.array(pickle.load(open(self.cfg.data_dir + '/caches/vg_objects_glove_word2vector_300.pkl', 'rb'))))
        self.glove_word_attribute_vector = torch.from_numpy(np.array(pickle.load(open(self.cfg.data_dir + '/caches/vg_attribute_glove_word2vector_300.pkl', 'rb'))))

        self.object_adj = pickle.load(open(self.cfg.data_dir + '/caches/vg_objects_num_adj_1601.pkl', 'rb'))['all_object_adj']
        self.object_num = pickle.load(open(self.cfg.data_dir + '/caches/vg_objects_num_adj_1601.pkl', 'rb'))['all_object_num'] + 1e-10

        self.attribute_adj = pickle.load(open(self.cfg.data_dir + '/caches/vg_attribute_num_adj_401.pkl', 'rb'))['all_attributes_adj']
        self.attribute_num = pickle.load(open(self.cfg.data_dir + '/caches/vg_attribute_num_adj_401.pkl', 'rb'))['all_attribute_num'] + 1e-10
        

        self.likelihood_object = torch.from_numpy(self.object_adj / self.object_num)
        self.likelihood_attribute = torch.from_numpy(self.attribute_adj / self.attribute_num)

        self.object_y_do_x = torch.from_numpy(pickle.load(open(self.cfg.data_dir + '/caches/vg_object_y_do_x.pkl', 'rb'))['object_y_do_x'])
        self.attr_y_do_x = torch.from_numpy(pickle.load(open(self.cfg.data_dir + '/caches/vg_attr_y_do_x.pkl', 'rb'))['attr_y_do_x'])
                                 

        self.obj_attr_stat = np.load(self.cfg.data_dir + '/caches/vg_scene_object_attr_stat.npy')

        self.obj_attr_dict = {}



    def set_no_grad(self, model):
        for param in model.parameters():
            param.required_grad = False

    def prepare_feature_logits(self, config):

        if config.vg_img_feature is not None:
            self.vg_img_feature = np.load(config.vg_img_feature)
        if self.cfg.vg_img_logits is not None:
            self.vg_img_logits = np.load(config.vg_img_logits)
            obj_pred = np.where(self.vg_img_logits > 0.9)[1]
            self.vg_all_logits_dis = np.zeros(config.n_categories)
            for obj in obj_pred:
                self.vg_all_logits_dis[obj] += 1
            self.vg_all_logits_dis = self.vg_all_logits_dis / np.sum(self.vg_all_logits_dis)
        
        if config.vg_attr_feature is not None:
            self.vg_attr_feature = np.load(config.vg_attr_feature)

        self.all_img_feats = torch.from_numpy(self.vg_img_feature)
        # self.all_img_attrs = torch.from_numpy(self.vg_img_logits)

        if self.cfg.cuda:
            # self.all_img_feats, self.all_img_attrs = self.all_img_feats.cuda(), self.all_img_attrs.cuda()
            self.all_img_feats = self.all_img_feats.cuda()

    def prepare_feature_logits_gt(self, config):

        self.logit_gts = []
        max_len = 0
        for scene in self.loaddb:
            logit_gt = scene[6]
            if len(logit_gt) > max_len:
                max_len = len(logit_gt)
            self.logit_gts.append(logit_gt)
        self.logits = np.zeros((len(self.loaddb), max_len), dtype=np.int) - 1
        for i, logit in enumerate(self.logit_gts):
            le = len(self.logit_gts[i])
            self.logits[i, :le] = self.logit_gts[i]

        # 3.5 property ground truth
        self.property_logit_gts = []
        max_len = 0
        for scene in self.loaddb:
            property_logit_gt = scene[-3]
            if len(property_logit_gt) > max_len:
                max_len = len(property_logit_gt)
            self.property_logit_gts.append(property_logit_gt)
        self.property_logits = np.zeros((len(self.loaddb), max_len), dtype=np.int) - 1
        for i, logit in enumerate(self.property_logit_gts):
            le = len(self.property_logit_gts[i])
            self.property_logits[i, :le] = self.property_logit_gts[i]


    def updata_txt(self, unfilter_a, unfilter_a_attr):
        for obj_ind in unfilter_a:
            if obj_ind in [self.db.class_to_ind[n] for n in self.obj_attr_pari.keys()]:
                txt = self.db.classes[obj_ind][0]
                attr_gt = []
                for u in self.db.classes[obj_ind]:
                    if u not in self.obj_attr_pari.keys():
                        continue
                    for v in self.obj_attr_pari[u]:
                        attr_gt.extend([self.db.attribute_to_ind[m] for m in v]) 
                jj = set(unfilter_a_attr) & set(attr_gt)
                txt1 = txt                 
                if txt not in self.obj_attr_dict.keys():
                    self.obj_attr_dict[txt] = []                      
                    for jjj in jj:
                        self.obj_attr_dict[txt].append(self.db.attributes[jjj][0])
                        self.obj_attr_dict[txt] = list(set(self.obj_attr_dict[txt]))
                        txt1 = self.db.attributes[jjj][0] + " " + txt1
                    if txt1 not in self.txt:
                        self.txt.append(txt1)
                else:
                    for i, t in enumerate(self.txt):
                        if (t.split(" ")[-1]) == txt and (i > (self.cfg.test_turns - 1)) and (len(jj) > 0):
                            del self.txt[i]
                            break
                    for jjj in jj:
                        self.obj_attr_dict[txt].append(self.db.attributes[jjj][0])
                        self.obj_attr_dict[txt] = list(set(self.obj_attr_dict[txt]))
                    for kk in self.obj_attr_dict[txt]:
                        txt = kk + " " + txt
                    if txt not in self.txt:
                        self.txt.append(txt)

    def step(self, action1, action2, split):
        """
        compute next observation, reward and termination signal
        :param action:
        :param dis: distribution of actions, [1, n_categories]
        :return:
        """
        # update text
        unfilter_a, filter_a = filter_actions(action1, self.logit)
        unfilter_a_attr, filter_a_attr = filter_actions(action2, self.attr_logit)
        logging.info('False objects:')
        for a in filter_a:
            logging.info(self.db.classes[a][0])

        logging.info('False attributes:')
        for a in filter_a_attr:
            logging.info(self.db.attributes[a][0])

        self.updata_txt(unfilter_a, unfilter_a_attr)

        txt_object_index = list(set([self.db.class_to_ind[n] for m in self.txt for n in m.split(" ") if n in self.db.class_to_ind.keys()]))
        txt_attribute_index = list(set([self.db.attribute_to_ind[n] for m in self.txt for n in m.split(" ") if n in self.db.attribute_to_ind.keys()]))

        if len(txt_object_index) or len(txt_attribute_index):
            if len(txt_object_index) > 0  and len(txt_attribute_index) == 0:
                glove_vector = torch.mean(self.glove_word_vector[txt_object_index], dim=0, keepdim=True)
            elif len(txt_object_index) and len(txt_attribute_index):
                glove_vector = torch.mean(self.glove_word_vector[txt_object_index], dim=0, keepdim=True)
                attribute_glove_vector = torch.mean(self.glove_word_attribute_vector[txt_attribute_index], dim=0, keepdim=True)
                new_vector = torch.cat((glove_vector, attribute_glove_vector), dim=0)
                glove_vector = torch.mean(new_vector, dim=0, keepdim=True)
            elif len(txt_object_index) == 0 and len(txt_attribute_index) > 0:
                glove_vector = torch.mean(self.glove_word_attribute_vector[txt_attribute_index], dim=0, keepdim=True)
        else:
            glove_vector = torch.zeros(size=(1,300))

        word_inds, word_msks = self.tokenize()
        logging.info('all query:')
        for t in self.txt: 
            logging.info(t)

        # masm some images using threshold
        for a in filter_a:
            filter_img = np.where(self.vg_img_logits[:, a] > 0.9)
            self.mask[filter_img] *= 0.9


        # generate next obs
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)
        txt_feat = txt_feat.detach()

        if self.cfg.vg_img_feature is not None and self.cfg.vg_img_logits is not None:
            reward, inds, rank = self.retrieve_precomp(word_inds, word_msks)
        else:
            reward, inds, rank = self.retrieve(word_inds, word_msks)

        # get top10 image
        top_inds = inds[:10]
        img_inds = [self.db.scenedb[idx]['image_index'] for idx in top_inds]
        logging.info('Top image: ')
        for i in img_inds:
            logging.info(i)

        top_img_logits = self.vg_img_logits[inds[:100]]
        top_img_logits_dis = softmax(np.sum(top_img_logits, axis=0))

        logits_dis = top_img_logits_dis
        logits_dis = torch.tensor(logits_dis, dtype=torch.float).unsqueeze(0)

        if self.cfg.use_glove:
            obs = torch.cat([torch.mean(txt_feat, dim=0, keepdim=True),
                         logits_dis.type_as(txt_feat),
                         glove_vector.type_as(txt_feat)], dim=-1)
        else:
            obs = torch.cat([torch.mean(txt_feat, dim=0, keepdim=True),
                         logits_dis.type_as(txt_feat)], dim=-1)
                        

        if rank[0] < self.cfg.ppo_stop_rank:
            return obs, reward, True, inds, self.logit_return, txt_object_index, txt_attribute_index
        else:
            return obs, reward, False, inds, self.logit_return, txt_object_index, txt_attribute_index


    def gen_obj_logit(self):
        txt = self.txt
        word_inds = []
        for tx in txt:
            tokens = [w for w in word_tokenize(tx)]
            word_inds.extend([self.db.class_to_ind[w]  for w in tokens if w in self.db.class_to_ind.keys()])
        word_inds = list(set(word_inds))
        logit1 = np.sum(self.object_adj[word_inds], axis=0)
        logit2 = np.sum(self.obj_attr_stat[word_inds], axis=0)
        if len(word_inds) == 0:
            logit1 = logit1 / (np.sum(logit1) + 1e6)
            logit2 = logit2 / (np.sum(logit2) + 1e6)
        else:
            logit1 = logit1 / np.sum(logit1)
            logit2 = logit2 / np.sum(logit2)
        
        return logit1, logit2

    def gen_obj_logit_pm(self):
        txt = self.txt
        word_inds = []
        for tx in txt:
            tokens = [w for w in word_tokenize(tx)]
            word_inds.extend([self.db.class_to_ind[w]  for w in tokens if w in self.db.class_to_ind.keys()])
        word_inds = list(set(word_inds))
        logit1 = np.sum(self.object_adj[word_inds], axis=0)
        if len(word_inds) == 0:
            logit1 = logit1 / (np.sum(logit1) + 1e6)
        else:
            logit1 = logit1 / np.sum(logit1)
        
        return logit1

    def reset(self, txt, index):
        """
        reset environment
        :param txt: a str
        :return
        """
        self.txt = txt
        self.index = index
        word_inds, word_msks = self.tokenize()
        logging.info('Index: {}'.format(index))

        scene = self.db.scenedb[index]
        obj_to_ind = self.db.class_to_ind
        attr_to_ind = self.db.attribute_to_ind
        objs = [scene['objects'][obj]['name'] for obj in scene['objects'] if
                scene['objects'][obj]['name'] in obj_to_ind.keys()]
        obj_inds = list(set([obj_to_ind[obj] for obj in objs]))
        self.logit = obj_inds

        self.logit_return = np.zeros(self.cfg.n_categories)
        self.logit_return[self.logit] = 1.0


        attrs = []
        self.obj_attr_pari = {}
        self.scene_attr_object_pari = []
        for obj in scene['objects']:
            if scene['objects'][obj]['name'] in obj_to_ind.keys():
                filter_attr = [n for n in scene['objects'][obj]['atts'] if n in attr_to_ind.keys()]
                if scene['objects'][obj]['name'] in self.obj_attr_pari.keys():
                    self.obj_attr_pari[scene['objects'][obj]['name']].append(filter_attr)
                else:
                    self.obj_attr_pari[scene['objects'][obj]['name']] = [filter_attr]
        for k, v in self.obj_attr_pari.items():
            txt = k
            temp_attr = []
            for v_ in v:
                if v_ != []:             
                    for v__ in v_:
                        attrs.append(v__)    
                        if v__ not in temp_attr:
                            temp_attr.append(v__)
                            txt = v__ + " " + txt
            self.scene_attr_object_pari.append(txt) 
        attrs_inds = list(set(attr_to_ind[attr] for attr in attrs))
        attrs = list(set(attrs))
        self.attr_logit = attrs_inds
        self.attr_logit_return = np.zeros(self.cfg.n_property)
        self.attr_logit_return[self.attr_logit] = 1.0


        # compute text feature
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)
        self.txt_feat = txt_feat.detach()

        if self.vg_img_logits is not None:
            self.mask = np.ones(len(self.vg_img_logits))

        self.obj_attr_dict = {}


    def tokenize(self):

        word_inds = []
        lengths = []
        for t in self.txt:
            tokens = [w for w in word_tokenize(t)]
            word_ind = [self.db.lang_vocab(w) for w in tokens]
            word_ind.append(self.cfg.EOS_idx)
            lengths.append(len(word_ind))
            word_inds.append(torch.Tensor(word_ind))
        max_length = max(lengths)
        new_word_inds = torch.zeros(len(word_inds), max_length).long()
        new_word_msks = torch.zeros(len(word_inds), max_length).long()
        for i in range(len(lengths)):
            new_word_inds[i, :lengths[i]] = word_inds[i]
            new_word_msks[i, :lengths[i]] = 1.0
        return new_word_inds.unsqueeze(0), new_word_msks.unsqueeze(0)

    def compute_similarity(self, txt_feat, all_img_feats):
        # compute similarity
        torch.cuda.empty_cache()
        raw_sim = torch.matmul(txt_feat, all_img_feats.transpose(-1, -2))
        txt_feat_norm = torch.norm(txt_feat, 2, dim=-1, keepdim=True)
        all_img_feats_norm = torch.norm(all_img_feats, 2, dim=-1, keepdim=True)
        raw_sim_norm = txt_feat_norm.unsqueeze(0) * all_img_feats_norm.transpose(-1, -2)
        sim = raw_sim / raw_sim_norm

        attn = torch.softmax(sim * self.cfg.lse_lambda, dim=-1)
        sim = torch.mean(torch.mean(attn * sim, dim=-1, keepdim=True), dim=-2, keepdim=True).squeeze()

        # rank
        sim = sim.detach().cpu().numpy()
        sim = sim * self.mask
        return sim

    def compute_attr_similarity(self, txt_feat, all_attr_feats):
        # compute similarity
        raw_sim = torch.matmul(txt_feat, all_attr_feats.transpose(0, 1))
        txt_feat_norm = torch.norm(txt_feat, 2, dim=-1, keepdim=True)
        all_attr_feats_norm = torch.norm(all_attr_feats, 2, dim=-1, keepdim=True)
        raw_sim_norm = txt_feat_norm * all_attr_feats_norm.transpose(0, 1)
        sim = raw_sim / raw_sim_norm

        attn = torch.softmax(sim * self.cfg.lse_lambda, dim=-1)
        sim = torch.mean(attn * sim, dim=-2, keepdim=True).squeeze()
        return sim

    def retrieve_precomp(self, word_inds, word_msks):
        """
        rank image accroding to a caption
        :param word_inds: [1, nturns, word num], LongTensor
        :return:
        """
        # compute text feature
        if self.cfg.cuda:
            word_inds = word_inds.cuda(non_blocking=True)
            word_msks = word_msks.cuda(non_blocking=True)
        bsize, nturns, nwords = word_inds.size()
        _, txt_feat, _ = self.txt_img_matching_trainer.net.txt_enc(word_inds.view(-1, nwords),
                                                                   word_msks.view(-1, nwords))
        if bsize == 1:
            txt_feat = txt_feat.view(nturns, self.cfg.n_feature_dim)
        else:
            raise Exception('Batchsize should be 1.')
        if self.cfg.l2_norm:
            txt_feat = l2norm(txt_feat)

        sim = self.compute_similarity(txt_feat, self.all_img_feats)
        sim_mean = np.mean(sim)

        reward = (sim[self.index] - sim_mean) * 100.0
        inds = np.argsort(sim)[::-1]
        rank = np.where(inds == self.index)
        logging.info('Now rank {}'.format(rank))

        return reward, inds, rank
#    