#!/usr/bin/env python

import os, sys, cv2, json
import random, pickle, math
import numpy as np
import os.path as osp
from PIL import Image
from time import time
from copy import deepcopy
from glob import glob
from nltk.tokenize import word_tokenize
from utils.config import get_train_config, get_test_config
from utils.utils import *
from datasets.vg import vg
from utils.vocab import Vocabulary
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class region_loader(Dataset):
    def __init__(self, imdb):
        self.cfg = imdb.cfg
        self.db = imdb
        self.obj_to_ind = imdb.class_to_ind
        self.attr_to_ind = imdb.attribute_to_ind
        self.attr = imdb.attributes
        # print(self.obj_to_ind)

    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, scene_index):

        scene = self.db.scenedb[scene_index]

        image_index = scene['image_index']

        # objects and attributes
        objs = [scene['objects'][obj]['name'] for obj in scene['objects'] if scene['objects'][obj]['name'] in self.obj_to_ind.keys()]
        objs = list(set(objs))
        objs_inds = list(set([self.obj_to_ind[obj] for obj in objs]))
        attrs_inds = []
        attrs = []
        atts = {}
        for obj in scene['objects']:
            if scene['objects'][obj]['name'] in self.obj_to_ind.keys():
                filter_attr = [n for n in scene['objects'][obj]['atts'] if n in self.attr_to_ind.keys()]
                for m in scene['objects'][obj]['atts']:
                    if m in self.attr_to_ind.keys():
                        attrs_inds.append(self.attr_to_ind[m])
                        attrs.append(m)
                if scene['objects'][obj]['name'] in atts.keys():
                    atts[scene['objects'][obj]['name']].append(filter_attr)
                else:
                    atts[scene['objects'][obj]['name']] = [filter_attr]
        attrs_inds = list(set(attrs_inds))
        attrs = list(set(attrs))
        # region features
        region_path = self.db.region_path_from_index(image_index)
        with open(region_path, 'rb') as fid:
            regions = pickle.load(fid, encoding='latin1')
        region_boxes = torch.from_numpy(regions['region_boxes']).float()
        region_feats = torch.from_numpy(regions['region_feats']).float()
        region_clses = torch.from_numpy(regions['region_clses']).long()
        # captions
        if self.db.name == 'coco':
            all_captions = scene['captions']
        else:
            all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]
            all_captions = [x['caption'] for x in all_meta_regions]

        width = scene['width']
        height = scene['height']
        if self.db.split in ['val', 'test']:
            captions = all_captions[:self.cfg.max_turns]
        else:
            num_captions = len(all_captions)
            caption_inds = np.random.permutation(range(num_captions))
            captions = [all_captions[x] for x in caption_inds[:self.cfg.max_turns]]

        if self.cfg.paragraph_model:
            for i in range(1, len(captions)):
                captions[i] = captions[i - 1] + ';' + captions[i]

        if self.cfg.use_attr:
            captions = aug_txt_with_attr(captions, objs)

        captions = tuple(captions)
        sent_inds = []
        for i in range(self.cfg.max_turns):
            tokens = [w for w in word_tokenize(captions[i])]
            # tokens = further_token_process(tokens)
            word_inds = [self.db.lang_vocab(w) for w in tokens]
            word_inds.append(self.cfg.EOS_idx)
            sent_inds.append(torch.Tensor(word_inds))
        sent_inds = tuple(sent_inds)

        obj2word_inds = []
        for obj in objs:
            token = [w for w in word_tokenize(obj)]
            word_ind = [self.db.lang_vocab(w) for w in token]
            word_ind.append(self.cfg.EOS_idx)
            obj2word_inds.append(torch.Tensor(word_ind))
        obj2word_inds = tuple(obj2word_inds)

        attr2word_inds = []
        for attr in attrs:
            token = [w for w in word_tokenize(attr)]
            word_ind = [self.db.lang_vocab(w) for w in token]
            word_ind.append(self.cfg.EOS_idx)
            attr2word_inds.append(torch.Tensor(word_ind))
        attr2word_inds = tuple(attr2word_inds)
        return sent_inds, captions, region_boxes, region_feats, region_clses, objs, objs_inds, obj2word_inds, attrs, width, height, image_index, scene_index, attrs_inds, attr2word_inds, atts


def region_collate_fn(data):
    sent_inds, captions, region_boxes, region_feats, region_clses, objs, objs_inds, obj2word_inds, attrs, width, height, image_indices, scene_indices, attrs_inds, attr2word_inds, attrs = zip(
        *data)

    # regions
    lengths = [region_boxes[i].size(0) for i in range(len(region_boxes))]
    max_length = max(lengths)

    new_region_boxes = torch.zeros(len(region_boxes), max_length, region_boxes[0].size(-1)).float()
    new_region_feats = torch.zeros(len(region_feats), max_length, region_feats[0].size(-1)).float()
    new_region_clses = torch.zeros(len(region_clses), max_length).long()
    new_region_masks = torch.zeros(len(region_clses), max_length).long()

    for i in range(len(region_boxes)):
        end = region_boxes[i].size(0)
        new_region_boxes[i, :end] = region_boxes[i]
        new_region_feats[i, :end] = region_feats[i]
        new_region_clses[i, :end] = region_clses[i]
        new_region_masks[i, :end] = 1.0

    # captions
    lengths = [len(sent_inds[i][j]) for i in range(len(sent_inds)) for j in range(len(sent_inds[0]))]
    max_length = max(lengths)
    new_sent_inds = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()
    new_sent_msks = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()
    for i in range(len(sent_inds)):
        for j in range(len(sent_inds[0])):
            end = len(sent_inds[i][j])
            new_sent_inds[i, j, :end] = sent_inds[i][j]
            new_sent_msks[i, j, :end] = 1

    # objects
    lengths = [len(objs_inds[i]) for i in range(len(objs_inds)) ]
    max_length = max(lengths)
    new_objs_inds = torch.zeros(len(objs_inds), max_length).long()
    new_objs_msks = torch.zeros(len(objs_inds), max_length).long()
    for i in range(len(objs_inds)):
        for j in range(len(objs_inds[i])):
            new_objs_inds[i, j] = objs_inds[i][j]
            new_objs_msks[i, j] = 1.0

    # attributes
    lengths = [len(attrs_inds[i]) for i in range(len(attrs_inds)) ]
    max_length = max(lengths)
    new_attrs_inds = torch.zeros(len(attrs_inds), max_length).long()
    new_attrs_msks = torch.zeros(len(attrs_inds), max_length).long()
    for i in range(len(attrs_inds)):
        for j in range(len(attrs_inds[i])):
            new_attrs_inds[i, j] = attrs_inds[i][j]
            new_attrs_msks[i, j] = 1.0

    # objects to words
    new_obj2word_inds = torch.zeros(len(obj2word_inds), 50, 5).long()
    new_obj2word_msks = torch.zeros(len(obj2word_inds), 50, 5).long()
    for i in range(len(obj2word_inds)):
        for j in range(len(obj2word_inds[i])):
            end = len(obj2word_inds[i][j])
            new_obj2word_inds[i,j,:end] = obj2word_inds[i][j]
            new_obj2word_msks[i,j,:end] = 1.0

    new_attr2word_inds = torch.zeros(len(attr2word_inds), 50, 5).long()
    new_attr2word_msks = torch.zeros(len(attr2word_inds), 50, 5).long()
    for i in range(len(attr2word_inds)):
        for j in range(len(attr2word_inds[i])):
            end = len(attr2word_inds[i][j])
            new_attr2word_inds[i,j,:end] = attr2word_inds[i][j]
            new_attr2word_msks[i,j,:end] = 1.0



    entry = {
        'region_boxes': new_region_boxes,
        'region_feats': new_region_feats,
        'region_masks': new_region_masks,
        'region_clses': new_region_clses,
        'sent_inds': new_sent_inds,
        'sent_msks': new_sent_msks,
        'objs_inds': new_objs_inds,
        'objs_msks': new_objs_msks,
        'obj2word_inds': new_obj2word_inds,
        'obj2word_msks': new_obj2word_msks,
        'captions': captions,
        'objects': objs,
        'attributes': attrs,
        'widths': width,
        'heights': height,
        'image_inds': image_indices,
        'scene_inds': torch.Tensor(scene_indices),
        'attrs_inds': new_attrs_inds,
        'attrs_msks': new_attrs_msks,
        'attr2word_inds': new_attr2word_inds,
        'attr2word_msks': new_attr2word_msks,
        'atts':attrs
    }

    return entry


def test():
    config, unparsed = get_train_config()

    traindb = vg(config, 'train')
    train_loaddb = region_loader(traindb)
    train_loader = torch.utils.data.DataLoader(train_loaddb, batch_size=10, shuffle=False, collate_fn=region_collate_fn)
    print(len(train_loaddb))
    for data in train_loader:
        for key in data:
            print(key, data[key].shape if isinstance(data[key], torch.Tensor) else data[key])


if __name__ == '__main__':
    test()
