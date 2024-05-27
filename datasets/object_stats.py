# from functools import cache
from operator import truediv
from optparse import Values
from re import L
import sys
from tkinter import W
import numpy as np
sys.path.append('./')
import tqdm
import threading
import os.path as osp
from datasets.vg import vg
from datasets.loader import region_loader
from utils.config import get_train_config
from utils.vocab import Vocabulary
import matplotlib.pylab as plt
import torch
import torchtext.vocab as vocab
same_obj = 0
idx = 0


def gen_object_adj(loaddbs):

    result = {}
    classes = {0: '__background__'} # each entry contains a list of names, except the '__background__' one
    class_to_ind = {}
    class_to_ind['__background__'] = 0

    with open(osp.join('./data/caches', 'vg_objects_vocab_1600.txt')) as f:
        count = 1
        for obj_alias in f.readlines():
            names = [x.lower().strip() for x in obj_alias.split(',')]
            classes[count] = names
            for x in names:
                class_to_ind[x] = count
            count += 1

    class2count = classes
    
    for k in class2count.keys():
        class2count[k]=0
    
    object_adj = np.zeros(shape=(1601, 1601))
    for loaddb in loaddbs:
        for scene in loaddb:
            obj, obj_ind = scene[5], scene[6]
            for i, index in enumerate(obj_ind):
                for j in range(i+1,len(obj_ind)):
                    object_adj[index][obj_ind[j]] += 1
                    object_adj[obj_ind[j]][index] += 1

            for o in obj:
                if o not in class_to_ind:
                    print(o)
                else:
                    class2count[class_to_ind[o]] += 1

    result['all_object_adj'] = object_adj
    all_object_num = np.array([class2count[index] for index in class2count])
    result['all_object_num'] = all_object_num

    import pickle
    file_path = './data/caches/vg_objects_num_adj_1601.pkl'
    with open(file_path,'wb') as f:
        pickle.dump(result, f)

def gen_glove_word2vec(loaddbs):
    glove = vocab.pretrained_aliases['glove.840B.300d'](cache = './')
    print("一共包含%d个词。" % len(glove.stoi))
    classes = loaddbs[0].db.classes
    print(classes.values())
    object_word2vec = []
    for words in classes.values():
        single_word = []
        vector = []
        if len(words) == 2:          
            for word in words:
                if " " in word:
                    word1, word2 = word.split(" ")
                    if "'s" in word1:
                        word1 = word1.split("'s")[0] 
                    if "'s" in word2:
                        word2 = word2.split("'s")[0]
                    single_word.append(word1)
                    single_word.append(word2)
                else:
                    if "'s" in word:
                        word = word.split("'s")[0]
                    single_word.append(word)
            vector = np.mean([glove.vectors[glove.stoi[sub]].numpy() for sub in single_word], axis=0)
        elif len(words) == 1:
            if " " in words[0]:
                word1, word2 = words[0].split(" ")
                if "'s" in word1:
                    word1 = word1.split("'s")[0] 
                if "'s" in word2:
                    word2 = word2.split("'s")[0]
                single_word.append(word1)
                single_word.append(word2)
                vector = np.mean([glove.vectors[glove.stoi[sub]].numpy() for sub in single_word], axis=0)
            else:
                if "'s" in words[0]:
                    words[0] = words[0].split("'s")[0]
                single_word.append(words[0])
                vector = glove.vectors[glove.stoi[words[0]]].numpy()
        else:
            vector = np.zeros(shape = (300,), dtype=np.float32)
        object_word2vec.append(vector)
    print(len(object_word2vec))
    import pickle
    file_path = './data/caches/vg_objects_glove_word2vector_300.pkl'
    with open(file_path,'wb') as f:
        pickle.dump(object_word2vec, f)


def main():
    config, unparsed = get_train_config()
    loaddbs = []
    for split in ['train']:
        db = vg(config, split)
        loaddb = region_loader(db)
        print(len(loaddb))
        loaddbs.append(loaddb)

    gen_glove_word2vec(loaddbs)
    gen_object_adj(loaddbs)


if __name__ == '__main__':


    main()



