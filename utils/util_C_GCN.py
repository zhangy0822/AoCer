import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F


'''gen_A: co-occur matrix generation'''
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))

    _adj = result['adj']    # (ndarray) (300, 300), count the co-accur numbers for each word in vocab
    _nums = result['nums']   # (ndarray) (300), count the total emerging numbers for each word in vocab

    # turn mat to binary according to threshold t (default t=0.4)
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1

    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)   # identity square matrix
    return _adj


def gen_glove_vector(adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    return result


''' define concept adj_matrix'''
def gen_A_concept(num_classes, t, adj_file, gen_mode='ML_GCN'):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))

    _nums = result['all_object_num']
    _nums = _nums[:, np.newaxis].astype(np.float64)
    _nums[0,0] = 1e-6
    _adj_all = result['all_object_adj'] / _nums

    _adj_all = rescale_adj_matrix(_adj_all)
    _adj_all[_adj_all < 0.1] = 0
    _adj_all[_adj_all >= 0.1] = 1
    _adj_all = _adj_all * 0.25 / (_adj_all.sum(0, keepdims=True) + 1e-6)
    _adj_all = _adj_all + np.identity(num_classes, np.int)


    return result['all_object_adj'], _adj_all


'''define the function to smooth the adj_matrix'''
def rescale_adj_matrix(adj_mat, t=5, p=0.02):
# def rescale_adj_matrix(adj_mat, t=5, p=0.02):
    """This function is to smooth the adj_matrix for dealing with the long-tail effect
    adj_mat: co-occurence adj matrix

    t: parameter_1, determine the amplify/shrink rate
    p: parameter_2, determine the borderline prob value of un-important concept to shrink
    context_word_length: we need to know the nums of context word,
                        because we need to suppress the role of context words for the whole representation
    """
    adj_mat_smooth = np.power(t, adj_mat - p) - np.power(t,  -p)
    # adj_mat_smooth = torch.pow(t, adj_mat - p) - torch.pow(t,  -p)
    return adj_mat_smooth


'''Laplacian Matrix transorm'''
def gen_adj(A):
    A = A.float()
    diag_1_mean = torch.diag(A.mean(1))
    A_1 = A + diag_1_mean
    D_1 = torch.pow(A_1.sum(1).float(), -0.5)
    D_1 = torch.diag(D_1)
    adj1 = torch.matmul(torch.matmul(A_1, D_1).t(), D_1)

    diag_2_mean = torch.diag(A.mean(0))
    A_2 = A + diag_2_mean
    D_2 = torch.pow(A_2.sum(0).float(), -0.5)
    D_2 = torch.diag(D_2)
    adj2 = torch.matmul(torch.matmul(A_2, D_2).t(), D_2)
 
    adj = (adj1 + adj2) / 2

    return adj


'''Laplacian Matrix transform for concept graph'''
def gen_adj_concept(A):

    adj = {}
    for key, value in A.items():
        if key == 'adj_O_P':
            D = torch.pow(A['adj_O_P'].sum(1).float(), -0.5)
            D = torch.diag(D)
            adj['adj_O_P'] = torch.matmul(torch.matmul(A['adj_O_P'], D).t(), D)
            adj['adj_O_P'].detach()

        if key == 'adj_O_M':
            D = torch.pow(A['adj_O_M'].sum(1).float(), -0.5)
            D = torch.diag(D)
            adj['adj_O_M'] = torch.matmul(torch.matmul(A['adj_O_M'], D).t(), D)
            adj['adj_O_M'].detach()

        elif key == 'adj_all':
            D = torch.pow(A['adj_all'].sum(1).float(), -0.5)
            D = torch.diag(D)
            adj['adj_all'] = torch.matmul(torch.matmul(A['adj_all'], D).t(), D)
            adj['adj_all'].detach()

    return adj
