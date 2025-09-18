'''
ARI: Adjusted Rand Index
NMI: Normalized Mutual Informtion
ACC: Accuracy
'''

from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as NMI
from sklearn.metrics import accuracy_score as ACC
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import torch
import torch.nn.functional as F

def best_mapping(labels_true, labels_pred):
    '''
    :param labels_true: list or np.array
    :param labels_pred: list or np.array
    find the best mapping between labels_true and labels_pred
    '''
    D = max(max(labels_true), max(labels_pred)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(labels_pred)):
        w[labels_pred[i], labels_true[i]] += 1
    mapping = linear_assignment(w.max() - w)
    old_pred, new_pred = mapping
    label_map = dict(zip(old_pred, new_pred))
    labels_pred = [label_map[x] for x in labels_pred]
    labels_pred = np.array(labels_pred)
    return labels_true, labels_pred


def evaluation_(labels_true, labels_pred):
    '''
    labels_true: list or np.array
    labels_pred: list or np.array
    '''
    ARI_score = ARI(labels_true, labels_pred)
    NMI_score = NMI(labels_true, labels_pred)
    ACC_score = ACC(labels_true, labels_pred)
    return NMI_score, ARI_score, ACC_score


def evaluation(labels_true, labels_pred):
    '''
    labels_true: list or np.array
    labels_pred: list or np.array
    '''
    labels_true, labels_pred = best_mapping(labels_true, labels_pred)
    ARI_score = ARI(labels_true, labels_pred)
    NMI_score = NMI(labels_true, labels_pred)
    ACC_score = ACC(labels_true, labels_pred)
    return NMI_score, ARI_score, ACC_score

def curvature(z_i, z_j,z_k,z_l):
    vec1=z_i-z_j
    vec2=z_i-z_k
    vec3=z_i-z_l
    cos_sim12 = F.cosine_similarity(vec1, vec2)
    cos_sim13 = F.cosine_similarity(vec1, vec3)
    cos_sim23 = F.cosine_similarity(vec2, vec3)
    curv= 2*np.pi-torch.acos(cos_sim12)-torch.acos(cos_sim13)-torch.acos(cos_sim23)
    return curv