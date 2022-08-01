import torch
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform

def get_eff(msa, eff_cutoff=0.8): # eff_cutoff=0.62 for metapsicov
    if msa.ndim == 3: msa = msa.argmax(-1)    
    # pairwise identity  
    msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
    # weight for each sequence
    msa_w = (msa_sm >= eff_cutoff).astype(float)
    msa_w = 1/np.sum(msa_w,-1)

    return msa_w

def subsample_msa(msa, neff=10, eff_cutoff=0.8):
    if msa.shape[0] == 1:
        return msa

    weights = get_eff(msa, eff_cutoff=eff_cutoff)
    
    current_neff = weights[0]
    
    pick = [msa[0]]
    
    idx = np.arange(msa.shape[0]-1) + 1
    np.random.shuffle(idx)
    weights = weights[idx]
    msa = msa[idx]
    
    idx = np.argsort(weights)[::-1]
    
    weights = weights[idx]
    msa = msa[idx]
    
    for i, w in enumerate(weights):
        # print(w+current_neff, neff, w)
        if w + current_neff > neff or i > 126:
            break
        pick.append(msa[i])
        current_neff += w
        
    return np.array(pick)

def subsample_msa_sequentially(msa, neff=10, eff_cutoff=0.8):
    if msa.shape[0] == 1:
        return msa
    
    new = [msa[0]]
    
    msa = msa[1:]
    
    idx = np.arange(msa.shape[0])
    np.random.shuffle(idx)
    msa = msa[idx]
    
    for m in msa:
        new.append(m)
        neff_ = get_eff(np.array(new), eff_cutoff=eff_cutoff).sum()
        # print(neff_, len(new))
        if neff_ > neff or len(new) > 126:
            new.pop()
            break
            
    return np.array(new)


def subsample_msa_random(msa, neff=10, eff_cutoff=0.8):
    if msa.shape[0] == 1:
        return msa

    weights = get_eff(msa, eff_cutoff=eff_cutoff)
    
    current_neff = weights[0]
    
    pick = [msa[0]]
    
    msa = msa[1:]
    weights = weights[1:]
    
    idx = np.arange(msa.shape[0])
    np.random.shuffle(idx)
    weights = weights[idx]
    msa = msa[idx]
    
    for i, w in enumerate(weights):
        if w + current_neff > neff:
            break
        pick.append(msa[i])
        current_neff += w
        
    return np.array(pick)
