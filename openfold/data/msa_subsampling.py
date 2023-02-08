import torch
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform

# adapted from: https://github.com/sokrypton/GREMLIN_CPP

def get_eff(msa, eff_cutoff=0.8): # eff_cutoff=0.62 for metapsicov
    if msa.ndim == 3: msa = msa.argmax(-1)    
    # pairwise identity  
    msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
    # weight for each sequence
    msa_w = (msa_sm >= eff_cutoff).astype(float)
    msa_w = 1/np.sum(msa_w,-1)

    return msa_w

def subsample_msa(msa, neff=10, eff_cutoff=0.8, cap_msa=True):
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
        if cap_msa:
            if w + current_neff > neff or i > 126:
                break
        else:
            if w + current_neff > neff:
                break           
        pick.append(msa[i])
        current_neff += w
        
    return np.array(pick)

# if cap_msa is enabled, we bypass the ExtraMSAStack, helps with determinism for |MSA| < 128
def subsample_msa_sequentially(msa, neff=10, eff_cutoff=0.8, cap_msa=True):
    if msa.shape[0] == 1:
        return np.array([0])

    indices = [0]

    idx = np.arange(msa.shape[0] - 1) + 1
    np.random.shuffle(idx)

    new = [msa[0]]

    for i in idx:
        new.append(msa[i])
        indices.append(i)
        neff_ = get_eff(np.array(new), eff_cutoff=eff_cutoff).sum()

        if cap_msa:
            if neff_ > neff or len(new) > 126:
                new.pop()
                indices.pop()
                break
        else:
            if neff_ > neff:
                new.pop()
                indices.pop()
                break

    return np.array(indices)

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
