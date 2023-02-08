import torch
import numpy as np
import math

def sample_ambiguous_links(crosslinker_position,partner_position, L):
    # we assume for now there can be at most 6 ambiguous links
    p = [0.5,0.1,0.1,0.1,0.1,0.1]
    e = np.arange(6)
    no = np.random.choice(e,p=p)
    if no == 0:
        return [(crosslinker_position,partner_position)]
    
    start = np.random.choice(np.arange(max(0,partner_position-no+1),partner_position+1))
    partner = list(range(start,min(start+no,L)))
    return zip([crosslinker_position] * len(partner), partner)

def round_down(x, a=0.05):
    return math.floor(x / a) * a

def sample_xl(tp,fp,seq,fdr=0.05,cov=0.1,ambiguous=False, stochastic=False):
    
    photoAA = ['L', 'K']

    pick_tp_ = int(len(tp) * cov)
    pick_tp = len(tp) if pick_tp_ == 0 else pick_tp_

    if stochastic:
        pick_fp = torch.sum(torch.bernoulli(torch.full((pick_tp,),fdr))).long()
    else:
        pick_fp = max(1, math.ceil(fdr / (1 - fdr) * pick_tp))


    tp = tp[torch.randperm(len(tp))][:pick_tp]
    fp = fp[torch.randperm(len(fp))][:pick_fp]    

    tp = torch.cat([tp,torch.ones(len(tp))[:,None]],dim=1)
    fp = torch.cat([fp,torch.zeros(len(fp))[:,None]],dim=1)

    xl = torch.cat([tp,fp],dim=0)

    xl = xl[torch.randperm(len(xl))]
    
    # add fdr
    count = 0
    n = len(xl)
    for i, row in enumerate(xl):
        count += 1 - row[2]
        fdr_ = min(1-fdr, round_down((n - count) / n))
        xl[i,2] = fdr_

    groups = np.arange(n) + 1 # 0th group for non-xl
    np.random.shuffle(groups)

    n = len(seq)


    xl_ = np.zeros((n,n,1))
    grouping = np.zeros((n,n,1))
    real = np.zeros((n,n))
    
    for i, (r1,r2,fdr) in enumerate(xl):
        r1 = int(r1.item())
        r2 = int(r2.item())
        first, second = (r1,r2) if seq[r1] in photoAA else (r2,r1)
        
        if ambiguous:
            for aa1, aa2 in sample_ambiguous_links(first,second,n):
                xl_[aa1,aa2,0] = xl_[aa2,aa1,0] = fdr
                grouping[aa1,aa2,0] = grouping[aa2,aa1,0] = groups[i]
        else:
            xl_[first,second,0] = xl_[second,first,0] = fdr
            grouping[first,second,0] = grouping[second,first,0] = groups[i]
                    

    return xl_, grouping, real
