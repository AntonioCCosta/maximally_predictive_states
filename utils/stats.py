import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy import ndimage as ndi


def bootstrap(l,n_times,confidence_interval=95,median=False,maximum=False,log=False):
    if median:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),len(l))
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.median(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.median(l,axis=0),cil,ciu
    elif maximum:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),len(l))
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.max(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.max(l,axis=0),cil,ciu
    elif log:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),len(l))
            new_list=[np.log10(l[idx]) for idx in indices]
            new_means.append((ma.mean(new_list,axis=0)))
        new_means=10**(ma.vstack(new_means))
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return 10**(ma.mean(np.log10(np.array(l)),axis=0)),cil,ciu
    else:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,len(l)),len(l))
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.mean(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.mean(l,axis=0),cil,ciu


def state_lifetime(states,tau):
    durations=[]
    for state in np.sort(np.unique(states.compressed())):
        gaps = states==state
        gaps_boundaries = np.where(np.abs(np.diff(np.concatenate([[False], gaps, [False]]))))[0].reshape(-1, 2)
        durations.append(np.hstack(np.diff(gaps_boundaries))*tau)
    return durations

def cumulative_dist(data,lims,label='label'):
    lim0,lim1=lims
    data = np.array(data)
    data = data[data>=lim0]
    data = data[data<=lim1]
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted,p

def complementary_cumulative_dist(data,lims,label='label'):
    lim0,lim1=lims
    data = np.array(data)
    data = data[data>=lim0]
    data = data[data<=lim1]
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted,1-p
