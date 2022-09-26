import numpy as np
import numpy.ma as ma
from scipy.sparse import csc_matrix as sparse_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy.sparse import diags,identity,coo_matrix,csr_matrix
import msmtools.estimation as msm_estimation
import msmtools.analysis as msm_analysis
import stats
from scipy.signal import find_peaks
from scipy.signal import find_peaks


import matplotlib.pyplot as plt


def segment_maskedArray(tseries,min_size=50):
    '''
    Segments  time series in case it has missing data
    '''
    if len(tseries.shape)>1:
        mask = ~np.any(tseries.mask,axis=1)
    else:
        mask = ~tseries.mask
    segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
    return segments


def get_count_matrix(labels,lag,nstates):
    observable_seqs = ma.compress_rows(ma.vstack([labels[:-lag],labels[lag:]]).T)

    row = observable_seqs[:,0]
    col = observable_seqs[:,1]

    data = np.ones(row.size)
    C = coo_matrix((data, (row, col)), shape=(nstates, nstates))
    # export to output format
    count_matrix = C.tocsr()

    return count_matrix


def get_count_ms(dtrajs,delay,nstates):
    if len(dtrajs.shape)>1:
        count_ms = coo_matrix((nstates,nstates))
        for dtraj in dtrajs:
            try:
                count_ms+=get_count_matrix(dtraj,delay,nstates)
            except:
                print('Warning! No samples.')
                continue
    else:
        try:
            count_ms=get_count_matrix(dtrajs,delay,nstates)
        except:
            print('Warning! No samples.')
    return count_ms

def tscales_samples(labels,delay,dt,size,n_modes=5,reversible=True):
    dtrajs = get_split_trajs(labels,size)
    nstates = np.max(labels)+1
    P_traj=[]
    ts_traj = []
    for sample_traj in dtrajs:
        count_ms = get_count_ms(sample_traj,delay,nstates)
        connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
        P = msm_estimation.tmatrix(connected_count_matrix)
        if reversible:
            R = get_reversible_transition_matrix(P)
            tscale = compute_tscales(R,delay,dt,k=n_modes+1)
        else:
            tscale = compute_tscales(P,delay,dt,k=n_modes+1)
        ts_traj.append(tscale)
        P_traj.append(P)
    return ts_traj,P_traj

def transition_matrix(labels,lag,return_connected=False):
    nstates = np.max(labels)+1
    count_matrix = get_count_matrix(labels,lag,nstates)
    connected_count_matrix = msm_estimation.connected_cmatrix(count_matrix)
    P = msm_estimation.tmatrix(connected_count_matrix)
    if return_connected:
        lcs = msm_estimation.largest_connected_set(count_matrix)
        return lcs,P
    else:
        return P

def get_connected_labels(labels,lcs):
    final_labels = ma.zeros(labels.shape,dtype=int)
    for key in np.argsort(lcs):
        final_labels[labels==lcs[key]]=key+1
    final_labels[final_labels==0] = ma.masked
    final_labels-=1
    return final_labels

def sorted_spectrum(R,k=5,which='LR'):
    eigvals,eigvecs = eigs(R,k=k,which=which)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    return eigvals[sorted_indices],eigvecs[:,sorted_indices]

def compute_tscales(P,delay,dt=1,k=2):
    try:
        if P.shape[1]<=10:
            eigvals = np.sort(eig(P.toarray())[0])[::-1][:k]
        else:
            eigvals = eigs(P,k=k,which='LR',return_eigenvectors=False)
        sorted_indices = np.argsort(eigvals.real)[::-1]
        eigvals = eigvals[sorted_indices][1:].real
        eigvals[np.abs(eigvals-1)<1e-12] = np.nan
        eigvals[eigvals<1e-12] = np.nan
        return -(delay*dt)/np.log(np.abs(eigvals))
    except:
        return np.array([np.nan]*(k-1))


def get_reversible_transition_matrix(P):
    probs = stationary_distribution(P)
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def get_split_trajs(labels,size = 0):
    if size == 0:
        size = len(labels)/20
    return ma.array([labels[kt:kt+size] for kt in range(0,len(labels)-size,size)])


def implied_tscale(labels,size,delay,dt,n_modes,reversible=True):
    dtrajs = get_split_trajs(labels,size)
    nstates = np.max(labels)+1
    count_ms = get_count_ms(dtrajs,delay,nstates)
    connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
    P = msm_estimation.tmatrix(connected_count_matrix)
    if reversible:
        R = get_reversible_transition_matrix(P)
        tscale = compute_tscales(R,delay,dt,k=n_modes+1)
    else:
        tscale = compute_tscales(P,delay,dt,k=n_modes+1)
    return tscale


def get_bootstrapped_Ps(labels,delay,n_samples,size = 0):
    #get dtrajs to deal with possible nans
    dtrajs = get_split_trajs(labels,size)
    nstates = np.unique(labels.compressed()).shape[0]

    sample_Ps=[]
    for k in range(n_samples):
        sample_trajs = [dtrajs[k] for k in np.random.randint(0,len(dtrajs),len(dtrajs))]
        count_ms = get_count_ms(sample_trajs,delay,nstates)
        connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
        P = msm_estimation.tmatrix(connected_count_matrix)
        sample_Ps.append(P)
    return sample_Ps


def boostrap_tscales(labels,delay,dt,n_modes,n_samples = 1000,size=0,reversible=True):
    Ps = get_bootstrapped_Ps(labels,delay,n_samples,size)
    tscales=np.zeros((n_samples,n_modes))
    for k,P in enumerate(Ps):
        if reversible:
            R = get_reversible_transition_matrix(P)
            tscale = compute_tscales(R,delay,dt,k=n_modes+1)
        else:
            tscale = compute_tscales(P,delay,dt,k=n_modes+1)
        tscales[k,:]=tscale
    return tscales


def bootstrap_tscale_sample(labels,delay,dt,n_modes,size=0,reversible=True):
    dtrajs = get_split_trajs(labels,size)
    nstates = np.unique(labels.compressed()).shape[0]

    sample_trajs = [dtrajs[k] for k in np.random.randint(0,len(dtrajs),len(dtrajs))]
    count_ms = get_count_ms(sample_trajs,delay,nstates)
    connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
    P = msm_estimation.tmatrix(connected_count_matrix)
    if reversible:
        R = get_reversible_transition_matrix(P)
        tscale = compute_tscales(R,delay,dt,k=n_modes+1)
    else:
        tscale = compute_tscales(P,delay,dt,k=n_modes+1)
    return tscale


def bootstrap_tscales_delays(range_delays,labels,n_modes,dt,n_samples=1000,size=0,reversible=True):
    dtrajs = get_split_trajs(labels,size)
    nstates = np.unique(labels.compressed()).shape[0]
    sample_trajs = [dtrajs[k] for k in np.random.randint(0,len(dtrajs),len(dtrajs))]
    tscales=np.zeros((len(range_delays),n_modes))
    for kd,delay in enumerate(range_delays):
        try:
            count_ms = get_count_ms(sample_trajs,delay,nstates)
            connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
            P = msm_estimation.tmatrix(connected_count_matrix)
            if reversible:
                R = get_reversible_transition_matrix(P)
                tscale = compute_tscales(R,delay,dt,k=n_modes+1)
            else:
                tscale = compute_tscales(P,delay,dt,k=n_modes+1)
            tscales[kd,:] = tscale
        except:
            continue
    return tscales


def compute_implied_tscales(labels,range_delays,dt=1,n_modes=5,n_samples=1000,size=0,reversible=False,confidence = 95):
    if length==0:
        length = np.max(range_delays)*2
    cil = (100-confidence)/2
    ciu = 100-cil
    cil_delay=np.zeros((len(range_delays),n_modes))
    ciu_delay=np.zeros((len(range_delays),n_modes))
    mean_delay=np.zeros((len(range_delays),n_modes))
    bootstrapped_tscales  = []
    for kd,delay in enumerate(range_delays):
        mean_tscale = implied_tscale(labels,size,delay,dt,n_modes,reversible)
        tscales_samples = boostrap_tscales(labels,delay,dt,n_modes,n_samples,size,reversible)
        mean_delay[kd,:] = mean_tscale
        cil_delay[kd,:] =  np.nanpercentile(tscales_samples,cil,axis=0)
        ciu_delay[kd,:] = np.nanpercentile(tscales_samples,ciu,axis=0)
    return cil_delay,ciu_delay,mean_delay



def stationary_distribution(P):
    probs = msm_analysis.statdist(P)
    return probs


def get_entropy(labels):
    #get dtrajs to deal with possible nans
    P = transition_matrix(labels,1)
    probs = stationary_distribution(P)
    logP = P.copy()
    logP.data = np.log(logP.data)
    return (-diags(probs).dot(P.multiply(logP))).sum()



def simulate(P,state0,iters):
    '''
    Monte Carlo simulation of the markov chain characterized by the matrix P
    state0: initial system
    iters: number of iterations of the simulation
    '''
    states = np.zeros(iters,dtype=int)
    states[0]=state0
    state=state0
    for k in range(1,iters):
        new_state = np.random.choice(np.arange(P.shape[1]),p=list(np.hstack(P[state,:].toarray())))
        state=new_state
        states[k]=state
    return states



def optimal_partition(phi2,inv_measure,P,return_rho = True):

    X = phi2
    c_range = np.sort(phi2)[1:-1]
    rho_c = np.zeros(len(c_range))
    rho_sets = np.zeros((len(c_range),2))
    for kc,c in enumerate(c_range):
        labels = np.zeros(len(X),dtype=int)
        labels[X<=c] = 1
        rho_sets[kc] = [(inv_measure[labels==idx]*(P[labels==idx,:][:,labels==idx])).sum()/inv_measure[labels==idx].sum()
                      for idx in range(2)]
    rho_c = np.min(rho_sets,axis=1)
    peaks, heights = find_peaks(rho_c, height=0.5)
    if len(peaks)==0:
        print('No prominent coherent set')
        return None
    else:
        idx = peaks[np.argmax(heights['peak_heights'])]

        c_opt = c_range[idx]
        kmeans_labels = np.zeros(len(X),dtype=int)
        kmeans_labels[X<c_opt] = 1

        if return_rho:
            return c_range,rho_sets,idx,kmeans_labels
        else:
            return kmeans_labels
