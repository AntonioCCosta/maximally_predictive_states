import numpy as np
import numpy.ma as ma
import sys
sys.path.append('/home/a/antonio-costa/theory_manuscript/') #change path
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import h5py
import argparse
from scipy.spatial import distance as sdist
import time

def main(argv):
    start_t = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-idx','--Idx',help="idx",default=0,type=int)
    args=parser.parse_args()
    idx = args.Idx
    #txt file with indices and delays for parallelization in computing cluster
    K,idx = np.array(np.loadtxt('/home/a/antonio-costa/theory_manuscript/Foraging/embedding_analysis/iteration_indices_K_1_60.txt')[idx],dtype=int)
    print(K,idx)

    seed_range = np.array(10**np.array(np.arange(.5,3.8,.25),dtype=float),dtype=int)

    print('Loading data ...')
    #change path to load split eigenworm trajectories
    f = h5py.File('/flash/StephensU/antonio/Foraging/split_ts_trajs/split_ts_trajs.h5','r')
    tseries = ma.masked_invalid(np.array(f[str(idx)]['traj']))
    tseries[tseries==0]=ma.masked
    f.close()

    print(tseries.shape)
    print(tseries[:20])
    #tseries = tseries[:1000]
    #seed_range = seed_range[:2]
        
    print('Compute entropies ...')
    H_K_s = np.zeros(len(seed_range))
    prob_K_s = np.zeros(len(seed_range))
    h_K_s = np.zeros(len(seed_range))
    Ipred_K_s = np.zeros(len(seed_range))
    eps_K_s = np.zeros(len(seed_range))
    
    traj_matrix = embed.trajectory_matrix(tseries,K=K-1)
    print(traj_matrix.shape)
    for ks,n_seeds in enumerate(seed_range):
        labels,centers = cl.kmeans_knn_partition(traj_matrix,n_seeds,return_centers=True)
        labels = ma.array(labels,dtype=int)
        max_epsilon = np.mean([np.max(np.linalg.norm(traj_matrix[labels==kc]-centers[kc],axis=0)) for kc in np.sort(np.unique(labels.compressed()))])
        
        P = op_calc.transition_matrix(labels,1)
        prob = op_calc.stationary_distribution(P)
        H = -np.sum(prob*np.log(prob))
        h = op_calc.get_entropy(labels)
        prob_K_s[ks] = np.mean(prob)
        H_K_s[ks] = H
        h_K_s[ks] = h
        Ipred_K_s[ks] = H-h
        eps_K_s[ks] = max_epsilon
        print(n_seeds)
    
    print('Saving results ...')
    #change output path
    f = h5py.File('/flash/StephensU/antonio/Foraging/embedding_results/entropic_properties_K_{}_{}.h5'.format(K,idx),'w')
    probs_ = f.create_dataset('probs',prob_K_s.shape)
    probs_[...] = prob_K_s
    H_ = f.create_dataset('entropies',H_K_s.shape)
    H_[...] = H_K_s
    h_ = f.create_dataset('entropy_rates',h_K_s.shape)
    h_[...] = h_K_s
    eps_ = f.create_dataset('eps_scale',eps_K_s.shape)
    eps_[...] = eps_K_s
    seeds_ = f.create_dataset('seed_range',seed_range.shape)
    seeds_[...] = seed_range
    Ipreds_ = f.create_dataset('Ipreds',Ipred_K_s.shape)
    Ipreds_[...] = Ipred_K_s
    f.close()
    end_t = time.time()
    print('It took {} s.'.format(end_t-start_t))

if __name__ == "__main__":
    main(sys.argv)
