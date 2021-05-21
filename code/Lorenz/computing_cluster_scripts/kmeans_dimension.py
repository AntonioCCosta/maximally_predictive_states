import numpy as np
import numpy.ma as ma
import sys
#change path to mother scripts
sys.path.append('/home/a/antonio-costa/theory_manuscript/')
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import h5py
import argparse
from scipy.spatial import distance as sdist
import time


def max_norm_dist(X,Y):
    return np.max(np.abs(X-Y),axis=1)

def main(argv):
    start_t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help="idx",default=0,type=int)
    args=parser.parse_args()
    n_seeds = np.array(np.logspace(2.5,5,51),dtype=int)[args.Idx]
    print(n_seeds)
    print('Loading data ...')
    
    #path to simulation data
    f = h5py.File('/bucket/StephensU/antonio/Lorenz/simulations/simulation.h5','r')
    sim = np.array(f['simulation'])
    dt = np.array(f['MetaData/integration_step'])[0]
    f.close()
    print(sim[:10])
    
    print('Compute dimensions ...')
    
    labels,centers = cl.kmeans_knn_partition(sim,n_seeds,return_centers=True)
    print(labels.shape)
    all_epsilons = [np.max(max_norm_dist(sim[labels==kc],centers[kc])) for kc in np.sort(np.unique(labels))]
    P = op_calc.transition_matrix(labels,1)
    prob = op_calc.stationary_distribution(P)
    prob_s = prob
    eps_s = np.array(all_epsilons)

    
    print('Saving results ...')
    #change output path
    f = h5py.File('/flash/StephensU/antonio/Lorenz/dimension_results/scaling_properties_nclusters_{}.h5'.format(n_seeds),'w')
    probs_ = f.create_dataset('probs',prob_s.shape)
    probs_[...] = prob_s
    eps_ = f.create_dataset('eps_scale',eps_s.shape)
    eps_[...] = eps_s
    f.close()
    end_t = time.time()
    print('It took {} s.'.format(end_t-start_t))

if __name__ == "__main__":
    main(sys.argv)
