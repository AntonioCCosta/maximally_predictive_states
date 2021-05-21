import os
import h5py
import numpy as np
import numpy.ma as ma
import argparse
import sys
#change path
sys.path.append('/home/a/antonio-costa/theory_manuscript/')
import operator_calculations as op_calc
import time
from scipy.sparse import csr_matrix,lil_matrix


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_clusters','--NClusters',help="num clusters",default=1000,type=int)
    parser.add_argument('-idx','--Idx',help="traj idx",default=12,type=int)
    args=parser.parse_args()
    n_clusters = args.NClusters
    
    #delay_range = np.array([10,50,100,500])

    t0_ = time.time()
    delay_range = np.arange(1,400,1)
    #change path to symbolic sequence chunks
    f = h5py.File('/flash/StephensU/antonio/Foraging/split_trajs/split_dtrajs_{}_clusters.h5'.format(n_clusters),'r')
    d_ = f[str(args.Idx)]
    labels = ma.array(d_['traj'],dtype=int)
    mask = np.array(d_['mask'],dtype=int)
    labels[mask==1] = ma.masked
    seq_length = np.array(f['seq_length'],dtype=int)[0]
    f.close()
    
    print(labels[:10],labels.shape,labels.compressed().shape,seq_length,flush=True)
    dt = 1/16    
    #change output path
    outpath = '/flash/StephensU/antonio/Foraging/kinetic_properties/'
    
    f = h5py.File(outpath+'coarse_properties_{}_clusters_{}.h5'.format(n_clusters,args.Idx),'w')
    #delay_range=np.arange(200,1200,200)
    timp = np.zeros(delay_range.shape[0])
    for kd,delay in enumerate(delay_range):
        try:
            lcs,P=op_calc.transition_matrix(labels,delay,return_connected=True)
            inv_measure = op_calc.stationary_distribution(P)
            final_labels = op_calc.get_connected_labels(labels,lcs)
            R = op_calc.get_reversible_transition_matrix(P)
            eigvals,eigvecs = op_calc.sorted_spectrum(R,k=2)
            eigfunctions = eigvecs.real/np.linalg.norm(eigvecs.real,axis=0)
            phi2 = eigfunctions[:,1]   
            c_range,rho_sets,thresh_idx,kmeans_labels = op_calc.optimal_partition(phi2,inv_measure,P,return_rho=True)
            cluster_traj = ma.copy(final_labels)
            cluster_traj[~final_labels.mask] = ma.array(kmeans_labels)[final_labels[~final_labels.mask]]
            cluster_traj[final_labels.mask] = ma.masked
            P_coarse = op_calc.transition_matrix(cluster_traj,delay)
            eigvals = np.linalg.eigvals(P_coarse.todense())
            print(delay,eigvals,flush=True)
            timp[kd] = -(delay*dt)/np.log(np.min(np.abs(eigvals)))
        except:
            print('Error for kd={}'.format(kd),flush=True)
            continue
            #print(kd,flush=True)
    timps_ = f.create_dataset('timp',timp.shape)
    timps_[...]=timp
    kds_ = f.create_dataset('delay_range',delay_range.shape)
    kds_[...] = delay_range
    f.close()

    tf_ = time.time()
    print('It took {:.2f}s'.format(tf_-t0_))

if __name__ == "__main__":
    main(sys.argv)
