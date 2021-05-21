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
from scipy.integrate import trapz

def get_model(labels,delay):
    lcs,P=op_calc.transition_matrix(labels,delay,return_connected=True)
    inv_measure = op_calc.stationary_distribution(P)
    final_labels = op_calc.get_connected_labels(labels,lcs)
    R = op_calc.get_reversible_transition_matrix(P)
    eigvals,eigvecs = op_calc.sorted_spectrum(R,k=2)
    eigfunctions = eigvecs.real/np.linalg.norm(eigvecs.real,axis=0)
    phi2 = eigfunctions[:,1]   
    c_range,rho_sets,thresh_idx,kmeans_labels = op_calc.optimal_partition(phi2,inv_measure,P,return_rho=True)
    return P,phi2,thresh_idx


def organize_matrix(P,phi2):
    y = np.argsort(phi2)
    S = lil_matrix(P.shape)
    S[np.arange(y.shape[0]),y]=1
    A = S*P.T*S.T
    return A


def compute_fpt_dist(labels,delay,kd_max=10000,tol=1e-10,dt=1/16.):
    P,phi2,thresh_idx = get_model(labels,delay)
    A = organize_matrix(P,phi2)
    
    eigvals,eigvecs = op_calc.sorted_spectrum(A,k=2)
    A_dense = A.todense()
    p_star = eigvecs[:,0].real/eigvecs[:,0].real.sum()
    Aaa = A_dense[:thresh_idx,:thresh_idx]
    Abb = A_dense[thresh_idx:,thresh_idx:]
    Aba = A_dense[:thresh_idx,thresh_idx:]
    Aab = A_dense[thresh_idx:,:thresh_idx]
    w_a = p_star[:thresh_idx].reshape(-1,1)
    w_b = p_star[thresh_idx:].reshape(-1,1)
    ua = np.ones((1,Aaa.shape[0]))
    ub = np.ones((1,Abb.shape[0]))
    print(kd_max,flush=True)
    kd_range = np.arange(kd_max)
    f = np.zeros((kd_range.shape[0],2))
    for kd in kd_range:
        f[kd,0] = (ub*Aab*(Aaa**kd)*Aba*w_b)/(ua*Aba*w_b)
        f[kd,1] = (ua*Aba*(Abb**kd)*Aab*w_a)/(ub*Aab*w_a)
        if np.all(np.array([1.,1.])-f[:kd+1,:].sum(axis=0)<tol):
            break
        print(kd,f[:kd+1,:].sum(axis=0),flush=True)
    return kd_range,f,trapz(kd_range.reshape(-1,1)*f,kd_range,axis=0)*delay*dt*.5


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_clusters','--NClusters',help="num clusters",default=1000,type=int)
    parser.add_argument('-idx','--Idx',help="traj idx",default=12,type=int)
    parser.add_argument('-delay0','--D0',help="first delay",default=2,type=int)
    parser.add_argument('-delayf','--Df',help="last delay",default=400,type=int)
    args=parser.parse_args()
    n_clusters = args.NClusters
    d0,df = args.D0,args.Df

    delay_range = np.arange(d0,df,2)
    print(delay_range,flush=True)

    t0_ = time.time()
    #change path to split symbolic sequences
    f = h5py.File('/flash/StephensU/antonio/Foraging/split_trajs/split_dtrajs_{}_clusters.h5'.format(n_clusters),'r')
    d_ = f[str(args.Idx)]
    labels = ma.array(d_['traj'],dtype=int)
    mask = np.array(d_['mask'],dtype=int)
    labels[mask==1] = ma.masked
    seq_length = np.array(f['seq_length'],dtype=int)[0]
    f.close()
    
    print(labels[:10],labels.shape,labels.compressed().shape,flush=True)
    dt = 1/16. 
    #change output path
    outpath = '/flash/StephensU/antonio/Foraging/kinetic_properties/'
    
    f = h5py.File(outpath+'kinetic_properties_clusters_{}_idx_{}_delays_{}_{}.h5'.format(n_clusters,args.Idx,d0,df),'w')
    fd_delay = f.create_group('full_dist')
    kd_delay = f.create_group('evaluated_steps')
    timps = np.zeros((delay_range.shape[0],2))
    for kd,delay in enumerate(delay_range):
        try:
            kd_range,fd,tscales = compute_fpt_dist(labels,delay)
            timps[kd] = np.sort(tscales)
            print(delay*dt,fd.sum(axis=0),timps[kd],flush=True)
            fd_save = fd_delay.create_dataset(str(kd),fd.shape)
            fd_save[...] = fd
            kd_save = kd_delay.create_dataset(str(kd),kd_range.shape)
            kd_save[...] = kd_range
        except:
            print('Bug for {} delays'.format(delay))
    timps_ = f.create_dataset('timps',timps.shape)
    timps_[...]=timps
    f.close()

    tf_ = time.time()
    print('It took {:.2f}s'.format(tf_-t0_))

if __name__ == "__main__":
    main(sys.argv)
