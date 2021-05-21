import os
import h5py
import numpy as np
import numpy.ma as ma
import argparse
import sys
#path to mother scripts
sys.path.append('/home/a/antonio-costa/theory_manuscript/')
import new_op_calc as op_calc
import msmtools.estimation as msm_estimation


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_clusters','--N',help="num clusters",default=5,type=int)
    parser.add_argument('-idx','--Idx',help="traj idx",default=19,type=int)
    parser.add_argument('-T_idx','--T_IDX',help="temp idx",default=0,type=int)

    args=parser.parse_args()
    n_clusters=args.N
    idx = args.Idx
    
    k_B_T = np.arange(0.5,2.6,0.25)[args.T_IDX]

    print(k_B_T)

    dt = .05
    
    delay_range = np.arange(1,1200,1)

    #path to split symbolic trajectories
    f = h5py.File('/flash/StephensU/antonio/DoubleWell/split_trajs_longer/split_dtrajs_{}_clusters_{}_T.h5'.format(n_clusters,k_B_T),'r')
    d_ = f[str(idx)]
    labels = ma.array(d_['traj'],dtype=int)
    mask = np.array(d_['mask'],dtype=int)
    labels[mask==1] = ma.masked
    seq_length = np.array(f['seq_length'],dtype=int)[0]
    f.close()

    #labels = labels[:2000]
    #delay_range = delay_range[:2]
    print(labels[:100])
    print(labels.shape)
    print(labels.compressed().shape)
    print(seq_length)

    n_modes = 3
    
    nstates = n_clusters
    ts_traj = np.zeros((len(delay_range),n_modes))
    for kd,delay in enumerate(delay_range):
        count_ms = op_calc.get_count_ms(labels,delay,nstates)
        connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
        P = msm_estimation.tmatrix(connected_count_matrix)
        R = op_calc.get_reversible_transition_matrix(P)
        ts_traj[kd,:] = op_calc.compute_tscales(R,delay,dt,k=n_modes+1)
        print(delay)

    #change output path
    output_path = '/home/a/antonio-costa/theory_manuscript/DoubleWell_final/tscales_analysis_longer_length/tscales_clusters_{}_T_{}/'.format(n_clusters,k_B_T)

    if os.path.isdir(output_path):
        print('Path already exists')
    else:
        os.mkdir(output_path)
        
    f = h5py.File(output_path+'tscales_{}.h5'.format(idx),'w')
    t_ = f.create_dataset('ts_traj',ts_traj.shape)
    t_[...] = ts_traj
    sl = f.create_dataset('seq_length',(1,))
    sl[...] = seq_length
    dl = f.create_dataset('delay_range',delay_range.shape)
    dl[...] = delay_range
    f.close()
    
if __name__ == "__main__":
    main(sys.argv)
