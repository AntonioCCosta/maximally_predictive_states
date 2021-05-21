import os
import h5py
import numpy as np
import numpy.ma as ma
import argparse
import sys
sys.path.append('/home/a/antonio-costa/theory_manuscript/') #change path
import operator_calculations as op_calc
import msmtools.estimation as msm_estimation


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_clusters','--N',help="num clusters",default=1000,type=int)
    parser.add_argument('-idx','--Idx',help="traj idx",default=12,type=int)

    args=parser.parse_args()
    n_clusters=args.N
    idx = args.Idx
    
    delay_range = np.arange(1,9600,1)

    #change path to split symbolic trajectories
    f = h5py.File('/flash/StephensU/antonio/Foraging/split_trajs/split_dtrajs_{}_clusters.h5'.format(n_clusters),'r')
    d_ = f[str(idx)]
    labels = ma.array(d_['traj'],dtype=int)
    mask = np.array(d_['mask'],dtype=int)
    labels[mask==1] = ma.masked
    seq_length = np.array(f['seq_length'],dtype=int)[0]
    f.close()

    print(seq_length,flush=True)

    #labels = labels[:2000]
    #delay_range = delay_range[:2]
    print(labels[:100])
    print(labels.shape)
    print(labels.compressed().shape)

    n_modes = 100
    
    dt = 1/16.
    nstates = n_clusters
    ts_traj = np.zeros((len(delay_range),n_modes))
    for kd,delay in enumerate(delay_range):
        count_ms = op_calc.get_count_ms(labels,delay,nstates)
        connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
        P = msm_estimation.tmatrix(connected_count_matrix)
        R = op_calc.get_reversible_transition_matrix(P)
        ts_traj[kd,:] = op_calc.compute_tscales(R,delay,dt,k=n_modes+1)
        print(delay,flush=True)

    #change output path
    output_path = '/flash/StephensU/antonio/Foraging/tscales_analysis_per_worm/tscales_clusters_{}/'.format(n_clusters)

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
