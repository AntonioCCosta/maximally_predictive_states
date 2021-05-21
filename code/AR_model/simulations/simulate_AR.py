import numpy as np
import numpy.ma as ma
import h5py
import matplotlib.pyplot as plt
import sys
import argparse
from scipy.integrate import odeint

def gen_obs(theta,y,lag=1):
    '''
    Returns a simulation of an AR process
    theta are the parameters of the model
    y is the interval over which we want to simulate
    '''
    size_y = y.shape[0]
    sim = np.zeros(size_y,dtype=np.float64)
    #decompose theta
    inter = theta[0]
    coef = theta[1:lag+1]
    std = theta[-1]
    #draw an error vector with the given variance and mean 0
    eps = np.random.normal(0,std,size_y)
    #define the initial condition for the simulation
    sim[:lag]=y[:lag]
    for i in range(lag,len(sim)):
        past = np.hstack([sim[i-1-p] for p in range(lag)])
        sim[i]=inter+np.dot(coef,past)+eps[i]
    return sim

def main():
    
    T = 5000000
    discard_t = 100
    T_total = T+discard_t
    ts = np.arange(0, T)

    theta = np.array([0,1.88,-.95,1])
    y = np.zeros(T_total)
    sim = gen_obs(theta,y,2)[discard_t:]
    
    print('Saving results...')
    
    output_path = './'
    f = h5py.File(output_path+'simulation_Kantz.h5','w')
    metaData = f.create_group('MetaData')
    th_ = metaData.create_dataset('theta',theta.shape)
    th_[...] = theta
    t_disc = metaData.create_dataset('discard_t',(1,))
    t_disc[...] = discard_t
    total_T = metaData.create_dataset('T',(1,))
    total_T[...] = T
    tseries = f.create_dataset('simulation',sim.shape)
    tseries[...] = sim
    f.close()
    

if __name__ == "__main__":
    main()
