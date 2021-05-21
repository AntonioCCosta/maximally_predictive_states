import numpy as np
import numpy.ma as ma
from scipy.interpolate import CubicSpline


def unwrapma(x):
    idx= ma.array(np.arange(0,x.shape[0]),mask=x.mask)
    idxc=idx.compressed()
    xc=x.compressed()
    dd=np.diff(xc)
    ddmod=np.mod(dd+np.pi,2*np.pi)-np.pi
    ddmod[(ddmod==-np.pi)&(dd>0)]=np.pi
    phc_correct = ddmod-dd
    phc_correct[np.abs(dd)<np.pi] = 0
    ph_correct = np.zeros(x.shape)
    ph_correct[idxc[1:]] = phc_correct
    up = x + ph_correct.cumsum()
    return up


def compute_phi_omega_a3(tseries,t0,tf,frameRate=16.):
    time=np.arange(t0,tf)
    X=tseries[time]
    phi=-np.arctan2(X[:,1],X[:,0])
    cs = CubicSpline(time, phi)
    phiFilt=cs(time)
    phi_unwrap=unwrapma(phi)
    sel=~phi_unwrap.mask
    cs = CubicSpline(time[sel], phi_unwrap[sel])
    #normalize by frame rate
    phiFilt_unwrap=cs(time[sel])
    omegaFilt=cs(time[sel],1)*frameRate/(2*np.pi)
    return phiFilt,omegaFilt,X[:,2]


def find_zeros(curve,distance_condition,threshold):
    zeros=[]
    t=0
    for t in range(len(curve)-1):
        if distance_condition:
            if curve[t]*curve[t+1]<0:
                if np.any([np.abs(curve[t])<threshold,np.abs(curve[t+1])<threshold]):
                    t_zero=np.array([t,t+1])[np.argmin([np.abs(curve[t]),np.abs(curve[t+1])])]
                    zeros.append(t_zero) 
        else:
            if curve[t]*curve[t+1]<0:
                t_zero=np.array([t,t+1])[np.argmin([np.abs(curve[t]),np.abs(curve[t+1])])]
                zeros.append(t_zero)
    return zeros

def poincare_map(space,z_section,direction,withTime=False,distance_condition=False,threshold=.5):
    r0=z_section[0]
    a=z_section[1]
    U=[]
    for point in space:
        U.append(np.dot((point-r0),a))
    crossing_time=find_zeros(U,distance_condition,threshold)
    x_map=space[crossing_time,0]
    y_map=space[crossing_time,1]
    map_points=np.vstack((space[crossing_time,direction],space[crossing_time,2])).T
    if withTime:
        return map_points,crossing_time
    else:
        return map_points


def model_index(time,windows):
    for k,window in enumerate(windows):
        if window[0]<time<window[1]:
            return k

def corresponding_dynamics(peak_idx,crossing_times_worm,tseries_w,windows_w):
    number_crossings_worm=np.cumsum([len(crossing_times) for crossing_times in crossing_times_worm])
    worm_idx=np.where(peak_idx<number_crossings_worm)[0][0]
    if worm_idx>0:
        new_idx=peak_idx-number_crossings_worm[worm_idx-1]
    else:
        new_idx=peak_idx
    corresponding_time=crossing_times_worm[worm_idx][new_idx]
    t0=corresponding_time-1
    tf=corresponding_time+1
    tseries=tseries_w[worm_idx]
    phi,omega,a3=compute_phi_omega_a3(tseries,t0,tf,frameRate=16.)
    ws=windows_w[worm_idx]
    model_idx=model_index(corresponding_time,ws)
    return worm_idx,model_idx,phi,omega,a3

def sim_pred(window,tseries,theta,lag=1):
    t0,tf=window
    y=tseries[t0:tf]
    yp=lvar.get_yp(y,lag)
    inter,coef,sigma=lvar.decomposed_theta(theta)
    beta=np.vstack((inter,coef))
    x_inter=ma.vstack((ma.ones(yp.shape[0]),yp.T)).T
    pred=ma.zeros(y.shape)
    pred[0]=ma.masked
    pred[1:]=ma.dot(x_inter,beta)
    return pred
