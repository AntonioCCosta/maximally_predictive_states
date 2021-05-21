import numpy as np
import numpy.ma as ma


def rot_matrix(theta):
    r = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))
    return r

def angle_between(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    angle = sign * np.arccos(dot_p)
    return angle

# @njit(cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_trajs_data(labels_traj,X,vecsize):
    trajs_state=[]
    len_trajs_state = []
    for idx in np.unique(labels_traj.compressed()):
        mask = labels_traj==idx
        segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
        segments = segments[segments[:,0]>int(vecsize/2)]
        trajs = []
        len_trajs = []
        for t0,tf in segments:
            traj = X[t0-int(vecsize/2):tf]
            traj = traj-traj[0]
            trajs.append(traj)
            len_trajs.append(tf-t0)
        trajs_state.append(trajs)
        len_trajs_state.append(np.hstack(len_trajs))
    return trajs_state,len_trajs_state


def get_data_indices(cluster_traj):
    data_indices = []
    t0 = 0
    tf=0
    while tf < len(cluster_traj):
        if tf>=len(cluster_traj)-1:
            break
        if ma.count_masked(cluster_traj[tf])==0:
            while cluster_traj[tf+1]==cluster_traj[tf]:
                if tf+1>=len(cluster_traj)-1 and tf-t0>0:
                    data_indices = np.vstack(data_indices)
                    data_indices = data_indices[data_indices[:,1]>0]
                    return np.vstack(data_indices)
                else:
                    tf+=1
            else:
                data_indices.append([cluster_traj[tf],tf-t0])
                t0=tf
                tf+=1
        else:
            t0 = tf
            tf = t0+1   
    data_indices = np.vstack(data_indices)
    data_indices = data_indices[data_indices[:,1]>0]
    return np.vstack(data_indices)



def sample_traj_dur(trajs_,diff,vecsize,dur):    
    if np.any(diff>=0):
        sel1 = diff==0
        if sel1.sum()>0:
            sample_traj = np.random.choice(np.array(trajs_)[sel1])
        else:
            idx = np.argsort(diff)[np.where(np.sort(diff)>0)[0][0]]
            sel2 = diff==diff[idx]
            sample_traj = np.random.choice(np.array(trajs_)[sel2])[:dur]
    else:
        print('generating longer sequence ',dur)
        sample_traj_seq = np.random.choice(np.array(trajs_),len(trajs_))
        st_=[sample_traj_seq[0],]
        full_traj = np.vstack(st_)
        k=1
        while full_traj.shape[0]<dur:
            st = sample_traj_seq[k]
            vecs1 = np.diff(full_traj[-int(vecsize/2):],axis=0)
            vec1 = np.mean(vecs1,axis=0)
            vecs2 = np.diff(st[:int(vecsize/2)],axis=0)
            vec2 = np.mean(vecs2,axis=0)

            theta = angle_between(vec2,vec1)
            rotated_traj = np.vstack([rot_matrix(-theta).dot(x_) for x_ in st])
            rotated_traj = rotated_traj[int(vecsize/2)-1:] - rotated_traj[int(vecsize/2)-1]
            rt = rotated_traj+full_traj[-1]
            st_.append(rt[1:])
            full_traj = np.vstack(st_)
            k+=1
        sample_traj = full_traj[:dur]
    return sample_traj


def sample_trajectory(cluster_traj,X,n_states,vecsize=4,return_trajs=False):
    #minimum_vecsize=4
    trajs_state,len_trajs_state = get_trajs_data(cluster_traj,X,vecsize)
    data_indices = get_data_indices(cluster_traj)

    #sample first trajectory
    sample_trajs = [np.zeros(2),]
    state,dur = data_indices[0]
    trajs_ = trajs_state[state]
    diff = len_trajs_state[state]-dur
    sample_traj = sample_traj_dur(trajs_,diff,vecsize,dur)
    if sample_traj.shape[0]<=int(vecsize/2):
        sample_trajs.append(sample_traj[1:])
    else:
        sample_trajs.append(sample_traj[int(vecsize/2):])
    full_traj = np.vstack(sample_trajs)
    #sample remaining trajectories
    for state,dur in data_indices[1:]:
        vecs1 = np.diff(full_traj[-int(vecsize/2):],axis=0)
        vec1 = np.mean(vecs1,axis=0)

        trajs_ = trajs_state[state]
        diff = len_trajs_state[state]-dur
        sample_traj = sample_traj_dur(trajs_,diff,vecsize,dur)
        vecs2 = np.diff(sample_traj[:int(vecsize/2)],axis=0)
        vec2 = np.mean(vecs2,axis=0)
        theta = angle_between(vec2,vec1)

        rotated_traj = np.vstack([rot_matrix(-theta).dot(x_) for x_ in sample_traj])
        rotated_traj = rotated_traj[int(vecsize/2)-1:] - rotated_traj[int(vecsize/2)-1]
        sample_traj_ = full_traj[-1]+rotated_traj
        sample_trajs.append(sample_traj_[1:])
        full_traj = np.vstack(sample_trajs)
    if return_trajs:
        return full_traj,sample_trajs
    return full_traj


def sample_trajectory_sim(cluster_traj,cluster_traj_sim,X,n_states,vecsize=4,return_trajs=False):
    #minimum_vecsize=4
    trajs_state,len_trajs_state = get_trajs_data(cluster_traj,X,vecsize)
    data_indices = get_data_indices(cluster_traj_sim)

    #sample first trajectory
    sample_trajs = [np.zeros(2),]
    state,dur = data_indices[0]
    trajs_ = trajs_state[state]
    diff = len_trajs_state[state]-dur
    sample_traj = sample_traj_dur(trajs_,diff,vecsize,dur)
    if sample_traj.shape[0]<=int(vecsize/2):
        sample_trajs.append(sample_traj[1:])
    else:
        sample_trajs.append(sample_traj[int(vecsize/2):])
    full_traj = np.vstack(sample_trajs)
    #sample remaining trajectories
    for state,dur in data_indices[1:]:
        vecs1 = np.diff(full_traj[-int(vecsize/2):],axis=0)
        vec1 = np.mean(vecs1,axis=0)

        trajs_ = trajs_state[state]
        diff = len_trajs_state[state]-dur
        sample_traj = sample_traj_dur(trajs_,diff,vecsize,dur)
        vecs2 = np.diff(sample_traj[:int(vecsize/2)],axis=0)
        vec2 = np.mean(vecs2,axis=0)
        theta = angle_between(vec2,vec1)

        rotated_traj = np.vstack([rot_matrix(-theta).dot(x_) for x_ in sample_traj])
        rotated_traj = rotated_traj[int(vecsize/2)-1:] - rotated_traj[int(vecsize/2)-1]
        sample_traj_ = full_traj[-1]+rotated_traj
        sample_trajs.append(sample_traj_[1:])
        full_traj = np.vstack(sample_trajs)
    if return_trajs:
        return full_traj,sample_trajs
    return full_traj