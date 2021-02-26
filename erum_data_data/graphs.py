import numpy as np
import pandas as pd
import awkward
import uproot_methods

class LoadGraph:
    """
    Transformation from the dataset to a graph 
    
    returns graph features, the adjacency matrix and a mask (per default all True)
    """
    
    def spinodal_graph(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import Spinodal
        """
        transforms the Spinodal dataset into a graph
        """
        
        X,y = Spinodal.load(split, path, force_download)
        X_feats = X[0]
        
        X_adj = _adjacency_matrix_img(X_feats)
        X_mask = _dummy_mask(shape=X_feats.shape[0])
        
        return X_feats, X_adj, X_mask
    
    
    def eosl_graph(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import EOSL
        """
        transforms the EOSL dataset into a graph
        """
        
        X,y = EOSL.load(split, path, force_download)
        X_feats = X[0]
        
        X_adj = _adjacency_matrix_img(X_feats)
        X_mask = _dummy_mask(shape=X_feats.shape[0])
        
        return X_feats, X_adj, X_mask
        
    def TopTagging_graph(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import TopTagging
        """
        transforms the TopTagging dataset into a graph
        """
        
        X,y = TopTagging.load(split, path, force_download)
        #Currently hardcoded:
        #K
        #max_part
        #max_part_pad
        K = 7
        max_part = 200
        max_part_pad = 100

        #Lisa
        print("reduced size for testing:")
        X[0] = X[0][0:10000,0:max_part,:]
        y = y[0:10000]
        #X[0] = X[0][:,0:max_part,:]
        X = np.reshape(X[0], (len(X[0]), (X[0].shape[1]*X[0].shape[2])))
        v = convert(X,max_part)
        feature_dict = {}
        feature_dict['points'] = ['part_etarel', 'part_phirel']
        feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel']
        feature_dict['mask'] = ['part_pt_log']
        data_format='channel_last'
        stack_axis = 1 if data_format=='channel_first' else -1
        X_feats, X_adj = load_top(v,feature_dict,K,max_part_pad,stack_axis)
        
        return X_feats, X_adj, y
        
        
        
        
        
        
        
            
def _dummy_mask(shape=(1,1)):
    """
    returns a dummy mask with all True in a specified shape
    """
    return np.ones(shape, dtype=bool)
            
            
        
def _adjacency_matrix_img(inputs):
    """
    calculate adjacency matrix for images
    """
    
    shape = inputs.shape
    n = shape[1]
    N = n*n
    bs = shape[0]

    adj = np.zeros((N, N))
    for i in range(N):
        adj[i][i] = 1
        if i+1 < N and (i+1)%n!=0:
            adj[i][i+1] = 1
            adj[i+1][i] = 1
        if i+n < N:
            adj[i+n][i] = 1
            adj[i][i+n] = 1
        if i-n > 0:
            adj[i-n][i] = 1
            adj[i][i-n] = 1
        if i-1 > 0 and (i)%n!=0:
            adj[i-1][i] = 1
            adj[i][i-1] = 1
    adj = np.broadcast_to(adj, [bs, N, N])
    return adj  

def _transform(dataframe, max_part, start=0, stop=-1, jet_size=0.8):
    from collections import OrderedDict
    v = OrderedDict()

    if stop==-1:
        df = dataframe.iloc[:]
    else:
        df = dataframe.iloc[start:stop]
    def _col_list(prefix, max_particles=max_part):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]
    
    _px = df[_col_list('PX')].values
    _py = df[_col_list('PY')].values
    _pz = df[_col_list('PZ')].values
    _e = df[_col_list('E')].values
    
    mask = _e>0
    n_particles = np.sum(mask, axis=1)

    px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward.JaggedArray.fromcounts(n_particles, _e[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    pt = p4.pt

    jet_p4 = p4.sum()

    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_mass'] = jet_p4.mass
    v['n_parts'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt/v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(energy)
    v['part_erel'] = energy/jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])

    def _make_image(var_img, rec, n_pixels = 64, img_ranges = [[-0.8, 0.8], [-0.8, 0.8]]):
        wgt = rec[var_img]
        x = rec['part_etarel']
        y = rec['part_phirel']
        img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))
        for i in range(len(wgt)):
            hist2d, xedges, yedges = np.histogram2d(x[i], y[i], bins=[n_pixels, n_pixels], range=img_ranges, weights=wgt[i])
            img[i] = hist2d
        return img

    return v

def convert(X, max_part, step=None):
    cols = []
    for i in range(max_part):
        cols.append('E_%d'%(i))
        cols.append('PX_%d'%(i))
        cols.append('PY_%d'%(i))
        cols.append('PZ_%d'%(i))

    df = pd.DataFrame(X, index = None, columns = cols)
    v = _transform(df,max_part)
    return v

def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)

def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x

def batch_distance_matrix_numpy(A, B):
    r_A = np.sum(A * A, axis=2, keepdims=True)
    r_B = np.sum(B * B, axis=2, keepdims=True)
    m = np.matmul(A, np.transpose(B,(0, 2, 1)))
    D = r_A - 2 * m + np.transpose(r_B,(0, 2, 1))
    return D

def to_adj(points,mask,K):
    # takes input of the shape batch (B), number of particles (P), list of k-nearest neighbor indices (k_max)
    # ie the output of https://github.com/erikbuh/erum_data_data/blob/main/models/particle_net.py#L186

    D = batch_distance_matrix_numpy(points, points)  # (N, P, P)
    #L
    #argsort starts from the smaller values, no need to put a negative sign
    #care must be taken with how zero-padded elements are handled
    indices = np.array((D).argsort(axis=-1)[:, :, :K+1]) # (N, P, K+1)
    matrix = indices[:, :, 1:]  # (N, P, K)

    # get the shapes
    B,P,k_max = matrix.shape

    # create initial adjacency matrix (all zeros)
    adj = np.zeros((B,P,P))

    # produce batch indices
    # will be 000111222...
    i_batch = np.repeat(np.arange(B), P)
    # produce row indices of particles
    # will be 012012012
    rows = np.tile(np.arange(P),B)
    #mask the empty constituents
    #put a 1 when particle is valid
    m = (mask)!=0
    np.place(mask, m, 1)

    # loop over k nearest neighbours
    for k in range(k_max):
        # produce column indices
        # reading the k-est index at the approriate place from the input
        cols = matrix[:,:,k].flatten()
        # use fancy indexing of batch/row/column lists 
        # together to change the elements of the adjacency 
        # matrix for this position from 0 to 1
        adj[i_batch,rows,cols] = 1
        # make symmetric --> asymmetric models give better scores
        #adj[i_batch,cols,rows] = 1

    # diagonal
    adj[i_batch,rows,rows] = 1

    #set to zero elements of adj related to empty particles:
    #mask --> transform in a proprer size matrix
    mask_row = np.tile(mask,adj.shape[2])
    #create boolean mask
    mask_row[ np.where( ( np.transpose(mask_row,(0,2,1)) )==0 ) ]=0

    #find  the indices and broadcast to adj
    adj[np.where(mask_row==0)]=0

    ##too expensive computationally
    #tr_mask = np.transpose(mask,(0,2,1))
    #adj = tr_mask*adj
    #adj = adj*mask
    
    #optional: normalize rows
    #row_sums = adj.sum(axis=2)
    #row_sums[row_sums == 0] = 1
    #adj = adj / row_sums[:, np.newaxis]

    #optional: normalize columns
    #col_sums = adj.sum(axis=1)
    #col_sums[col_sums == 0] = 1
    #adj = adj / col_sums[:, np.newaxis]

    #transpose
    adj = np.transpose(adj,(0,2,1))

    # return the adjacency matrix
    return adj


def load_top(v,feature_dict,K,pad_len,stack_axis):
    values = {}
    counts = None
    for k in feature_dict:
        cols = feature_dict[k]
        if not isinstance(cols, (list, tuple)):
            cols = [cols]
        arrs = []
        for col in cols:
            if counts is None:
                counts = v[col].counts
            else:
                assert np.array_equal(counts, v[col].counts)
            #L: need to put 9999 for zero padding;                                                                              
            #L: 0 is a bad default ('fake' neighbours)                                                                          
            if ('etarel' in col or 'phirel' in col):#L                                                                          
                value=9999.#L                                                                                                   
            else:#L                                                                                                             
                value=0.#L                                                                                                      
            arrs.append(pad_array(v[col], pad_len, value=value))

        #define values!!!! and stack_axis
        values[k] = np.stack(arrs, axis=stack_axis)
    data_size = values['points'].shape[0]
    step=min(10000,data_size)
    idx=-1
    full_arr = []
    while True:
        idx+=1
        start=idx*step
        if start>=data_size: break
        temp_adj = to_adj(values['points'][start:start+step,:,:],values['mask'][start:start+step,:,:],K)
        full_arr.append(temp_adj)
    values['adj_matrix'] = np.vstack(full_arr)
    return values['features'], values['adj_matrix']
