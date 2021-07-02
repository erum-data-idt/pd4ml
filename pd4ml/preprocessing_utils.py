import pandas as pd
import awkward
import uproot_methods
import multiprocessing as mp
import platform
import numpy as np

##############################################################################
#                           Top Tagging utils                                #
##############################################################################
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
    # ie the output of https://github.com/erikbuh/pd4ml/blob/main/models/particle_net.py#L186

    D = batch_distance_matrix_numpy(points, points)  # (N, P, P)
    #L
    #argsort starts from the smaller values, no need to put a negative sign
    #care must be taken with how zero-padded elements are handled
    indices = np.array((D).argsort(axis=-1)[:, :, :K+1]) # (N, P, K+1)
    matrix = indices[:, :, 1:]  # (N, P, K)

    # get the shapes
    B,P,k_max = matrix.shape

    # create initial adjacency matrix (all zeros)
    adj = np.zeros((B,P,P), dtype='int8')

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
        adj[i_batch,rows,cols] = 1

    # diagonal
    adj[i_batch,rows,rows] = 1

    #set to zero elements of adj related to empty particles:
    #mask --> transform in a proprer size matrix
    mask_row = np.tile(mask,adj.shape[2])
    #create boolean mask
    mask_row[ np.where( ( np.transpose(mask_row,(0,2,1)) )==0 ) ]=0

    #find  the indices and broadcast to adj
    adj[np.where(mask_row==0)]=0
    
    #transpose
    adj = np.transpose(adj,(0,2,1))
    adj = adj.astype('int8')
    return adj


def load_top(v, feature_dict, pad_len, stack_axis, K = None, build_graph = True):
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
            value = 0.
            arrs.append(pad_array(v[col], pad_len, value=value))
        values[k] = np.stack(arrs, axis=stack_axis)

    if build_graph:
        data_size = values['points'].shape[0]
        step = min(10000,data_size)
        idx = -1
        full_arr = []
        params_list = []
        while True:
            idx += 1
            start = idx*step
            if start >= data_size: break
            params = [values['points'][start:start+step,:,:],values['mask'][start:start+step,:,:],K]
            params_list.append(params)
            if platform.system() == 'Windows':
                temp_adj = to_adj(values['points'][start:start+step,:,:],values['mask'][start:start+step,:,:],K)
                full_arr.append(temp_adj)
        
    ## multiprocess the functions execution (only on Linux based systems)
        if platform.system() != 'Windows':
            with mp.Pool(mp.cpu_count()-1) as p:    # using all available CPUs except 1
                full_arr = p.map(_multiprocess_adj, params_list)
            
        values['adj_matrix'] = np.vstack(full_arr)
        return values['features'], values['adj_matrix']
    return values['features']
    
def _multiprocess_adj(params):
    return to_adj(params[0],params[1],params[2])




##############################################################################
#                               Belle utils                                  #
##############################################################################


def adjacency_matrix_from_mothers_np(mother_indices, symmetrize=True, add_diagonal=True):
    """
    Calculate adjacency matrix from mother indices (numpy version). Assumes
    that mother indices are -1 padded.
    """
    N = mother_indices.shape[1]
    adj = np.eye(N + 1, dtype=np.int8)[mother_indices][:, :, :-1]
    if symmetrize:
        adj = adj + np.transpose(adj, (0, 2, 1))
    if add_diagonal:
        adj += np.where(
            (mother_indices != -1)[:, :, np.newaxis],
            np.eye(N, dtype=np.int8),
            0
        )
    adj[adj > 1] = 1
    return adj


def adjacency_matrix_from_mothers_tf(mother_indices, symmetrize=True, add_diagonal=True):
    """
    Like `adjacency_matrix_from_mothers_np`, but using tensorflow
    """
    import tensorflow as tf

    shape = tf.shape(mother_indices)
    N = shape[1]
    bs = shape[0]
    inputs = mother_indices

    idx = tf.where(inputs < 0, tf.cast(N, dtype=tf.int64), inputs)
    adj = tf.one_hot(tf.cast(idx, dtype=tf.int32), N + 1)[:, :, :-1]

    if symmetrize:
        adj = adj + tf.linalg.matrix_transpose(adj)

    if add_diagonal:
        diagonal = tf.broadcast_to(tf.eye(N), (bs, N, N))
        diagonal = tf.where(
            tf.repeat(tf.reshape(inputs != -1, (bs, N, 1)), N, axis=2),
            diagonal,
            tf.zeros_like(adj),
        )
        adj = adj + diagonal

    adj = tf.cast(adj != 0, dtype=tf.float32)
    return adj


def get_sorted_pdg():
    from pd4ml import Belle
    data = Belle.load()
    mapped_pdg, counts = np.unique(data[0][0][:, :, -1].ravel(), return_counts=True)
    mapped = mapped_pdg[np.argsort(counts)][::-1].astype(int)
    return mapped[mapped != 0]


# generated with `get_sorted_pdg()`
sorted_pdg = [
   506, 423, 236, 143, 321, 171, 323,  15, 114,  19,  96, 394, 372,
    46,  44, 190, 134, 141, 375,   8, 337, 220, 415, 248, 249,  11,
    34, 434, 263,   4, 126, 378, 418, 183, 182,  10, 245,  85,  45,
   210, 178, 139, 275, 322, 419, 121, 437,  42, 118, 424, 480, 483,
    79, 414,  61, 393,  38, 349, 108,  86,  32, 357, 238, 173, 305,
    57, 413, 491, 158, 355,  78, 176, 164, 352, 187, 205, 471,  71,
   458, 350, 101, 280, 389, 376,  89, 120, 457, 436, 186, 444,  76,
   390, 402, 342, 258, 391, 407, 162, 185, 152, 276, 149, 302, 106,
   110, 401, 109, 487, 279, 192, 446, 334,  56, 359, 194, 314, 202,
   420, 208, 427, 475, 495, 465, 442, 489, 360, 448, 216, 215, 300,
   214,  13, 227, 501, 502, 399, 344, 474, 150,  53, 325, 129, 326,
   195, 366, 370, 336, 163, 154,  66, 105, 285, 473, 455, 421,  43,
   119, 271, 373, 363, 428, 304, 229, 425, 133, 270, 219, 479, 112,
   250, 217,  33, 403, 338, 206,  39, 268, 130, 328,  93, 346, 212
]


def remap_pdg(mapped_pdg):
    """
    Remap mapped pdg ids again for later one-hot encoding to `num_pdg`
    number of dimensions. They are sorted by number of occurence.
    """

    remap_dict = {v : k for k, v in enumerate(sorted_pdg)}

    @np.vectorize
    def remap(pdg_id):
        return remap_dict.get(pdg_id, len(sorted_pdg) + 1)

    return remap(mapped_pdg)


def np_onehot(indices, depth):
    """
    Behaviour like tf.one_hot
    """
    indices = np.array(indices)
    indices[indices >= depth] = depth
    return np.eye(depth + 1, dtype=np.int8)[indices][..., :-1]


def test_belle_graphs_tf_np_consistency():

    from collections import defaultdict
    
    max_entries = 1000
    x_train, y_train = LoadGraph.belle_graph('train', path = './datasets', max_entries=max_entries)
    ds_train = LoadGraph.belle_graph(
        'train',
        path = './datasets',
        validation_split=None,
        max_entries=max_entries,
        batch_size=64,
        as_tf_data=True
    )
    x_train_tf = defaultdict(list)
    for batch in ds_train:
        for k in x_train:
            x_train_tf[k].append(batch[0][k].numpy())
    for k in x_train_tf:
        assert (x_train[k] == np.concatenate(x_train_tf[k])).all()

 
##############################################################################
#                        2D structured data util                             #
##############################################################################


              
def _adjacency_matrix_img(inputs):
    """
    calculate 4-connected adjacency matrix for images
    """
    
    shape = inputs.shape
    n = shape[1]
    N = n*n
    bs = shape[0]

    adj = np.zeros((N, N), dtype='int8')
    for i in range(N):
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
  
def _adjacency_matrix_img_8connected(inputs):
    """
    calculate the 8-connected adjacency matrix for images
    """
    
    shape = inputs.shape
    n = shape[1]
    N = n*n
    bs = shape[0]

    adj = np.zeros((N, N), dtype='int8')
    for i in range(N):
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
            
        if i-n > 0 and (i)%n!=0:
            adj[i-(n+1)][i] = 1
            adj[i][i-(n+1)] = 1
        if i-n > 0 and i+1 < N and (i+1)%n!=0:
            adj[i-(n-1)][i] = 1
            adj[i][i-(n-1)] = 1
        if i+n < N and i-1 >= 0 and (i)%n!=0:
            adj[i+(n-1)][i] = 1
            adj[i][i+(n-1)] = 1
        if i+n < N and (i+1)%n!=0:
            adj[i+(n+1)][i] = 1
            adj[i][i+(n+1)] = 1
    adj = np.broadcast_to(adj, [bs, N, N])
    return adj  
