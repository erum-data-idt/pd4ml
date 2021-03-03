import os
import tensorflow as tf
import numpy as np
import pandas as pd
import awkward
import uproot_methods

from template import NetworkABC
from erum_data_data.erum_data_data import TopTagging

max_part = 200
max_part_pad = 100

def _transform(dataframe, start=0, stop=-1, jet_size=0.8):
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

def convert(X, step=None):
    cols = []
    for i in range(max_part):
        cols.append('E_%d'%(i))
        cols.append('PX_%d'%(i))
        cols.append('PY_%d'%(i))
        cols.append('PZ_%d'%(i))

    df = pd.DataFrame(X, index = None, columns = cols)
    v = _transform(df)
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

class Dataset(object):

    def __init__(self, vec, feature_dict = {}, pad_len=max_part_pad, K=7, data_format='channel_first'):
        self.vec = vec
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel']
            feature_dict['mask'] = ['part_pt_log']
        self.pad_len = pad_len
        self.K = K
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._load()

    def _load(self):
        counts = None
        a = self.vec
        for k in self.feature_dict:
            cols = self.feature_dict[k]
            if not isinstance(cols, (list, tuple)):
                cols = [cols]
            arrs = []
            for col in cols:
                if counts is None:
                    counts = a[col].counts
                else:
                    assert np.array_equal(counts, a[col].counts)
                #L: need to put 9999 for zero padding;
                #L: 0 is a bad default ('fake' neighbours)
                if ('etarel' in col or 'phirel' in col):#L
                    value=9999.#L
                else:#L
                    value=0.#L
                arrs.append(pad_array(a[col], self.pad_len, value=value))
            self._values[k] = np.stack(arrs, axis=self.stack_axis)
        data_size = self._values['points'].shape[0]
        step=min(10000,data_size)
        idx=-1
        full_arr = []
        while True:
            idx+=1
            start=idx*step
            if start>=data_size: break
            temp_adj = to_adj(self._values['points'][start:start+step,:,:],self._values['mask'][start:start+step,:,:],self.K)
            full_arr.append(temp_adj)
        self._values['adj_matrix'] = np.vstack(full_arr)

    def __getitem__(self, key):
        return self._values[key]
    
    @property
    def X(self):
        return self._values        

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

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr *= 0.01
    elif epoch > 10:
        lr *= 0.1
    return lr




class _DotDict:
    pass


class Network(NetworkABC):

    model_name = '_simple_graph_'

    def __init__(self):
        pass


    metrics   = [tf.keras.metrics.BinaryAccuracy(name = "acc")]  ##list of metrics to be used
    compile_args = {'loss':'binary_crossentropy',#'categorical_crossentropy'
                    'optimizer':tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                    'metrics': metrics
                   }                      ##dictionary of the arguments to be passed to the method compile()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True),
                 tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                 tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                  min_delta =0.0001,
                                                  patience=15,
                                                  restore_best_weights = True),
                ]                                              ##list of callbacks to be used in model.
    fit_args = {'batch_size': 1024,
                'epochs': 100,
                'validation_split': 0.2,
                'shuffle': True,
                'callbacks': callbacks
               }                      ##dictionary of the arguments to be passed to the method fit()

    compatible_datasets = [TopTagging]         ## we would also ask you to add a list of the datasets that would be compatible with your implementation 

    def preprocessing(self, X):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routine, it should return the modified data
        in the desired shapes
        """
        X[0] = X[0][:,0:max_part,:]
        X = np.reshape(X[0], (len(X[0]), (X[0].shape[1]*X[0].shape[2])))
        v = convert(X)
        dataset = Dataset(v, data_format='channel_last')
        return dataset.X

    def get_shapes(self, input_dataset):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routine, it should return the modified data
        in the desired shapes
        """
        input_shapes = {k:input_dataset[k].shape[1:] for k in input_dataset}
        return input_shapes

    def model(self,ds,shapes):
        """
        ...
        Parameters
        ----------
        input_shapes : dict
            The shapes of each input (`points`, `features`, `mask`, `adj_matrix`).
        """
        units = 256
        feature_input = tf.keras.layers.Input(shape=shapes['features'], name="features")
        adjacency_input = tf.keras.layers.Input(shape=shapes['adj_matrix'], name="adj_matrix")

        adj_l = adjacency_input

        #symmetric
        #adj_l = adj_l + tf.linalg.matrix_transpose(adj_l)

        #have values either 0 or 1
        #(the adding above could produce higher values than 1)
        #adj_l = tf.cast(adj_l != 0, dtype=tf.float32)#--->  gives error, not needed

        #normalization
        #calculate outer product of degree vector and multiply with adjaceny matrix
        #deg_diag = tf.reduce_sum(adj_l, axis=2)
        #deg12_diag = tf.where(deg_diag > 0, deg_diag ** -0.5, 0)
        #adj_l = (
        #    tf.matmul(
        #        tf.expand_dims(deg12_diag, axis=2),
        #        tf.expand_dims(deg12_diag, axis=1),
        #    )
        #    * adj_l
        #)

        p = feature_input
        for i in range(3):
            p = tf.keras.layers.Dense(units, activation="relu")(p)

        for i in range(3):
            p = SimpleGCN(units, activation="relu")([p, adj_l])

        x = tf.keras.layers.GlobalAveragePooling1D()(p)

        for i in range(3):
            x = tf.keras.layers.Dense(units, activation="relu")(x)

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(inputs=[feature_input,adjacency_input], outputs=outputs, name=ds.name)


class SimpleGCN(tf.keras.layers.Layer):
    """
    Simple graph convolution. Should be equivalent to Kipf & Welling (https://arxiv.org/abs/1609.02907)
    when fed a normalized adjacency matrix.
    """

    def __init__(self, units, activation="relu"):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        feat, adjacency = inputs
        return self.activation(tf.matmul(adjacency, self.dense(feat)))
