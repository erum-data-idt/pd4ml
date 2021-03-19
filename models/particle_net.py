import os
import tensorflow as tf
import numpy as np
import pandas as pd
import awkward
import uproot_methods

from template import NetworkABC
from erum_data_data.erum_data_data import TopTagging


def _transform(dataframe, start=0, stop=-1, jet_size=0.8):
    from collections import OrderedDict
    v = OrderedDict()

    if stop==-1:
        df = dataframe.iloc[:]
    else:
        df = dataframe.iloc[start:stop]
    def _col_list(prefix, max_particles=200):
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
    for i in range(200):
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

    def __init__(self, vec, feature_dict = {}, pad_len=100, data_format='channel_first'):
        self.vec = vec
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel']
            feature_dict['mask'] = ['part_pt_log']
        self.pad_len = pad_len
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
                arrs.append(pad_array(a[col], self.pad_len))
            self._values[k] = np.stack(arrs, axis=self.stack_axis)

    def __getitem__(self, key):
        return self._values[key]
    
    @property
    def X(self):
        return self._values        

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D


def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)


def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', pooling='average', name='edgeconv'):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    with tf.name_scope('edgeconv'):

        # distance
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
        _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)

        fts = features
        knn_fts = knn(num_points, K, indices, fts)  # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(channels):
            x = tf.keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            if with_bn:
                x = tf.keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if activation:
                x = tf.keras.layers.Activation(activation, name='%s_act%d' % (name, idx))(x)

        if pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut
        sc = tf.keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
        if with_bn:
            sc = tf.keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
        sc = tf.squeeze(sc, axis=2)

        if activation:
            return tf.keras.layers.Activation(activation, name='%s_sc_act' % name)(sc + fts)  # (N, P, C')
        else:
            return sc + fts


def _outputs(points, features=None, mask=None, setting=None, name='particle_net'):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    with tf.name_scope(name):
        if features is None:
            features = points

        if mask is not None:
            mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

        fts = tf.squeeze(tf.keras.layers.BatchNormalization(name='%s_fts_bn' % name)(tf.expand_dims(features, axis=2)), axis=2)
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
            pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            fts = edge_conv(pts, fts, setting.num_points, K, channels, with_bn=True, activation='relu',
                            pooling=setting.conv_pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx))

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)

        if setting.fc_params is not None:
            x = pool
            for layer_idx, layer_param in enumerate(setting.fc_params):
                units, drop_rate = layer_param
                x = tf.keras.layers.Dense(units, activation='relu')(x)
                if drop_rate is not None and drop_rate > 0:
                    x = tf.keras.layers.Dropout(drop_rate)(x)
            out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            return out  # (N, num_classes)
        else:
            return pool
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

    def __init__(self):
        pass


    def metrics(self, task): return [tf.keras.metrics.BinaryAccuracy(name = "acc")]  ##list of metrics to be used
    def compile_args(self, task): return {'loss':'binary_crossentropy',#'categorical_crossentropy'
                    'optimizer':tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                    'metrics': self.metrics(task)
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
                'epochs': 30,
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
        X = np.reshape(X[0], (len(X[0]), (X[0].shape[1]*X[0].shape[2])))
        v = convert(X)
        dataset = Dataset(v, data_format='channel_last')
        #   write your preprocessing routine here
        return dataset.X

    def get_shapes(self, input_dataset):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routine, it should return the modified data
        in the desired shapes
        """
        input_shapes = {k:input_dataset[k].shape[1:] for k in input_dataset}
        return input_shapes


    
    def model(self, ds, shapes):
        r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
        <https://arxiv.org/abs/1902.08570>`_ paper.
        Parameters
        ----------
        input_shapes : dict
            The shapes of each input (`points`, `features`, `mask`).
        """

        setting = _DotDict()
        setting.num_class = 2 #num_classes
        # conv_params: list of tuple in the format (K, (C1, C2, C3))
        setting.conv_params = [
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
            ]
        # conv_pooling: 'average' or 'max'
        setting.conv_pooling = 'average'
        # fc_params: list of tuples in the format (C, drop_rate)
        setting.fc_params = [(256, 0.1)]
        setting.num_points = shapes['points'][0]

        points = tf.keras.Input(name='points', shape=shapes['points'])
        features = tf.keras.Input(name='features', shape=shapes['features']) if 'features' in shapes else None
        mask = tf.keras.Input(name='mask', shape=shapes['mask']) if 'mask' in shapes else None
        outputs = _outputs(points, features, mask, setting, name='top_model')

        return tf.keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet_{}'.format(ds.name))

    def model_lite(self, ds, shapes):
        r"""ParticleNet-Lite model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
        """
        setting = _DotDict()
        setting.num_class = 2
        # conv_params: list of tuple in the format (K, (C1, C2, C3))
        setting.conv_params = [
            (7, (32, 32, 32)),
            (7, (64, 64, 64)),
            ]
        # conv_pooling: 'average' or 'max'
        setting.conv_pooling = 'average'
        # fc_params: list of tuples in the format (C, drop_rate)
        setting.fc_params = [(128, 0.1)]
        setting.num_points = shapes['points'][0]

        points = tf.keras.Input(name='points', shape=shapes['points'])
        features = tf.keras.Input(name='features', shape=shapes['features']) if 'features' in shapes else None
        mask = tf.keras.Input(name='mask', shape=shapes['mask']) if 'mask' in shapes else None
        outputs = _outputs(points, features, mask, setting, name='ParticleNet')

        return tf.keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet_lite')
