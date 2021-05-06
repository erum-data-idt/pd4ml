import numpy as np
from functools import cached_property
from sklearn.preprocessing import StandardScaler
from .preprocessing_utils import load_top, convert
from .preprocessing_utils import np_onehot, remap_pdg, adjacency_matrix_from_mothers_np, adjacency_matrix_from_mothers_tf 
from .preprocessing_utils import _adjacency_matrix_img_8connected
#from . import belle_graph_utils as _belle
#from .tt_graph_utils import load_top, convert

class LoadPreprocessedData:
    """
    Transformation from the dataset to a graph 
    
    returns graph features, the adjacency matrix and a mask (per default all True)
    """

 
    def spinodal_data(split = "train", path = "./datasets", graph = False, force_download = False):
        from erum_data_data import Spinodal
        """
        transforms the Spinodal dataset into a graph
        """
        
        X,y = Spinodal.load(split, path, force_download)
        X_graph = {}
        if not graph:
            X_graph['features'] = X
        elif graph: 
            X_adj = _adjacency_matrix_img_8connected(X[0])
            X_feats = X[0].reshape(X[0].shape[0],X[0].shape[1]**2,1)
            X_graph['features'] = X_feats
            X_graph['adj_matrix'] = X_adj
        
        return X_graph, y
    
    def eosl_data(split = "train", path = "./datasets", graph = False, force_download = False):
        from erum_data_data import EOSL
        """
        transforms the EOSL dataset into a graph
        """
        
        X,y = EOSL.load(split, path, force_download)
        X[0] = X[0].reshape(X[0].shape[0], X[0].shape[1]* X[0].shape[2])
        if split == 'train':
            scaler = StandardScaler().fit(X[0])
            scaler.transform(X[0])
        elif split == 'test':
            X_train, _ = EOSL.load('train', path, False)
            X_train = X_train[0].reshape(X_train[0].shape[0], X_train[0].shape[1]* X_train[0].shape[2])
            scaler = StandardScaler().fit(X_train)
            scaler.transform(X[0])      

        X_graph = {}
        if not graph:
            X_graph['features'] = X
        elif graph:
            X_adj = _adjacency_matrix_img_8connected(X[0])
            X_feats = X[0].reshape(X[0].shape[0],X[0].shape[1]**2,1)
            X_graph['features'] = X_feats
            X_graph['adj_matrix'] = X_adj
        
        return X_graph, y
        
        
    def top_tagging_data(split = "train", path = "./datasets", graph = False, force_download = False):
        from erum_data_data import TopTagging
        """
        transforms the TopTagging dataset into a graph
        """
        
        X,y = TopTagging.load(split, path, force_download)

        #Currently hardcoded:
        K = 7
        max_part = 200
        max_part_pad = 100
        X_graph = {}
        data_format='channel_last'
        stack_axis = 1 if data_format=='channel_first' else -1


        X[0] = X[0][:,0:max_part,:]
        X = np.reshape(X[0], (len(X[0]), (X[0].shape[1]*X[0].shape[2])))
        v = convert(X,max_part)
        feature_dict = {}
        
        if not graph:
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel']
            X_graph['features'] = [load_top(v, feature_dict, max_part_pad, stack_axis, None, graph)]
        if graph:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel']
            feature_dict['mask'] = ['part_pt_log']
            X_graph['features'], X_graph['adj_matrix'] = load_top(v, feature_dict, max_part_pad, stack_axis, K, graph)
        
        return X_graph, y
        
        
    def belle_data(
        split="train",
        path="./datasets",
        graph=False,
        force_download=False,
        num_pdg=182,
        max_entries=None,
        as_tf_data=False,
        batch_size=1024,
        validation_split=0.2,
    ):
        from erum_data_data import Belle

        X, y = Belle.load(split, path, force_download)

        if max_entries is not None:
            # this is mainly for testing
            X = [x_i[:max_entries] for x_i in X]
            y = y[:max_entries]

        if not as_tf_data:
            X_graph = {}
            X_graph['features'] = [np.concatenate(
                                                   [X[0][:, :, :-1],
                                                    np_onehot(remap_pdg(X[0][:, :, -1]), num_pdg)
                                                   ],
                                                   axis=-1
                                                   ),
                                   ]
            if not graph:
                X_graph['features'].append(X[1])
            
            elif graph:
                X_graph['adj_matrix'] = adjacency_matrix_from_mothers_np(X[1].astype(np.int8))

            return X_graph, y
        else:

            import tensorflow as tf

            def tensor_slices(slicing):
                return {
                    "mother_indices" : X[1][slicing],
                    "pdg" : remap_pdg(X[0][slicing][:, :, -1]),
                    "features" : X[0][slicing][:, :, :-1],
                }

            def transform(*ds):
                return (
                    {
                        "adj_matrix" : adjacency_matrix_from_mothers_tf(ds[0]["mother_indices"]),
                        "features" : tf.concat(
                            [ds[0]["features"], tf.one_hot(ds[0]["pdg"], num_pdg)], axis=-1
                        )
                    },
                    ds[1]
                )

            def from_tensor_slices(slicing):
                return (
                    tf.data.Dataset.from_tensor_slices(
                        (tensor_slices(slicing), y[slicing])
                    )
                    .batch(batch_size)
                    .map(transform)
                )

            if validation_split is not None:
                split_at = int(len(y) * (1 - validation_split))
                ds_train = from_tensor_slices(slice(None, split_at))
                ds_val = from_tensor_slices(slice(split_at, None))
                return ds_train, ds_val
            else:
                return from_tensor_slices(slice(None))
            
            
    def airshower_data(split = "train", path = "./datasets", graph= False, force_download = False):
        from erum_data_data import Airshower
        """
        transforms the Airshower dataset into a graph
        """
        
        X,y = Airshower.load(split, path, force_download)
        X_graph = {}

        def norm_time(time, std=None):
            u = np.isnan(time)
            time -= np.nanmean(time, axis=(1, 2, 3), keepdims=True)

            if std is None:
                std = np.nanstd(time)

            time /= std
            time[u] = 0
            return time, std

        def norm_signal(signal):
            signal[np.less(signal, 0)] = 0
            signal = np.log10(signal + 1)
            signal[np.isnan(signal)] = 0
            return signal


        signal, time, _ = X
        signal[signal == -1] = np.nan  # replace -1 with nan's to make nanmean work
        time[time == -1] = np.nan
        time, _ = norm_time(time)
        signal = norm_signal(signal)
        if not graph: 
            X_graph['features'] = [signal, time]

        elif graph:
            X_graph['adj_matrix'] = _adjacency_matrix_img_8connected(X[0])
                    
            X_graph['features'] = [] #perform here the reshaping of the data 
        
        return X_graph, y

