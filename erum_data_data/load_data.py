import numpy as np
from .tt_graph_utils import load_top, convert
from . import belle_graph_utils as _belle


class LoadData:
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
        
        if graph:
            X_adj = _adjacency_matrix_img_8connected(X[0])
        
            X_feats = X[0].reshape(X[0].shape[0],X[0].shape[1]**2,1)
         
            X_graph = {}
            X_graph['features'] = X_feats
            X_graph['adj_matrix'] = X_adj
            return X_graph, y
        else:
            X = np.reshape(X[0], (len(X[0]), 20, 20, 1))
            return [X], y
    
    
    def eosl_data(split = "train", path = "./datasets", graph = False, force_download = False):
        from erum_data_data import EOSL
        """
        transforms the EOSL dataset into a graph
        """
        
        X,y = EOSL.load(split, path, force_download)
        
        if graph:
            X_adj = _adjacency_matrix_img_8connected(X[0])
            X_feats = X[0].reshape(X[0].shape[0],X[0].shape[1]**2,1)
            X_graph = {}
            X_graph['features'] = X_feats
            X_graph['adj_matrix'] = X_adj
            return X_graph, y
        else:
            return X, y #add preprocessing later
        
        
    def TopTagging_graph(split = "train", path = "./datasets", graph = False, force_download = False):
        from erum_data_data import TopTagging
        """
        transforms the TopTagging dataset into a graph
        """
        
        X,y = TopTagging.load(split, path, force_download)
        #Currently hardcoded:
        K = 7
        max_part = 200
        max_part_pad = 100

        #Lisa
        #print("reduced size for testing:")
        X[0] = X[0][:,0:max_part,:]
        #y = y[0:10000]
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
        
        X_graph = {}
        X_graph['features'] = X_feats
        X_graph['adj_matrix'] = X_adj
        
        if graph: 
            return X_graph, y
        else:
            return [X_graph['features']], y
        
    def belle_graph(
        split="train",
        path="./datasets",
        force_download=False,
        graph = False,
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
            X_graph['adj_matrix'] = _belle.adjacency_matrix_from_mothers_np(X[1].astype(np.int8))
            X_graph['features'] = np.concatenate(
                [
                    X[0][:, :, :-1],
                    _belle.np_onehot(_belle.remap_pdg(X[0][:, :, -1]), num_pdg)
                ],
                axis=-1
            )

            if graph:
                return X_graph, y
            else:
                return [X_graph['features']], y
        else:

            import tensorflow as tf

            def tensor_slices(slicing):
                return {
                    "mother_indices" : X[1][slicing],
                    "pdg" : _belle.remap_pdg(X[0][slicing][:, :, -1]),
                    "features" : X[0][slicing][:, :, :-1],
                }

            def transform(*ds):
                return (
                    {
                        "adj_matrix" : _belle.adjacency_matrix_from_mothers_tf(ds[0]["mother_indices"]),
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
            
            
    def airshower_data(split = "train", path = "./datasets", graph = False, force_download = False):
        from erum_data_data import Airshower
        """
        transforms the Airshower dataset into a graph
        """
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

        X,y = Airshower.load(split, path, force_download)
        signal, time, _ = X
        X_adj = _adjacency_matrix_img_8connected(signal)
        signal[signal == -1] = np.nan  # replace -1 with nan's to make nanmean work
        time[time == -1] = np.nan
        time, _ = norm_time(time)
        signal = norm_signal(signal)
       
        X_feats = np.concatenate((signal, time), axis=3).reshape(signal.shape[0],81,81)
        
        X_graph = {}
        X_graph['features'] = X_feats
        X_graph['adj_matrix'] = X_adj
        
        if graph: 
            return X_graph, y
        else:
            return [X_feats], y

              
              
 ## helper functions             
              
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

        

