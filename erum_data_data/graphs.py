import numpy as np
import tensorflow as tf
from .tt_graph_utils import load_top, convert
from . import belle_graph_utils as _belle


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
        
        X_adj = _adjacency_matrix_img(X[0])
        
        X_feats = X[0].reshape(X[0].shape[0],X[0].shape[1]**2,1)
        
        X_graph = {}
        X_graph['features'] = X_feats
        X_graph['adj_matrix'] = X_adj
        
        return X_graph, y
    
    
    def eosl_graph(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import EOSL
        """
        transforms the EOSL dataset into a graph
        """
        
        X,y = EOSL.load(split, path, force_download)
        
        X_adj = _adjacency_matrix_img(X[0])
        
        X_feats = X[0].reshape(X[0].shape[0],X[0].shape[1]**2,1)
        
        X_graph = {}
        X_graph['features'] = X_feats
        X_graph['adj_matrix'] = X_adj
        
        return X_graph, y
        
        
    def TopTagging_graph(split = "train", path = "./datasets", force_download = False):
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
        
        return X_graph, y
        
        
    def belle_graph(
        split="train",
        path="./datasets",
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
            X_graph['adj_matrix'] = _belle.adjacency_matrix_from_mothers_np(X[1].astype(np.int8))
            X_graph['features'] = np.concatenate(
                [
                    X[0][:, :, :-1],
                    _belle.np_onehot(_belle.remap_pdg(X[0][:, :, -1]), num_pdg)
                ],
                axis=-1
            )
            return X_graph, y
        else:

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


def _adjacency_matrix_img(inputs):
    """
    calculate adjacency matrix for images
    """
    
    shape = inputs.shape
    n = shape[1]
    N = n*n
    bs = shape[0]

    adj = np.zeros((N, N), dtype='int8')
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
