import numpy as np
from .tt_graph_utils import load_top, convert


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
        max_entries=None
    ):
        from erum_data_data import Belle

        X, y = Belle.load(split, path, force_download)

        if max_entries is not None:
            # this is mainly for testing
            X = [x_i[:max_entries] for x_i in X]
            y = y[:max_entries]

        X_graph = {}
        X_graph['adjacency'] = _Belle.adjacency_matrix_from_mothers_np(X[1].astype(np.int8))
        X_graph['features'] = np.concatenate(
            [
                X[0][:, :, :-1],
                _Belle.np_onehot(_Belle.remap_pdg(X[0][:, :, -1]), num_pdg)
            ],
            axis=-1
        )

        return X_graph, y

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


class _Belle:

    """
    Some helper functions for the graph version of the Belle dataset
    """

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


    def get_sorted_pdg():
        from erum_data_data import Belle
        data = Belle.load()
        mapped_pdg, counts = np.unique(data[0][0][:, :, -1].ravel(), return_counts=True)
        mapped = mapped_pdg[np.argsort(counts)][::-1].astype(int)
        return mapped[mapped != 0]


    # generated with `_Belle.get_sorted_pdg()`
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

        remap_dict = {v : k for k, v in enumerate(_Belle.sorted_pdg)}

        @np.vectorize
        def remap(pdg_id):
            return remap_dict.get(pdg_id, len(_Belle.sorted_pdg) + 1)

        return remap(mapped_pdg)


    def np_onehot(indices, depth):
        """
        Behaviour like tf.one_hot
        """
        indices = np.array(indices)
        indices[indices >= depth] = depth
        return np.eye(depth + 1, dtype=np.int8)[indices][..., :-1]
