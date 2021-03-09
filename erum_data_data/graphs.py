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
        
        X_adj = _adjacency_matrix_img_8connected(X[0])
        
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
        
        X_adj = _adjacency_matrix_img_8connected(X[0])
        
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
