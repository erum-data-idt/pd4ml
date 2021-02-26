import numpy as np


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
