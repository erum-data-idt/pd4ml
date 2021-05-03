import numpy as np
from .tt_graph_utils import load_top, convert
from . import belle_graph_utils as _belle


class LoadFlat:
    """
    Performing the preprocessing routines for each dataset
    
    """
    
    def spinodal_flat(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import Spinodal
        X,y = Spinodal.load(split, path, force_download)
        return X, y
    
    
    def eosl_flat(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import EOSL
        
        X,y = EOSL.load(split, path, force_download)
        
        return X, y
        
        
    def TopTagging_flat(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import TopTagging
        
        X,y = TopTagging.load(split, path, force_download)
        
        return conver(X), y
        



    def belle_flat(
        split="train",
        path="./datasets",
        force_download=False,
 #       num_pdg=182,
 #       max_entries=None,
 #       as_tf_data=False,
 #       batch_size=1024,
 #       validation_split=0.2,
    ):
        from erum_data_data import Belle

        X, y = Belle.load(split, path, force_download)
        #add preprocessing to flatten Belle dataset

        return X, y
            
            
    def airshower_flat(split = "train", path = "./datasets", force_download = False):
        from erum_data_data import Airshower
        """
        transforms the Airshower dataset into flat 
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
        signal[signal == -1] = np.nan  # replace -1 with nan's to make nanmean work
        time[time == -1] = np.nan
        time, _ = norm_time(time)
        signal = norm_signal(signal)
        #add proper reshaping here
        return [signal, time], y
        
        
