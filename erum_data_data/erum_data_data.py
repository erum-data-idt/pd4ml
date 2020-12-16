import os
import hashlib
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve
import numpy as np


# dictionary of urls from DESY cloud, filenames and md5 checksums
URL_DICT = {'test_filenames':[
                '1_top_tagging_1_2M.npz',
                '2_spinodal_200k.npz',
                '3_EOSL_or_EOSQ_180k.npz',
                '4_airshower_100k.npz',
                '5_LHCOlympics2020_350k.npz',
                '6_belle-selective_400k.npz',
                'test.npy',
                'test_half1.npy',
                'test_half2.npy',
                ],
             'test_urls': [
                 'https://desycloud.desy.de/index.php/s/ZcCtQ6G8df4F5px/download',
                 'https://desycloud.desy.de/index.php/s/gfd79exSBgYqspA/download',
                 'https://desycloud.desy.de/index.php/s/rLrLPLZDTtBLd5n/download',
                'https://desycloud.desy.de/index.php/s/Qcn38RExyDstEAP/download',
                 'https://desycloud.desy.de/index.php/s/izD3CN8nw68nWPp/download',
                 'https://desycloud.desy.de/index.php/s/Yd885LGfdMqoxbq/download',
                'https://desycloud.desy.de/index.php/s/aCcYSZmzHi3RXKm/download',
                'https://desycloud.desy.de/index.php/s/LJkzgbg6C4JRwTY/download',
                'https://desycloud.desy.de/index.php/s/mctQMTo7gfNmAa2/download',
                 ],
            'test_md5': [
                '7cee4581389e269d9214d51bcaa58fce',
                '7808e5dd8d0fa7e6a7f26991fd69a23f',
                '66a12b7ac5f2e2b5db5bffb216654ef5',
                'fcc0784f2dd781d7b1e02754fbffb360',
                'e996ca33ce3fa89877b7cb6bfad449ea',
                '5315e853616d12c39b590ff6d3fcc95e',
                '3ecb1f1e4f2c624d4e4361877c7466e2',
                'f96b949fc84dbce69f4e47f2eb66690a',
                '55fe2c80f22b92247eb85805d6f17f1e',
                ],
             }



def _check_md5(fpath):
    ''' returns md5 checksum for file in fpath ''' 
    hash_md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _get_filepath(filename, url, md5, cache_dir = './', cache_subdir = 'datasets'):
    ''' 
    returns filepath for specific data file;
    if file is not already downloaded, it downloads it from the DESY cloud; 
    performs md5 checksum check to verify file integrities 
    '''
    
    #filename = url.split('/')[-1].split('?')[0]
    
    # handle '~' in path
    datadir_base = os.path.expanduser(cache_dir)
    
    # ensure that directory exists, if not create
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    
    fpath = os.path.join(datadir, filename)
    
    # determine if file needs to be downloaded
    download = False
    if os.path.exists(fpath):
        if md5 is not None and not _check_md5(fpath) == md5:
            print('Local file hash does not match so we will redownload...')
            download = True
    else:
        download = True
        
    if download:
        print('Downloading {} from {} to {}'.format(filename, url, datadir))

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(url, fpath)
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(url, e.code, e.msg))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if md5 is not None:
            assert _check_md5(fpath) == md5, 'Hash of downloaded file incorrect.'
        print('Done Downloading')
    else:
        print('file was already downloaded')
        
    return fpath



def load(dataname, dataset = 'train', cache_dir = './', cache_subdir = 'datasets'):
    ''' 
    loads a datafile from a list of options 
    Returns a list of X feature numpy arrays for test and training set
    as well as numpy arrays for test and training label
    
    Additional descriptions of the datasets can be printed via: 
    
    erum_data_data.print_description() function
    
    Parameters
    ----------
    dataname: Name of dataset from a list of options:  
        1: 'top', 2: 'spinodal', 3: 'EOSL', 4: 'airshower', 5: 'LHCO', 6: 'belle'
    datset: chosse the training or testing set:
        dataset = 'train' or 'test'
    cache_dir: directory where the datasets are saved
        cache_dir = './'
    cache_subdir: name of subdirectory
        cache_subdir = 'datasets'
        
    Returns
    -------
    X, y
    X: a list of numpy arrays with X input features - see print_decription() for more details
    y: a numpy array with labels [0,1]
    '''

    if dataname == 'top':
        i = 0
    elif dataname == 'spinodal':
        i = 1
    elif dataname == 'EOSL':
        i = 2
    elif dataname == 'airshower':
        i = 3
    elif dataname == 'LHCO':
        i = 4
    elif dataname == 'belle':
        i = 5
    else:
        print('WARNING: NOT A VALID DATANAME')
        return
    
    filename = URL_DICT['test_filenames'][i]
    url = URL_DICT['test_urls'][i]
    md5 = URL_DICT['test_md5'][i]
    
    fpath = _get_filepath(filename, url, md5, cache_dir , cache_subdir)
    
    np_zip = np.load(fpath)
    
    X = []
    for i in range(int(len(np_zip)/2 - 1)):
        X.append(np_zip['X_{}_{}'.format(dataset, i)])
    y = np_zip['y_{}'.format(dataset)]
    
    return X, y


def print_description(dataname=None):
    '''
    This is a helper function that prints a description of the different datasets.

    Parameters for the datasets are, i.e. dataname='top'.

    All six datasets are tagged with:
    1: 'top'
    2: 'spinodal'
    3: 'EOSL'
    4: 'airshower'
    5: 'LHCO'
    6: 'belle'
    '''

    none_description = ('''
This is a helper function that prints a description of the different datasets.

Parameters for the datasets are, i.e. dataname='top'.

All six datasets are tagged with:
1: 'top'
2: 'spinodal'
3: 'EOSL'
4: 'airshower'
5: 'LHCO'
6: 'belle'
''')
    
    top_description = ('''
lets write
''')
    
    spinodal_description = ('''
a bit of text 
''')
    
    EOSL_description = ('''
another text
''')
    
    airshower_description = ('''
    
Airshower Proton vs Iron Classification

Based on https://doi.org/10.1016/j.astropartphys.2017.10.006

Produced by jonas.glombitza@rwth-aachen.de
    
----------------------------------    
Dataset shape:

Three sets of input data:
- first set of input data:
    - 70k events (airshowers)
    - 81 ground detector stations
    - 81 features
        - 1  time   (arrival time of first particles at each station)
        - 80 measured signal traces
    -padding: (-1) padding for instances that the detector / or timestep did not detect a particle
    
- second set of input data:
    - 70k events (airshowers)
    - 11 features per airshower:
        'logE',  --> (energy of cosmic ray)
        'Xmax',  --> (depth of shower maximum)  
        'showermax_x', 'showermax_y', 'showermax_z',     --> (point of showermaximum in x,y,z)
        'showeraxis_x', 'showeraxis_y', 'showeraxis_z',  --> (shower axis (arrival direction) in x,y,z)
        'showercore_x', 'showercore_y', 'showercore_z',  --> (shower core (intersection shower axis with detector plane) in x,y,z)
    
- thrid set of input data
    - detector geometry - for reference if needed
    - 81 ground detector stations
    - 3 features: x,y,z location of each station
  
---------------------------------- 
Label: 
Proton (1) vs. Iron (0) as shower origin particle.
Proton to Iron ratio in test & training set is 1:1.
''')
    
    LHCO_description = ('''
    
R&D Dataset for LHC Olympics 2020 Anomaly Detection Challenge

Classification between QCD dijet events and W'->XY events.

More information from the authors: https://zenodo.org/record/4287694#.X9o1_C1Q1pR

----------------------------------    
Dataset shape:


---------------------------------- 
Label: 

''')
    
    belle_description = ('''

SmartBKG dataset (Belle II - generated events passing downstream selection)

The goal of this classification problem is to identify generated events that pass a selection already before the expensive detector simulation and reconstruction.

Original dataset and additional information: https://github.com/kahn-jms/belle-selective-mc-dataset

----------------------------------    
Dataset shape:

Two sets of input data: 
- first set with shape:
    - 280000 belle collider events
    - 100 particles (zero padded)
    - 9 features ('prodTime', 'energy', 'x', 'y', 'z', 'px', 'py', 'pz', 'PID')
        - note: PID corresponding to a unique PDG particle ID, but mapped to a continous space
        
- second set with shape:
    - 280000 belle collider events
    - 100 indices of mother particles (adjacency matrix for creating a graph of the event)
        - note: these are -1 padded
---------------------------------- 
Label: 
event passes (1) or fails (0) a selection that was applied after detector simulation and reconstruction
''')
   
    if dataname == 'top':
        return print(top_description)
    elif dataname == 'spinodal':
        return print(spinodal_description)
    elif dataname == 'EOSL':
        return print(EOSL_description)
    elif dataname == 'airshower':
        return print(airshower_description)
    elif dataname == 'LHCO':
        return print(LHCO_description)
    elif dataname == 'belle':
        return print(belle_description)
    else:
        return print(none_description)