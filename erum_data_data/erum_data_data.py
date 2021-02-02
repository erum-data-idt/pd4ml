import os
import hashlib
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve
import numpy as np


# dictionary of urls from DESY cloud, filenames and md5 checksums
URL_DICT = {'test_filenames':[
                '1_top_tagging_2M.npz',
                '2_spinodal_29k.npz',
                '3_EOSL_or_EOSQ_180k.npz',
                '4_airshower_100k.npz',
                '5_LHCOlympics2020_features_1_1M.npz',
                '6_belle_selective_400k.npz',
                ],
             'test_urls': [
                 'https://desycloud.desy.de/index.php/s/aZqyNSg4B7nn8qQ/download',
                 'https://desycloud.desy.de/index.php/s/zZCCSfwwEkT5Pgk/download',
                 'https://desycloud.desy.de/index.php/s/DT7sWm6rNR5zss9/download',
                 'https://desycloud.desy.de/index.php/s/Le5MZCGar7ZEm9f/download',
                 'https://desycloud.desy.de/index.php/s/4DaZCFgmZ83o9FT/download',
                 'https://desycloud.desy.de/index.php/s/RKB4z3mMcPPY982/download',
                 ],
            'test_md5': [
                '708a8369d75ceff2229bd8c46b47afea',
                'c22326822d9bac7c074467ccccc6fe4f',
                'be5563e787f856bd6b2f9fe171277bc3',
                'b3afa0efe9f4c6a5b936364cd52b1904',
                'd9a29ff15679c7034b7acf58e82afa0f',
                '85b10c9df3903455ab247a0ab4b51e5f',
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
    #else:
        #print('file was already downloaded, is loading')
        
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
Top tagging dataset.

Description:
14 TeV, hadronic tops for signal, QCD djets background, Delphes ATLAS detector card with Pythia. No MPI/pile-up included
Particle-flow entries (produced by Delphes E-flow) have been clustered into anti-kT 0.8 jets in the pT range [550,650].
All top jets are matched to a parton-level top within ∆R = 0.8, and to all top decay partons within 0.8. Also,|eta|_jet < 2 has been required.

Ref: 
Deep-learned Top Tagging with a Lorentz Layer by A Butter, G Kasieczka, T and M Russell (arXiv: 1707.08966)

Dataset shape:
~2M events have been stored divided between training (~1.6M) and test (~400k)) and the shape of the dataset is (# of events, 200, 4).
The feature represent the leading 200 jet constituent four-momenta, with zero-padding for jets that have less than 200.
Constituents are sorted by pT, with the highest pT one first.

The second dataset that is included is just a flag "ttv" to identify what the event was before the reshaping operated by us. Here a legenda:
	0 = training event;
	1 = test event;
	2 = validation event;

Note that in the current splitting of the dataset, training and validation events have been merged together as a unique training dataset. So for most intents and purposes one should just train the model on the first dataset and omit the second 'ttv' dataset altogether.

The set label are 0 for QCD and 1 for top.
''')
    
    spinodal_description = ('''
Spinodal dataset.
Description:
The classification goal for this dataset is to identify the nature of the QCD phase transitic collisions at the CBM experiment and, in particular, whether signals for b-associated with the phase transition can be ound in the final momentum spectra of certain collisions.

Ref:
J. Steinheimer, L. Pang, K. Shou, V. Koch, J. Randrup and H.Stoecker, JHEP 19 doi:10.1007/JHEP12(2019)122 [arXiv:1906.06562 [nucl-th]]

Dataset shape:
The dataset is composed of 29'000 2D histograms of shape 20x20 describing pion momenta, divided in training (70%) and test (30%). So, the shape of the dataset is (# of events, 20, 20).

The set label is 1 for a Spinodal event and 0 for a Maxwell event.
''')
    
    EOSL_description = ('''
EOSL or EOSQ dataset.

Description:
The task here is to classify the QCD transition nature (two different equation of state: cross-over EOSL or 1st order EOSQ) happened in heavy-ion collisions from the final state pion spectra. The pion spectra in transverse momentum and azimuthal angle are simulated here with Hybrid (Viscous Hydrodynamics plus hadronic cascade UrQMD) modeling for heavy-ion collisions by varing different physical  arameters (collision energy, entrality, initial time, initial ondition models with fluctuations, shear viscousity, freeze-out temperature, switch time from hydro to hadronic cascade)

Ref:
An equation-of-state-meter of quantum chromodynamics transition from deep learning, Long-Gang Pang, Kai Zhou, Nan Su, Hannah Petersen, Horst Stoecker and Xin-Nian Wang, Nature Commun.9 (2018) no.1,210
Identifying the nature of the QCD transition in relativistic collision of heavy nuclei with deep learning, Yi-Lun Du, Kai Zhou, Jan Steinheimer, Long-Gang Pang, Anton Motornenko, Hong-Shi Zong, Xin-Nian Wang and Horst Stoecker, arXiv:1910.11530

Dataset shape:
The dataset is composed of ~180'000 2D histogram of shape 24x24 of the pion spectra and it is divided in 70% training and 30% test. So, the shape of the dataset is (# of events, 24, 24).
Label is 1 for EOSQ and 0 for EOSL.
''')
    
    airshower_description = ('''
    
Airshower Proton vs Iron Classification

Based on https://doi.org/10.1016/j.astropartphys.2017.10.006

Produced by jonas.glombitza@rwth-aachen.de
    
----------------------------------    
Dataset shape:

Three sets of input data:
- first set of input data (shape: [70k, 9, 9, 80]):
    - 70k events (airshowers)
    - 9x9 = 81 ground detector stations
    - 80 measured signal bins (forming one signal trace per station)
    -padding: (-1) padding for instances that the detector / or timestep did not detect a particle
    
- second set of input data (shape: [70k, 9, 9, 1]:
    - 70k events (airshowers)
    - 9x9 = 81 ground detector stations   
    - 1 starting time of the signal trace (arrival time of first particles at each station)

- third set of input data
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

Classification between 1M QCD dijet events and 100k W'->XY events. Hence this dataset is unbalanced with a label 0 to 1 ratio of 10:1. 

More information from the authors: https://zenodo.org/record/4287694#.X9o1_C1Q1pR
The dataset 'events_anomalydetection_v2.features.h5' was used and split into two a training set (70%) and a test set (30%).

----------------------------------    
Dataset shape:
- one numpy array of the shape (no. of events, 14)
    - 14 features, namely:

        The features (calculated using fastjet plugins) are the 3-momenta, invariant masses, 
        and n-jettiness variables tau1, tau2 and tau3 for the highest pT jet (j1) 
        and the second highest pT jet (j2):
    
        'pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 
        'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2'

---------------------------------- 
Label: 
0: QCD dijet event, 1: W'-> XY event (ratio: 10:1)
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
