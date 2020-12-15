import os
import hashlib
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve
import numpy as np


# dictionary of urls from DESY cloud, filenames and md5 checksums
URL_DICT = {'test_filenames':[
                '4_airshower_100k.npz',
                '5_LHCOlympics2020_350k.npz',
                '6_belle-selective_400k.npz',
                'test.npy',
                'test_half1.npy',
                'test_half2.npy',
                ],
             'test_urls': [
                'https://desycloud.desy.de/index.php/s/Qcn38RExyDstEAP/download',
                 'https://desycloud.desy.de/index.php/s/izD3CN8nw68nWPp/download',
                 'https://desycloud.desy.de/index.php/s/Yd885LGfdMqoxbq/download',
                'https://desycloud.desy.de/index.php/s/aCcYSZmzHi3RXKm/download',
                'https://desycloud.desy.de/index.php/s/LJkzgbg6C4JRwTY/download',
                'https://desycloud.desy.de/index.php/s/mctQMTo7gfNmAa2/download',
                 ],
            'test_md5': [
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



def load(dataname, cache_dir = './', cache_subdir = 'datasets'):
    ''' 
    loads a datafile from a list of options 
    Returns a list of X feature numpy arrays for test and training set
    as well as numpy arrays for test and training label
    '''

    if dataname == 'airshower':
        i = 0
    elif dataname == 'LHCO':
        i = 1
    elif dataname == 'belle':
        i = 2
    else:
        print('WARNING: NOT A VALID DATANAME')
        return
    
    filename = URL_DICT['test_filenames'][i]
    url = URL_DICT['test_urls'][i]
    md5 = URL_DICT['test_md5'][i]
    
    fpath = _get_filepath(filename, url, md5, cache_dir , cache_subdir)
    
    np_zip = np.load(fpath)
    
    X_train, X_test = [], []
    for i in range(int(len(np_zip)/2 - 1)):
        X_train.append(np_zip['X_train_{}'.format(i)])
        X_test.append(np_zip['X_test_{}'.format(i)])
    y_train = np_zip['y_train']
    y_test = np_zip['y_test']
    
    return X_train, y_train, X_test, y_test