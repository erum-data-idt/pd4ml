# erum_data_data

install this package as a python module with pip via:

```
pip install git+https://github.com/erikbuh/erum_data_data.git
```

The essential function so far is the "load" function to load the training and testing datasets. The datasets features "X" are returned as a list of numpy arrays. The labels are returend directly as a numpy array. 

```
import erum_data_data as edd

# loading training data into RAM (downloads dataset first time)
X_train, y_train  = edd.load('top', dataset='train', cache_dir = './', cache_subdir = 'datasets')

# loading test data into RAM (downloads dataset first time)
X_test, y_test = edd.load('top', dataset='test', cache_dir = './', cache_subdir = 'datasets')
```

Here a subfolder ./datasets is created. The datasets take up a total disk space of about 2.4 GB. For loading the training datasets a free RAM of at at least 5 GB is necessary (depending on the dataset).

Included datasets at the moment with the tags:
```
1: 'top', 2: 'spinodal', 3: 'EOSL', 4: 'airshower', 5: 'LHCO', 6: 'belle'
```

An description of the datasets can be printed via the function:
```
edd.print_description('top')
```

Some example plots can be found in the notebooks in the example folder.

---

### Simple Fully-Connected Network Implementation

A simple model implementation can be found in the folder 'simple_model'. To run the notebook one needs to additionally install at least [tensorflow](https://www.tensorflow.org/) version >= 2.0 and [scikit](https://scikit-learn.org/stable/) >= 0.22. 



---

The original datasets can be found here:
   1. Top Tagging at the LHC [link](https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit?usp=sharing), Publication: 1902.09914
   2. Spinodal or not? [link](https://vfs.fias.science/d/fa35025bf2/?p=/Example-Datasets-classification/Spinodal-or-not), Publication: 1906:06562
   3. EOSL or EOSQ [link](https://vfs.fias.science/d/fa35025bf2/?p=/Example-Datasets-classification/EOSL-or-EOSQ), Publication: 1910.11530
   4. Cosmic Airshower [link](https://desycloud.desy.de/index.php/s/QZ5kJGdKcPryaaf)
   5. LHC Olympics 2020 (Unsupervised anomaly detection) [link](https://lhco2020.github.io/homepage/)
   6. SmartBKG dataset (Belle II - generated events passing downstream selection) [link](https://github.com/kahn-jms/belle-selective-mc-dataset)
