# erum_data_data

install this package as a python module with pip via:

```
pip install git+https://github.com/erikbuh/erum_data_data.git
```

The only function so far is the "load" function to load the training and testing datasets. The datasets features "X" are returned as a list of numpy arrays. The labels are returend directly as a numpy array. 

```
import erum_data_data as edd
X_train, y_train, X_test, y_test = edd.load('LHCO', cache_dir = './', cache_subdir = 'datasets')
```
Here a subfolder ./datasets is created. The datasets take up a total disk space of about 2 GB. For loading the datasets a free RAM of at least 7 GB is necessary.

Included datasets at the moment with the tags: 
'airshower', 'LHCO', 'belle'




---

The original datasets can be found here:
   1. Top Tagging at the LHC [link](https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit?usp=sharing), Publication: 1902.09914
   2. Spinodal or not? [link](https://vfs.fias.science/d/fa35025bf2/?p=/Example-Datasets-classification/Spinodal-or-not), Publication: 1906:06562
   3. EOSL or EOSQ [link](https://vfs.fias.science/d/fa35025bf2/?p=/Example-Datasets-classification/EOSL-or-EOSQ), Publication: 1910.11530
   4. Cosmic Airshower [link](https://desycloud.desy.de/index.php/s/QZ5kJGdKcPryaaf)
   5. LHC Olympics 2020 (Unsupervised anomaly detection) [link](https://lhco2020.github.io/homepage/)
   6. SmartBKG dataset (Belle II - generated events passing downstream selection) [link](https://github.com/kahn-jms/belle-selective-mc-dataset)
