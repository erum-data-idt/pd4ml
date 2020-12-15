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
