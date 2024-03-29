# Physics Data for Machine Learning (pd4ml)

This repository contains datasets and model for machine learning from the publication "Shared Data and Algorithms for Deep Learning in Fundamental Physics" ([arXiv:2107.00656](https://arxiv.org/abs/2107.00656))

You can install this package as a python module with pip via:

```
pip install git+https://github.com/erum-data-idt/pd4ml

# or just git clone & 'pip install .' in this folder
```

The essential function is the `load` function to load the training and testing datasets. The datasets features "X" are returned as a list of numpy arrays. The labels are returend directly as a numpy array. 

```python
from pd4ml import Spinodal   # or any other dataset (see below) 

# loading training data into RAM (downloads dataset first time)
X_train, y_train  = Spinodal.load('train', path='./datasets')

# loading test data into RAM (downloads dataset first time)
X_test, y_test = Spinodal.load('test', path = './datasets')
```

Here a subfolder ./datasets is created. The datasets take up a total disk space of about 2.4 GB. For loading the training datasets a free RAM of at at least 5 GB is necessary (depending on the dataset).

Included datasets at the moment with the tags:
```
1: TopTagging, 2: Spinodal, 3: EOSL, 4: Airshower, 5: Belle
```

An description of the datasets can be printed via the function:
```python
Spinodal.print_description()
```

Show all available datasets:
```python
import pd4ml

for dataset in pd4ml.Dataset.datasets_register:
    print(dataset.name)
```

An additionally `load_data` function performs some basic preprocessing steps as well as allows the return of an adjecancy matrix:
```python
from pd4ml import Spinodal   # or any other dataset
x_train, y_train = Spinodal.load_data('train', path = './datasets', graph = True)
```

`x_train` is dictionary with the contents `features` and `adj_matrix`. If no adjecancy matrix is required, one may set `graph = False`. 

Some example plots can be found in the notebooks in the example folder.

---

### Creating a model:

In the folder `models` multiple model implementations can be found. Each can be imported in the `main.py` script and run on the specified datasets. If you'd like to contribute a model, feel free to implement it using the `template.py`. 

---
