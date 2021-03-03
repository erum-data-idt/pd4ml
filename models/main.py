### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

## import of the models
# from airshower_xmax import Network

from eos_cnn import Network

# from fcn import Network  # import your model function

# from gcn_belle import Network
# from cnn_spinodal import Network
# from particle_net import Network

#########################################
#####  EXAMPLE IMPLEMENTATION OF FCN  ###

nn = Network()

datasets = nn.compatible_datasets

for ds in datasets:

    X_train, y_train = ds.load(split="train")
    X_test, y_test = ds.load(split="test")
    x_train = nn.preprocessing(X_train)
    x_test = nn.preprocessing(X_test)

    model = nn.model(ds, shapes=nn.get_shapes(x_train))
    model.compile(**nn.compile_args)
    print(model.summary())
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)

    # From here on, one should be able to use already
    # defined methods as shown in the following lines.
    # Let us know if you face any issues with that.

    # training history plots etc.
    nn.evaluation(
        model=model,
        history=history,
        dataset_name=ds.name,
        x_test=x_test,
        y_test=y_test,
    )