##	William Korcari: william.korcari@desy.de

## import of the models
#from airshower_xmax import Network
#from gcn_belle import Network
#from cnn_spinodal import Network
#from particle_net import Network
from eos_cnn import Network

#########################################

nn = Network()

datasets = nn.compatible_datasets

for ds in datasets:

    X_train, y_train = ds.load(split="train")
    X_test, y_test = ds.load(split="test")
    nn.init_preprocessing(X_train)
    x_train = nn.preprocessing(X_train)
    x_test = nn.preprocessing(X_test)

    model = nn.model(ds, shapes=nn.get_shapes(x_train))
    model.compile(**nn.compile_args(ds.task))
    print(model.summary())
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)
    filepath = './trained_models/{}/{}_model'.format(ds.name, nn.model_tag(ds.name, nn.model_name)) 
    model.save(filepath)

    nn.evaluation(
        model=model,
        history=history,
        dataset=ds,
        x_test=x_test,
        y_test=y_test,
        model_name = nn.model_name,
        path = filepath,
    )
