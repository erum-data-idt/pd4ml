### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

## import of the models
#from graph_net import Network
from fcn import Network
#########################################

nn = Network()
build_graph = nn.build_graph
datasets = nn.compatible_datasets

for ds in datasets:

    x_train, y_train = ds.load_data('train', path = './datasets', graph = build_graph)
    x_test, y_test = ds.load_data('test', path = './datasets', graph = build_graph)

    x_train = nn.preprocessing(x_train)
    x_test = nn.preprocessing(x_test)

    model = nn.model(ds, shapes=nn.get_shapes(x_train))
    model.compile(**nn.compile_args(ds.task))
    print(model.summary())
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)
    
    # after training saving best model
    filepath = './trained_models/{}/{}_model'.format(ds.name, nn.model_tag(ds.name, nn.model_name)) 
    model.save(filepath)
    
    # evaluation after training
    nn.evaluation(
        model=model,
        history=history,
        dataset=ds,
        x_test=x_test,
        y_test=y_test,
        model_name = nn.model_name,
        path = filepath,
    )
