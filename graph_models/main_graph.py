### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

## import of the models
from simple_graph_net import Network

#########################################

nn = Network()

datasets = nn.compatible_datasets

for ds in datasets:

    x_train, y_train = ds.load_graph('train', path = './datasets')
    x_test, y_test = ds.load_graph('test', path = './datasets')
    
    model = nn.model(ds, shapes=nn.get_shapes(x_train))
    model.compile(**nn.compile_args(ds.task))
    print(model.summary())
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)
    
    # after trining saving best model
    filepath = './trained_models/{}/{}_checkpoint'.format(ds.name, nn.model_tag(ds.name, nn.model_name)) 
    model.save(filepath)
    
    # evaluation after training
    nn.evaluation(
        model=model,
        history=history,
        dataset=ds,
        x_test=x_test,
        y_test=y_test,
        model_name = nn.model_name,
    )