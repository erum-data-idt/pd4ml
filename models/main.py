### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

## import of the models
from fcn import Network  
#from airshower_xmax import Network
#from gcn_belle import Network
#from cnn_spinodal import Network
#from particle_net import Network
#from eos_cnn import Network
#from benchmark.benchmark import Benchmark

#########################################
#####  EXAMPLE IMPLEMENTATION OF FCN  ###

nn = Network()

datasets = nn.compatible_datasets

for ds in datasets:

    X_train, y_train = ds.load_flat(split="train")
    X_test, y_test = ds.load_flat(split="test")
    
    nn.init_preprocessing(X_train)
    x_train = nn.preprocessing(X_train)
    x_test = nn.preprocessing(X_test)

### hyperparameters fine-tuning
    #nn.fit_args["batch_size"] = 50
    #print(nn.fit_args)
###

    model = nn.model(ds, shapes=nn.get_shapes(x_train))
    model.compile(**nn.compile_args(ds.task))
    print(model.summary())
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)
    filepath = './trained_models/{}/{}_model'.format(ds.name, nn.model_tag(ds.name, nn.model_name)) 
    model.save(filepath)
    

    #benchmark = Benchmark(dataset=ds.name, network=model.name)
    #benchmark.snapshot(history=history.history)
    # print summary table or all recorded trainings
    #benchmark.summary_report()
    # From here on, one should be able to use already
    # defined methods as shown in the following lines.
    # Let us know if you face any issues with that.

    # training history plots etc.
    nn.evaluation(
        model=model,
        history=history,
        dataset=ds,
        x_test=x_test,
        y_test=y_test,
        model_name = nn.model_name,
        path = filepath,
    )
