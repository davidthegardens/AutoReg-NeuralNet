import numpy as np

from network import Network,prepforautoregress
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime,ReLu,ReLu_prime
from losses import mse, mse_prime

# training data
# x_train = np.array([[[1]], [[0]], [[1]], [[0]],[[1]], [[0]], [[1]], [[0]],[[1]], [[0]]])
# y_train = np.array([[[0]], [[-0.1]], [[0.2]], [[-0.3]],[[0.4]], [[-0.5]], [[0.6]], [[-0.7]],[[0.8]], [[-0.9]]])

def initializemodel(structure,inputsize,outputsize,activation,loss):
    
    net = Network()
    if activation==tanh:
        activator='tanh'
        primeactive=tanh_prime
    elif activation==ReLu:
        activator='ReLu'
        primeactive=ReLu_prime

    if loss==mse:
        primeloss=mse_prime
    
    for i in range(len(structure)-1):
        net.add(FCLayer(inputsize, structure[i],activator))
        net.add(ActivationLayer(activation, primeactive))
        inputsize=structure[i]


    net.add(FCLayer(inputsize, outputsize,activator))
    net.add(ActivationLayer(activation, primeactive))

    # train
    net.use(loss, primeloss)

    return net

def netwrapper(x_train,y_train,predictioninput,structure,autoregress,epochs,learning_rate,verbose,dynamic_learning,early_modelling,location,ImprovementThreshold):

    copy_x_train=x_train
    #add empty input
    if autoregress==True:
        x_train=prepforautoregress(x_train)

    inputsize=x_train.shape[2]
    outputsize=y_train.shape[2]

    net=initializemodel(structure,inputsize,outputsize,tanh,mse)

    optimal_model,hash=net.fit(x_train, y_train, epochs=epochs, learning_rate=learning_rate,autoregress=autoregress,verbose=verbose,dynamic_learning=dynamic_learning,location=location,structure=structure,ImprovementThreshold=ImprovementThreshold)
    if optimal_model!=None:
        if early_modelling==True:
            out1=net.predict(copy_x_train,autoregress=autoregress,optimal_modelling=True,optimal_model=optimal_model)
            out2 = net.predict(predictioninput,autoregress=autoregress,optimal_modelling=True,optimal_model=optimal_model)
        else:
            out1=net.predict(copy_x_train,autoregress=autoregress,optimal_modelling=False,optimal_model=None)
            out2 = net.predict(predictioninput,autoregress=autoregress,optimal_modelling=False,optimal_model=None)
        
        return np.array(out1,dtype=float),np.array(out2,dtype=float),hash
    else:
        return 'Nothing',None,None

def predict_from_load(predictioninput,autoregress,file,outputsize):
    optimal_model=np.load(file,allow_pickle=True)
    structure=optimal_model[1]
    easystructure=[]
    for i in structure:
        easystructure.append(str(i))
    easystructure="_".join(easystructure)
    optimal_model=optimal_model
    net=initializemodel(structure,predictioninput.shape[2]+1,outputsize,ReLu,mse)
    out=net.predict(predictioninput,autoregress=autoregress,optimal_modelling=True,optimal_model=optimal_model)
    return np.array(out,dtype=float),easystructure


#net=netwrapper(x_train,y_train,x_train,[6,8,7],autoregress=True,epochs=10000,learning_rate=0.1,dynamic_learning=False,verbose=True)

# # test
# out = net.predict(x_train,autoregress=True)
# print(out)