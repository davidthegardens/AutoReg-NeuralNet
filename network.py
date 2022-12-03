import numpy as np
import pandas as pd
import uuid


def prepforautoregress(x_train):
    sample=len(x_train)
    autoregressx_train=[]
    for idx in range(sample):
        newnew=np.append(x_train[idx][0],0)
        newnew=np.array([newnew],dtype=float)
        autoregressx_train.append(newnew)
    return np.array(autoregressx_train)

class Network:


    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None


    def export_best_model(self,model_data,location,hash):
        with open(location+str(hash)+'.pkl', 'wb') as f:
            np.save(f, np.array(model_data,dtype=object))
    

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data,autoregress,optimal_modelling,optimal_model):
        # sample dimension first
        if autoregress==True:
            input_data=prepforautoregress(input_data)
        
        samples = len(input_data)
        result = []
        if autoregress==True:
            prior=0
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]

            if autoregress==True:
                    output= input_data[i][0][:-1]
                    output=np.append(output,prior)
                    output=np.array([output],dtype=float)
            
            counter=0
            for layer in self.layers:
                if optimal_modelling!=False:
                    if layer.what()!='Activation':
                        output=layer.forward_prop_from_load(output,optimal_model[0][0][counter][0][0],optimal_model[0][0][counter][0][1])
                        counter=counter+1
                    else:
                        output = layer.forward_propagation(output)
                else:
                    output = layer.forward_propagation(output)

            prior=output
            result.append(output)
        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate,autoregress,verbose,dynamic_learning,location,structure,ImprovementThreshold):
        fiveMA=[]
        hash=str(uuid.uuid4().hex)
        if verbose==True:
            print('This session will be saved as '+location+hash+'.pkl ')
        # sample dimension first
        samples = len(x_train)

        if dynamic_learning==True:
            original_learning_rate=learning_rate
        
        besterr=2**256
        # training loop
        for i in range(epochs):
            err = 0

            if autoregress==True:
                prior=0

            if dynamic_learning==True:
                learning_rate=original_learning_rate

            for j in range(samples):

                # forward propagation
                output= x_train[j]

                if autoregress==True:
                    output= x_train[j][0][:-1]
                    output=np.append(output,prior)
                    output=np.array([output],dtype=float)
                
                if dynamic_learning==True:
                    learning_rate=original_learning_rate*(1+np.tanh(np.power(np.sum(np.power(y_train[j],2)),0.5)))

                for layer in self.layers:
                    output = layer.forward_propagation(output)
            
                # compute loss (for display purpose only)
                if verbose==True:
                    err += self.loss(y_train[j], output)

                if autoregress==True:
                    prior=y_train[j]
                
                # backward propagation
                error = self.loss_prime(y_train[j], output)


                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
        
            err /= samples
            if err <besterr:
                besterr=err
                bestmodel=[]
                
                for layer in self.layers:
                    if layer.export()!='Activation':
                        bestmodel.append([layer.export()])
                bestmodel=[[bestmodel],structure]
                self.export_best_model(bestmodel,location,hash)
                if verbose==True:
                    print('New best model overwrote the prior: '+location+hash+'.pkl ')

            if verbose==True:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        
            if len(fiveMA)<5:
                if len(fiveMA)==0:
                    fiveMA.append(1)
                else:
                    fiveMA.append(err/fiveMA[len(fiveMA)-1])
            else:
                fiveMA=[fiveMA[1],fiveMA[2],fiveMA[3],fiveMA[4],err/fiveMA[4]]
            
            if ((1-np.nanmean(fiveMA))>=ImprovementThreshold) and (len(fiveMA)>5):
                print('Training terminated due to ineffective learning.')
                break
        return bestmodel,hash