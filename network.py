import numpy as np
import pandas as pd
import uuid
import sys

sys.setrecursionlimit(1000000)

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
            print('Best model has been saved as '+location+str(hash)+'.pkl')
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
        errors=[]
        change=[]
        bestmodel=None
        badstreak=0
        hash=str(uuid.uuid4().hex)
        saved=False
        samplesize=168
        x_trainfull=x_train
        y_trainfull=y_train
        x_train1=x_train[-samplesize:]
        y_train1=y_train[-samplesize:]

        if verbose==True:
            print('This session will be saved as '+location+hash+'.pkl')

        if dynamic_learning==True:
            original_learning_rate=learning_rate
        
        naterr=0
        besterr=2**256
        # training loop
        for i in range(epochs):
            err = 0

            if autoregress==True:
                prior=0

            if dynamic_learning==True:
                learning_rate=original_learning_rate
            
            if i<0.8*epochs:
                sampleswitch=False
                x_train=x_train1
                y_train=y_train1
            elif i==round(0.8*epochs):
                sampleswitch=True
                x_train=x_trainfull
                y_train=y_trainfull
            
            samples = len(x_train)

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

                if autoregress==True:
                    prior=output
                
                # backward propagation
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                
                err+=np.power(y_train[j]-output, 2)

                if i==0 or sampleswitch==True:
                    naterr+=np.power(y_train[j], 2)



            # calculate average error on all samples

            err = np.mean(err)
            naterr=np.mean(naterr)

            if (round(err,5)>=round(naterr,5)) or (err==np.nan):
                print('Training session terminated due to blockage.')
                break

            err=err**0.5
            
            if err<besterr or sampleswitch==True:
                sampleswitch=False
                besterr=err
                badstreak=0
                bestmodel=[]
                for layer in self.layers:
                    if layer.export()!='Activation':
                        bestmodel.append([layer.export()])
                bestmodel=[[bestmodel],structure]
                saved=False
            else:
                badstreak+=1


            # if err <besterr:
            #     besterr=err
            #     bestmodel=[]
                
            #     # for layer in self.layers:
            #     #     if layer.export()!='Activation':
            #     #         bestmodel.append([layer.export()])
            #     # bestmodel=[[bestmodel],structure]
            #     #self.export_best_model(bestmodel,location,hash)
            #     if verbose==True:
            #         print('New best model overwrote the prior: '+location+hash+'.pkl ')

            errors.append(err)
            if i>0:
                change.append(err/errors[i-1])
            
            #ineffective early stopping mechanism
            if i!=0:
                masize=np.minimum(i,5)
                fiveMAchange=1-np.nanmean(change[-masize:])
            else:
                fiveMAchange=1
            
            # if i>=3:
            #     if ((fiveMAchange<=ImprovementThreshold and i>10) and (change[len(change)-1]>=1)) or np.round(change[-3:],6).tolist().count(1.000000)==3:
            #         print('Trainated due to ineffective learning.      5MA=%f      RMSE=%f' % (fiveMAchange,err))
            #         
            if verbose==True:
                print('epoch %d/%d     RMSE=%f    change=%f     change 5 epoch MA=%f     best RMSE=%f       time since last minimum=%f' % (i+1, epochs, err,err/errors[i-1],fiveMAchange,besterr,badstreak))
            
            if ((badstreak>=10) and ((1-(err/besterr))<=-ImprovementThreshold)) and err/errors[i-1]>1:
                print('Training session terminated due to improvement threshold.')
                self.export_best_model(bestmodel,location,hash)
                break

            if badstreak>=10 and saved==False:
                self.export_best_model(bestmodel,location,hash)
                saved=True
        return bestmodel,hash