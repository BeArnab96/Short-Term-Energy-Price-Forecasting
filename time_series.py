import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class time_series_analysis:
    
    def __init__(self,training_file,test_file):
        
        self.dataset_train=pd.read_csv(training_file)
        self.dataset_test=pd.read_csv(test_file)
        self.sc = MinMaxScaler(feature_range = (0, 1))
        
    def process_training_data(self,timestep,mode):
        #get input data in a form which can be processed
        
        #self.dataset_train=self.dataset_train.iloc[::-1]
        training_set=self.dataset_train.iloc[:,1:2].values
        #sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = self.sc.fit_transform(training_set)
        if mode=='predict':
        
            X_train = []
            y_train = []
            for i in range(timestep, len(training_set_scaled)):
            
                X_train.append(training_set_scaled[i-timestep:i, 0])
                y_train.append(training_set_scaled[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
    
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        elif mode=='training':
            X_train = []
            y_train= []
            for i in range(timestep, 2*timestep):
                X_train.append(training_set_scaled[i-timestep:i,0])
                y_train.append(training_set_scaled[i,0])
            X_train, y_train = np.array(X_train),np.array(y_train)

            X_train= np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        return X_train,y_train
    
    def predict(self,params,epoch):
        

        timestep=params['timestep']
        dropout_prob=params['dropout_prob']
        num_units=params['num_units']
        num_hidden_layers=params['num_hidden_layers']
        mode=params['mode']
        

        if mode=='predict':
            #get predictions
            self.fit(params,epoch)
            #self.dataset_test=self.dataset_test.iloc[::-1]

            # Getting the predicted stock price of 2019
            dataset_total = pd.concat((self.dataset_train['MCP'], self.dataset_test['MCP']), axis = 0)
            inputs = dataset_total[len(dataset_total) - len(self.dataset_test) - timestep:].values
            inputs = inputs.reshape(-1,1)
            inputs = self.sc.transform(inputs)
            X_test = []
            for i in range(timestep, timestep+len(self.dataset_test)):
                X_test.append(inputs[i-timestep:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_stock_price = self.regressor.predict(X_test)
            predicted_stock_price = self.sc.inverse_transform(predicted_stock_price)
        
        elif mode == 'training':
            
            #get predictions
            self.fit(params,epoch)
            #self.dataset_test=self.dataset_test.iloc[::-1]
            #dataset_test=self.dataset_train['Open']
            start=2*timestep
            end=int(round(2.5*timestep))
            dataset_test=self.dataset_train.iloc[start:end,:]

            dataset_train=self.dataset_train.iloc[0:2*timestep,:]
            # Getting the predicted stock price of 2019
            dataset_total = pd.concat((dataset_train['MCP'], dataset_test['MCP']), axis = 0)
            inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestep:].values
            inputs = inputs.reshape(-1,1)
            inputs = self.sc.transform(inputs)
            X_test = []
            for i in range(timestep, timestep+len(dataset_test)):
                X_test.append(inputs[i-timestep:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_stock_price = self.regressor.predict(X_test)
            predicted_stock_price = self.sc.inverse_transform(predicted_stock_price) 
        
        return predicted_stock_price
    
    def model(self,dropout_prob,num_hidden_layers,num_units,input_size):
        
        #define rnn model
        self.regressor=Sequential()
        # first RNN layer
    
        self.regressor.add(LSTM(units = num_units, return_sequences = True, input_shape = (input_size, 1)))
        self.regressor.add(Dropout(dropout_prob))
    
        #hidden layers
    
        for l in range(num_hidden_layers):
            if l==num_hidden_layers-1:
                Bool=False
            else:
                Bool=True
        self.regressor.add(LSTM(units = num_units, return_sequences = Bool))
        self.regressor.add(Dropout(dropout_prob)) 
    
        #output layer
    
        self.regressor.add(Dense(units = 1))
        self.regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        
    def fit(self,params,epoch):
        
        timestep=params['timestep']
        dropout_prob=params['dropout_prob']
        num_units=params['num_units']
        num_hidden_layers=params['num_hidden_layers']
        mode=params['mode']
        
        #fit the training data in the model
        X_train,y_train=self.process_training_data(timestep,mode)
        input_size=X_train.shape[1]
        self.model(dropout_prob,num_hidden_layers,num_units,input_size)
        
        # Fitting the RNN to the Training set
        self.regressor.fit(X_train, y_train, epochs = epoch, batch_size = 32)
        
    def get_validation_error(self,params,epoch):
        
        # get the validation error. This will be used as the objective function for heuristics
        if params['mode']=='predict':
            real_stock_price = self.dataset_test.iloc[:, 1:2].values    
            predicted_stock_price=self.predict(params,epoch)
        elif params['mode']=='training':
            start=2*params['timestep']
            end=int(round(2.5*params['timestep']))
            real_stock_price = self.dataset_train.iloc[start:end, 1:2].values    
            predicted_stock_price=self.predict(params,epoch)

        validation_error=mean_squared_error(real_stock_price,predicted_stock_price)
        return validation_error
        
    def plot_stock_price(self,params,epoch):

        timestep=params['timestep']
        dropout_prob=params['dropout_prob']
        num_units=params['num_units']
        num_hidden_layers=params['num_hidden_layers']
        mode=params['mode']
        
        # plot stock price
        real_stock_price = self.dataset_test.iloc[:, 1:2].values
        predicted_stock_price=self.predict(params,epoch)
        real_stock_price = self.dataset_test.iloc[:, 1:2].values
        
        plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
        plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
 
