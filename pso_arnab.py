# -*- coding: utf-8 -*-
"""PSO_arnab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZfzIZtBs5AoAorTk-1ljl6CJkOORjRBb
"""

from google.colab import files
import zipfile, io, os

def upload_dir_file(case_f):
    # author: yasser mustafa, 21 March 2018  
    # case_f = 0 for uploading one File or Package(.py) and case_f = 1 for uploading one Zipped Directory
    uploaded = files.upload()    # to upload a Full Directory, please Zip it first (use WinZip)
    for fn in uploaded.keys():
        name = fn  #.encode('utf-8')
        #print('\nfile after encode', name)
        #name = io.BytesIO(uploaded[name])
    if case_f == 0:    # case of uploading 'One File only'
        print('\n file name: ', name)
        return name
    else:   # case of uploading a directory and its subdirectories and files
        zfile = zipfile.ZipFile(name, 'r')   # unzip the directory 
        zfile.extractall()
        for d in zfile.namelist():   # d = directory
            print('\n main directory name: ', d)
            return d
print('Done!')

file_name = upload_dir_file(0)

import numpy as np
from time_series import *

training_file='Train4.csv'
test_file='Test4.csv'

npop=2
max_epochs=2
c1=2
c2=2
ndim=3
w=0.5
max_lim=np.array([120,50,5])
min_lim=np.array([20,10,2])
pop=np.zeros((npop,ndim))
fit=np.zeros(npop)
gbest=[np.empty([1,ndim]),0]

params={'timestep':20,
        'dropout_prob':0.1,
        'num_units': 50,
        'num_hidden_layers':4,
        'mode': 'training'}

rnn=time_series_analysis(training_file,test_file)

pop=min_lim+np.random.rand(npop,ndim)*(max_lim-min_lim)
vel=np.zeros([npop,ndim])
for i in range(npop):
  
    timestep,num_units,num_hidden_layers=pop[i,:]
    params['timestep']=int(round(timestep))
    params['num_hidden_layers']=int(round(num_hidden_layers))
    params['num_units']=int(round(num_units))
    
    fit[i]=np.divide(1,1+rnn.get_validation_error(params,1))
    
localbest_fit=fit
localbest=pop

gbest[0]=pop[np.argmax(fit),:]
gbest[1]=max(fit)

for iter in range(max_epochs):
  
  for n in range(npop):
    
    vel[n,:]=w*vel[n,:]+c1*np.random.randn()*(gbest[0]-pop[n,:])+c2*np.random.randn()*(localbest[n,:]-pop[n,:])
    
    for d in range(ndim):
      
      vel[n,d]=max(min_lim[d],vel[n,d])
      vel[n,d]=min(max_lim[d],vel[n,d])
      
    pop[n,:]=pop[n,:]+vel[n,:]
    
    for d in range(ndim):
      
      pop[n,d]=max(min_lim[d],pop[n,d])
      pop[n,d]=min(max_lim[d],pop[n,d])
    
    timestep,num_units,num_hidden_layers=pop[n,:]
    
    params['timestep']=int(round(timestep))
    params['num_hidden_layers']=int(round(num_hidden_layers))
    params['num_units']=int(round(num_units))
    
    fit[n]= np.divide(1,1+rnn.get_validation_error(params,1))
    
    if fit[n]>localbest_fit[n]:
      localbest_fit[n]=fit[n]
      localbest=pop[n,:]
      
  gbest[1]=max(localbest_fit)
  gbest[0]=pop[np.argmax(localbest_fit),:]
  w=w*0.99

timestep_opt,num_units_opt,num_hidden_opt=gbest[0]
params['timestep']=int(round(timestep))
params['num_hidden_layers']=int(round(num_hidden_layers))
params['num_units']=int(round(num_units))

print('The optimum timestep, number of units per layer and number of hidden layers are given respectively:',round(timestep),round(num_units),round(num_hidden_layers))

params['mode']='predict'
rnn.plot_stock_price(params,500)