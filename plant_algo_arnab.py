# -*- coding: utf-8 -*-
"""Plant_algo_arnab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qyyp1LaHlH0WzkHjfXP7k7yZndx0rOdd
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

npop=4
max_seeds=8
max_epochs=3
ndim=3
alpha=0.9
max_lim=np.array([120,50,5])
min_lim=np.array([20,10,2])
pop=np.zeros((npop,ndim))
fit=np.zeros(npop)
max_fit=[np.empty([1,ndim]),0]

params={'timestep':20,
        'dropout_prob':0.1,
        'num_units': 50,
        'num_hidden_layers':4,
        'mode': 'training'}

rnn=time_series_analysis(training_file,test_file)

pop=min_lim+np.random.rand(npop,ndim)*(max_lim-min_lim)

for i in range(npop):
  
    timestep,num_units,num_hidden_layers=pop[i,:]
    params['timestep']=int(round(timestep))
    params['num_hidden_layers']=int(round(num_hidden_layers))
    params['num_units']=int(round(num_units))
    
    fit[i]=np.divide(1,1+rnn.get_validation_error(params,1))

max_fit[0]=pop[np.argmax(fit),:]
max_fit[1]=max(fit)

numSeeds=(max_seeds*(fit/sum(fit)))

for g in range(max_epochs):
  
  for n in range(npop):
    
    sigma=(np.log(g+1)/(g+1))*(np.abs(max_fit[0]-pop[n,:]))
    
    bestChild_fit=0
    bestChild=[]
    
    for s in range(int(numSeeds[n])):
      
      
      if np.random.rand()>0.5:
          
        mean=max_fit[0]
        newplant=np.random.normal(mean,sigma)+np.random.randn()*max_fit[0]-np.random.randn()*pop[n,:]
        
      else:
        
        mean=pop[n,:]
        newplant=np.random.normal(mean,sigma)
        
      for d in range(ndim):
        
        newplant[d]=min(newplant[d],max_lim[d])
        newplant[d]=max(newplant[d],min_lim[d])
      
      timestep,num_units,num_hidden_layers=newplant
      params['timestep']=int(round(timestep))
      params['num_hidden_layers']=int(round(num_hidden_layers))
      params['num_units']=int(round(num_units))
      
      child_fit=np.divide(1,1+rnn.get_validation_error(params,1))
                           
      if (bestChild_fit<child_fit) :
                       bestChild_fit=child_fit
                       bestChild=newplant
                                                   
    if (fit[n]<bestChild_fit) :
                       pop[n,:]=bestChild
                       fit[n]=bestChild_fit
                              
      
                           
  max_fit[0]=pop[np.argmax(fit),:]
  max_fit[1]=max(fit)
  max_seeds=int(round(max_seeds*alpha))
  numSeeds=np.floor(max_seeds*(fit/sum(fit)))

timestep_opt,num_units_opt,num_hidden_opt=max_fit[0]
params['timestep']=int(round(timestep))
params['num_hidden_layers']=int(round(num_hidden_layers))
params['num_units']=int(round(num_units))

print('The optimum timestep, number of units per layer and number of hidden layers are given respectively:',round(timestep),round(num_units),round(num_hidden_layers))

params['mode']='predict'
rnn.plot_stock_price(params,500)