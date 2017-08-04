

from __future__ import division
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import json


activation_functions = ['logistic']
solver_functions = ['lbgfs']
alpha_parameter = [ 1e-6]
hidden_layer_sizes = [(13,13)] 
def Process_mutiple_data(raw_data):
	num_cols 		= len(raw_data.columns)
	features 		= raw_data.loc[:,1:num_cols-2]
	labels	 		= raw_data.loc[:,num_cols-1]
	features_list 	= features.values.tolist()
	labels_list		= labels.values.tolist()
	return [features_list,labels_list]

def cal_error(pre, true):
	res = 0
	for i in xrange(len(pre)):
		res+= abs(pre[i]-true[i])
	return res

optimal_dict={}
df = pd.read_csv("training_samples.csv", header=None)
for activation_f in activation_functions:
	for solver_f in solver_functions:
		for a in alpha_parameter:
			for hidden_layers in hidden_layer_sizes:
				error = 0.0
				parameter = activation_f +"\t"+ solver_f +"\t" + str(a) +"\t" + str(hidden_layers) 
				for i in xrange(1):
					
					clf = MLPClassifier(solver=solver_f, activation = activation_f, alpha=a, hidden_layer_sizes=hidden_layers, random_state=1,max_iter =10000)
					train_df, test_df = train_test_split(df, test_size = 0.2)
					train_data = Process_mutiple_data(train_df)

					test_data = Process_mutiple_data(test_df)

				
					model =  clf.fit(train_data[0],train_data[1])
					y_pred = model.predict(test_data[0])
					error += cal_error(y_pred, test_data[1])
				
				
				optimal_dict[parameter] = error

print optimal_dict