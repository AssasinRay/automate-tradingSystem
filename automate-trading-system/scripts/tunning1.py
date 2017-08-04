#tuning different parameter for NN

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

activation_functions = ['logistic', 'relu', 'tanh']
solver_functions = ['lbgfs', 'sgd', 'adam']
alpha_parameter = [ 1e-6, 1e-5,  5e-6, 1e-6]
hidden_layer_sizes = [(10),(13),(10,5),(10,10), (10,15), (10,10,10) , (13,13)] 

# activation_functions = ['logistic','relu']
# solver_functions = ['lbgfs']
# alpha_parameter = [ 1e-5]
# hidden_layer_sizes = [(10)] 

path1 = "../data/FB_features.csv"
path2 = "../data/AAPL_features.csv"
path3 = "../data/AMZN_features.csv"
path4 = "../data/CMG_features.csv"
path5 = "../data/F_features.csv"
path6 = "../data/JPM_features.csv"
path7 = "../data/JWN_features.csv"
path8 = "../data/KO_features.csv"
path9 = "../data/UA_features.csv"

paths = [path1, path2, path3, path4, path5,path6,path7,path8, path9]

def Merge_csv_DataFrame(paths):
	frames = []
	for path in paths:
		frames.append(pd.read_csv(path, header=None))
	result = pd.concat(frames)
	return result

def Process_mutiple_data(raw_data):
	num_cols 		= len(raw_data.columns)
	features 		= raw_data.loc[:,1:num_cols-2]
	labels	 		= raw_data.loc[:,num_cols-1]
	features_list 	= features.values.tolist()
	labels_list		= labels.values.tolist()
	return [features_list,labels_list]

def Process_data(path):
	raw_data 		= pd.read_csv(path,header=None)
	num_cols 		= len(raw_data.columns)
	features 		= raw_data.loc[:,1:num_cols-2]
	labels	 		= raw_data.loc[:,num_cols-1]
	features_list 	= features.values.tolist()
	labels_list		= labels.values.tolist()
	return [features_list,labels_list]

def normlize(matrix):
	total_0 = 0
	total_1 = 0
	total_2 = 0

	for i in xrange(3):
		total_0+=matrix[0][i]
		total_1+=matrix[1][i]
		total_2+=matrix[2][i]

	res = np.zeros((3, 3))
	total = [total_0,total_1,total_2]

	for i in xrange(3):
		for j in xrange(3):
			res[i][j]= round(float(matrix[i][j]/total[i]),5)

	return res

#df = Merge_csv_DataFrame(paths)

optimal_dict={}

def cal_error(pre, true):
	res = 0
	for i in xrange(len(pre)):
		res+= abs(pre[i]-true[i])
	return res
df = pd.read_csv("training_samples.csv", header=None)
for activation_f in activation_functions:
	for solver_f in solver_functions:
		for a in alpha_parameter:
			for hidden_layers in hidden_layer_sizes:
				error = 0.0
				parameter = activation_f +"\t"+ solver_f +"\t" + str(a) +"\t" + str(hidden_layers) 
				for i in xrange(10):
					
					clf = MLPClassifier(solver=solver_f, activation = activation_f, alpha=a, hidden_layer_sizes=hidden_layers, random_state=1,max_iter =10000)
					train_df, test_df = train_test_split(df, test_size = 0.2)
					train_data = Process_mutiple_data(train_df)

					test_data = Process_mutiple_data(test_df)

				
					model =  clf.fit(train_data[0],train_data[1])
					y_pred = model.predict(test_data[0])
					error += cal_error(y_pred, test_data[1])
				
				
				optimal_dict[parameter] = error

bestcombo =""
now =100000000 
for para, error in optimal_dict.iteritems():
	if error < now:
		bestcombo= para
		now = error


json.dump(optimal_dict, open("optimal_dict.txt",'w'))
print bestcombo
print now
