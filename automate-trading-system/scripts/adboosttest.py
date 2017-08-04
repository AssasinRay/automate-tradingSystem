
from __future__ import division
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np


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


df = Merge_csv_DataFrame(paths)
clf = MLPClassifier(solver="lbgfs", activation = "logistic", alpha=5e-06, hidden_layer_sizes=13, random_state=1,max_iter =10000)
#clf = RandomForestClassifier(n_estimators = 100)
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600,learning_rate=1.5,algorithm="SAMME")
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600,learning_rate=1.5)
train_df, test_df = train_test_split(df, test_size = 0.2)
train_data = Process_mutiple_data(train_df)
test_data = Process_mutiple_data(test_df)
y_pred = clf.fit(train_data[0],train_data[1]).predict(test_data[0])
cnf_matrix = confusion_matrix(test_data[1], y_pred)
norm_matrix = normlize(cnf_matrix)
print norm_matrix