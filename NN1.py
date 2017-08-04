from __future__ import division
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

path1 = "./Data/progress_FB_three_phase_features.csv"
path2 = "./Data/progress_AAPL_three_phase_features.csv"
path3 = "./Data/progress_AMZN_three_phase_features.csv"
path4 = "./Data/progress_CMG_three_phase_features.csv"
path5 = "./Data/progress_F_three_phase_features.csv"
path6 = "./Data/progress_JPM_three_phase_features.csv"
path7 = "./Data/progress_JWN_three_phase_features.csv"
path8 = "./Data/progress_KO_three_phase_features.csv"
path9 = "./Data/progress_UA_three_phase_features.csv"

paths = [path2,path3,path4,path5,path6,path7,path8,path9]

def Merge_csv_DataFrame(paths):
	frames = []
	for path in paths:
		frames.append(pd.read_csv(path, header=None))
	result = pd.concat(frames)
	return result

def Process_mutiple_data(raw_data):
	num_cols 		= len(raw_data.columns)
	features 		= raw_data.loc[:,1:num_cols-1]
	labels	 		= raw_data.loc[:,num_cols-1]
	features_list 	= features.values.tolist()
	labels_list		= labels.values.tolist()
	return [features_list,labels_list]

def Process_data(path):
	raw_data 		= pd.read_csv(path,header=None)
	num_cols 		= len(raw_data.columns)
	features 		= raw_data.loc[:,1:num_cols-1]
	labels	 		= raw_data.loc[:,num_cols-1]
	features_list 	= features.values.tolist()
	labels_list		= labels.values.tolist()
	return [features_list,labels_list]

def accuracy(predict_label, true_label):
	incorrect = 0
	correct = 0
	total	= len(predict_label)
	for i in xrange(len(predict_label)):
		#print predict_label[i]
		#print true_label[i]
		if predict_label[i] != true_label[i]:
			incorrect+=1
		else:
			correct+=1

	accuracy = correct/total
	print "#total = ", total
	print "#incorrect = ",incorrect
	print "#correct = ", correct
	print "accuracy = ", accuracy
	return accuracy



#data = Process_data("/Users/Zhuangyiwei/Desktop/446project/Data/progress_FB_three_phase_features.csv")

#print data[1]

#clf = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1,max_iter =10000)
#clf.fit(data[0],data[1])
#print data[1]
#predict_label = clf.predict(predict_dat[0]).tolist()
#print predict_label
#print predict_dat[1]
#accuracy(predict_label, predict_dat[1])
#print res


predict_dat = Process_data(path1)
files = Merge_csv_DataFrame(paths)
files.sample(n=3)
all_data = Process_mutiple_data(files)

clf.fit(all_data[0],all_data[1])
predict_label = clf.predict(predict_dat[0]).tolist()
accuracy(predict_label,predict_dat[1])
