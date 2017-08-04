"""
Created by Yiwei Zhuang Oct. 29th 2016
Edited by Huck Zou Oct. 31th 2016
"""

from __future__ import division
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
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
# path1, path2,path3,path4,
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

def accuracy(predict_label, true_label):
	incorrect = 0
	correct = 0
	total	= len(predict_label)
	correct_pred = np.ones(3)
	for i in xrange(len(predict_label)):
		#print predict_label[i]
		#print true_label[i]
		if predict_label[i] != true_label[i]:
			incorrect+=1
			# print "classifier has made a mistake"
			# print "predict label: ", predict_label[i]
			# print "true label: ", true_label[i]
		else:
			correct_pred[true_label[i]] = correct_pred[true_label[i]] + 1
			correct+=1

	accuracy = correct/total
	# print "#total = ", total
	# print "#incorrect = ",incorrect
	# print "#correct = ", correct
	# print "accuracy = ", accuracy
	temp = np.array(true_label)
	pred_temp = np.array(predict_label)
	# print "percentage of zeros = ", len(temp[temp==0])/len(temp)
	# print "accuracy of predicting zeros = ", correct_pred[0]/len(temp[temp==0]) 
	# print "accuracy of predicting ones = ", correct_pred[1]/len(temp[temp==1])
	# print "accuracy of predicting twos = ", correct_pred[2]/len(temp[temp==2])
	return accuracy, correct_pred[1]/len(temp[temp==1]), correct_pred[2]/len(temp[temp==2])



def generate_misclassified_data(test_df, test_y, pred_y):
	bool_arr = test_y != pred_y
	res_df = test_df.loc[bool_arr, [0,len(test_df.columns)-1]]
	res_df[2] = pred_y[bool_arr]
	res_df.to_csv("misclassified_data.csv")



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

'''
#activation : {identity, logistic, tanh, relu}, default relu
#solver : {lbfgs, sgd, adam}, default adam
'''#
clf = MLPClassifier(solver='lbgfs', activation = 'logistic', alpha=1e-5, hidden_layer_sizes=(10,5), random_state=1,max_iter =10000)
#relu is not as stable as logistic, because the gradient issue?
#lbfgs peforms significantly better than sgd, because it is second-order update which 
# takes the model closer to the optimal per iteration, but the cost per
# iteration is bigger than sgd

# predict_dat = Process_data(path1)
# This returns a pandas dataframe that contains all the features and labels data
df = Merge_csv_DataFrame(paths)


# #========================================================#
# #Multiple test runs to determine the avg accuracy of our NN model
# #========================================================#

# num_runs = 5
# accu_arr = np.zeros(num_runs)
# accu_ones = np.zeros(num_runs)
# accu_twos = np.zeros(num_runs)
# for itr in range(num_runs):
# 	# Split the data from train and test sets, 80% train, 20% test
# 	train_df, test_df = train_test_split(df, test_size = 0.2)
# 	# Serialize train_df and test_df for features and labels separately
# 	train_data = Process_mutiple_data(train_df)
# 	test_data = Process_mutiple_data(test_df)

# 	clf.fit(train_data[0],train_data[1])
# 	# clf.coefs_ returns the weights for the trained model
# 	# print(clf.coefs_)
# 	predict_label = clf.predict(test_data[0]).tolist()
# 	accu_arr[itr], accu_ones[itr], accu_twos[itr] = accuracy(predict_label,test_data[1])

# print "summary: "
# print "min accuracy = ", accu_arr.min()
# print "max accuracy = ", accu_arr.max()
# print "mean accuracy = ", accu_arr.mean()
# print "mean ones accuracy = ", accu_ones.mean()
# print "mean twos accuracy = ", accu_twos.mean()
# print "accuracy SD = ", accu_arr.std()
# print "ones SD = ", accu_ones.std()
# print "twos SD = ", accu_twos.std()
# #========================================================#

train_df, test_df = train_test_split(df, test_size = 0.2)
train_data = Process_mutiple_data(train_df)
test_data = Process_mutiple_data(test_df)
y_pred = clf.fit(train_data[0],train_data[1]).predict(test_data[0])
print clf.score(test_data[0], test_data[1])
generate_misclassified_data(test_df, test_data[1],y_pred)
#========================================================#
#Print the Confusion matrix of our classification result
#========================================================#
np.set_printoptions(precision=2)
cnf_matrix = confusion_matrix(test_data[1], y_pred)

class_names = ["Neutral","Bullish","Bearish"]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

