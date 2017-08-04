import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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
data = Process_mutiple_data(df)


title = "Learning Curve MLP"
cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
estimator = clf 
plot_learning_curve(estimator, title, data[0], data[1], (0.5, 1.01), cv=cv, n_jobs=1)
plt.show()
