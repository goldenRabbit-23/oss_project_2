# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/goldenRabbit-23/oss_project_2

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_dataset(dataset_path):
	#To-Do: Implement this function
	dataset = pd.read_csv(dataset_path)
	return dataset

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	feats = dataset_df.shape[1] - 1
	class0 = dataset_df.groupby("target").size()[0]
	class1 = dataset_df.groupby("target").size()[1]
	return feats, class0, class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	X = dataset_df.drop(columns="target")
	Y = dataset_df["target"]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testset_size)
	return X_train, X_test, Y_train, Y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)

	dt_predict = dt_cls.predict(x_test)

	dt_acc = accuracy_score(y_test, dt_predict)
	dt_prec = precision_score(y_test, dt_predict)
	dt_recall = recall_score(y_test, dt_predict)

	return dt_acc, dt_prec, dt_recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)

	rf_predict = rf_cls.predict(x_test)

	rf_acc = accuracy_score(y_test, rf_predict)
	rf_prec = precision_score(y_test, rf_predict)
	rf_recall = recall_score(y_test, rf_predict)

	return rf_acc, rf_prec, rf_recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_pipe = make_pipeline(StandardScaler(), SVC())
	svm_pipe.fit(x_train, y_train)

	svm_predict = svm_pipe.predict(x_test)

	svm_acc = accuracy_score(y_test, svm_predict)
	svm_prec = precision_score(y_test, svm_predict)
	svm_recall = recall_score(y_test, svm_predict)

	return svm_acc, svm_prec, svm_recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
