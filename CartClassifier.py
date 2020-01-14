# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:23:24 2019

@author: GANESH
"""


import pandas as pd # Read csv

from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model
from sklearn import tree
from sklearn.tree.export import export_text
import numpy as np

from statistics import mean

# Leaf node has the classified data (for printing)
class Leaf_Node:
    def __init__(self, rows):
        self.predictions = self.__count_class(rows)
    
    # counts each number of classes int the rows given just to give count while printing
    def __count_class(self,rows):
        class_counts={} # label and count dictionary
        for row in rows:
            label=row[-1] # label will be in last column
            if label not in class_counts:
                class_counts[label]=0
            class_counts[label]+=1
        return class_counts
    
# Condition to split the data
class Condition:
    def __init__(self,feature,f_value):
        self.feature=feature
        self.f_value=f_value
    
    def match(self,row):
        return row[self.feature]>=self.f_value
    
    def __repr__(self):
        return "Is Feature[%s] >= %s" % (self.feature, self.f_value)

# Has the decision made
class Decision_Node:
    def __init__(self,condition,positive_branch,negative_branch):
        self.condition=condition
        self.positive_branch=positive_branch
        self.negative_branch=negative_branch
        
class CART_Classifier:
    # counts each number of classes int the rows given
    def __count_class(self,rows):
        class_counts={} # label and count dictionary
        for row in rows:
            label=row[-1] # label will be in last column
            if label not in class_counts:
                class_counts[label]=0
            class_counts[label]+=1
        return class_counts
    
    # splits the rows into postives and negatives
    def __split(self,rows,condition):
        positive_rows,negative_rows=[],[]
        for row in rows:
            if condition.match(row):
                positive_rows.append(row)
            else:
                negative_rows.append(row)
        return positive_rows,negative_rows
    
    # Calculate gini index
    def __gini_index(self, rows):
        # has count of each class in the rows
        counts=self.__count_class(rows)
        # intital impurity
        impurity=1
        # gini_index = 1- summation(square of (probability of i))
        for label in counts:
            label_probability= counts[label]/float(len(rows))
            impurity-=label_probability**2
        return impurity
    
    # Information Gain:
    def __info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        # Reduce uncertainity after split
        return current_uncertainty - p * self.__gini_index(left) - (1 - p) * self.__gini_index(right)
    
    # iterates over all feature and its values to find the best information gain
    # finds the best condition to split on
    def __best_split(self,rows):
        max_gain=0 # has the best calculaed information gain
        best_condition=None # has the conditon with max gain
        current_uncertainity= self.__gini_index(rows)
        num_features=len(rows[0])-1 # calculating number of features assuming no missing values
        
        for column in range(num_features):
            # get set of all unique values in a feature
            values=set([row[column] for row in rows])
            
            for value in values:
                condition= Condition(column,value)
                
                #split the branch into two
                positive_rows,negative_rows=self.__split(rows,condition)
                
                #skip the split if data is undivided no point in adding useless conditions
                if len(positive_rows) == 0 or len(negative_rows) == 0:
                    continue
                
                #Calculate info gain
                gain=self.__info_gain(positive_rows, negative_rows, current_uncertainity)
                
                #if gain is greater than the gain claculated before it
                if gain>max_gain:
                    max_gain,best_condition=gain,condition
                    
        return max_gain,best_condition
                
        
    #Build the actual tree
    def __build_tree(self,rows):
        gain,condition=self.__best_split(rows)
        
        # When there is no more gain return the tree
        if gain==0:
            return Leaf_Node(rows)
        
        left_rows,right_rows= self.__split(rows,condition)
        
        #Split the left and right rows into branches again
        left_branch=self.__build_tree(left_rows)
        right_branch=self.__build_tree(right_rows)        
        
        return Decision_Node(condition,left_branch,right_branch)
    
    def fit(self, training_data, training_label):
        rows = np.hstack((training_data, training_label[:, None])) # concatenate 2 arrays as 1D array
        self.__root = self.__build_tree(rows)
    
    def __predict_node(self, test_row, node, mode='detail'):
        # If terminal node/ Leaf_Node is reached
        if isinstance(node, Leaf_Node):
            # mode=details returns all labels predicted as {label:count} for printing decision tree
            if mode=='detail':
                return node.predictions
            # returns only the predicted label with highest probability (to support metrics for sklearn)
            else:
                return max(node.predictions, key=node.predictions.get)
            
        # go to the branch that matches the condition            
        if node.condition.match(test_row):
            return self.__predict_node(test_row, node.positive_branch, mode)
        else:
            return self.__predict_node(test_row, node.negative_branch, mode)
            
    def predict(self,test_data,mode='detail'):
        # if data has only one row call predict otherwise make into a single row and then call it
        if test_data.ndim==1:
            return self.__predict_node(test_data, self.__root, mode) 
        else:
            predictions=[]
            for row in test_data:
                predictions.append(self.__predict_node(row, self.__root, mode) )
            return predictions

    # prints decision tree
    def __print_tree(self, node, spacing=" "):
        if isinstance(node, Leaf_Node):
            print(spacing + 'Predict', node.predictions)
            return
        
        print(spacing + str(node.condition))
    
        print(spacing + '--> True')
        self.__print_tree(node.positive_branch, spacing + "   ")
    
        print(spacing + '--> False')
        self.__print_tree(node.negative_branch, spacing + "   ")
        
    def visualize(self, spacing=" "):
        self.__print_tree(self.__root, spacing)
        
# reads the file,transposes and adds labels
def preprocess(path):
    raw_data = pd.read_csv(path, delimiter='\t',encoding='latin1', na_values="n/a", index_col = 0)
    data_transpose = raw_data.T
    return data_transpose

# split data into labels, feature values train and test
def split_data(data_preprocessed):
    feature_values= np.array(data_preprocessed.drop(columns = [data_preprocessed.columns[-1]]))
    labels=np.array(data_preprocessed[data_preprocessed.columns[-1]])
    # 1/3 =.333 is taken as test size
    data_train, data_test, label_train, label_test = train_test_split(feature_values, labels, test_size=0.333)
    return data_train, data_test, label_train, label_test

# trains classifier and returns it
def trainDecisionTree(data_train, label_train):
    classifier = CART_Classifier()
    classifier.fit(data_train, label_train)
    return classifier

# prints decision trees, accuracy of each classifier and report using scikit
def evaluate(data_test, label_test, classifier):
        predicted_labels=classifier.predict(data_test,'no_detail')
        accuracy=metrics.accuracy_score(label_test,predicted_labels)
        if(isinstance(classifier,CART_Classifier)):
            classifier.visualize()
        else:
            print(export_text(classifier))
        print('Accuracy :')
        print(accuracy)
        print('\nMetrics :')
        print(metrics.classification_report(label_test, predicted_labels))
        print('Confusion matrix :')
        print(metrics.confusion_matrix(label_test, predicted_labels))
        rows = np.vstack((label_test, predicted_labels)).T
        return accuracy,rows
            
def main():
    # preprocess and read file
    path=input('Enter the path of the file to read: ')
    data_preprocessed=preprocess(path)
    my_accuracies=[]
    sckit_accuracies=[]
    output_rows=[['Actual label','Predicted label']]

    # making random splits of test and training data and making reports
    for i in range(10):
        data_train, data_test, label_train, label_test = split_data(data_preprocessed)
        my_classifier = trainDecisionTree(data_train, label_train)
        sckit_classifier= tree.DecisionTreeClassifier()
        sckit_classifier.fit(data_train,label_train)
        print('\nMy Classifier '+ str(i+1) + ':')
        my_accuracy,my_predictions=evaluate(data_test, label_test, my_classifier)
        my_accuracies.append(my_accuracy)
        output_rows=np.concatenate((output_rows,my_predictions))
        print('\nScikit Classifier '+ str(i+1) + ':')
        sckit_accuracy,_=evaluate(data_test, label_test, sckit_classifier)
        sckit_accuracies.append(sckit_accuracy)
        
    # calculate mean accuracy for scikit models and my algorithm
    print('\nMean accuracy : ')
    print('My Algorithm : ' + str(mean(my_accuracies)) )
    print('Scikit Algorithm : ' + str(mean(sckit_accuracies)))
    # write the actual and predicted values as csv
    np.savetxt('predictions.csv', output_rows,delimiter=',', fmt="%s",)

if __name__ == '__main__':
    main()
    
    
    
    