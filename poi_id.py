
# coding: utf-8

# In[ ]:




# In[ ]:




# In[1]:

#!/usr/bin/python

import sys
import pickle
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest # load best feature selection class
from sklearn import cross_validation # load cross_validation function so as not to overfit data
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### Task 1: Select what features you'll use. 

def featureSelection(feature_number, features, labels):
    selectKfeatures = SelectKBest(k=feature_number) # get the k best features
    selectKfeatures.fit(features, labels)
    Kfeatures_list = selectKfeatures.get_support(indices=True) # Kfeatures_list is arrary of indices of the features_list we selected.
    selected_features_list = []
    for k in Kfeatures_list:
        selected_features_list.append(features_list[k + 1])
    print selected_features_list
    selected_features_list.insert(0, 'poi') # put label at front of list for later functions
    return selected_features_list


### Task 2: Remove outliers

def outlierRemoval(selected_features_list, my_dataset):
    
    # Find the mean value of each feature when poi and also when not poi.
    # get data points for each feature with max distance from mean feature value (for poi and not poi).
    
    ctr = 0
    ivalues = {x:0 for x in selected_features_list} # initiliaze feature mean values for 'poi' values
    # get total for each feature for 'poi' values
    for name in my_dataset:
        if my_dataset[name]['poi']:
            ctr += 1
            for fea in selected_features_list:
                if my_dataset[name][fea] != 'NaN':
                    ivalues[fea] = ivalues[fea] + my_dataset[name][fea]

    print 'poi number: ', ctr

    # get mean values for each feature for 'poi' values
    for feature in selected_features_list:
        ivalues[feature] = ivalues[feature] / ctr

    ctr = 0
    nvalues = {x:0 for x in selected_features_list} # initiliaze feature mean values for non 'poi' values
    # get total for each feature for non 'poi' values
    for name in my_dataset:
        if not my_dataset[name]['poi']:
            ctr += 1
            for fea in selected_features_list:
                if my_dataset[name][fea] != 'NaN':
                    nvalues[fea] = nvalues[fea] + my_dataset[name][fea]
                    
    # get mean values for each feature for non 'poi' values
    for feature in selected_features_list:
        nvalues[feature] = nvalues[feature] / ctr

    # create column in dataframe for each selected feature how much it deviates from mean.
    for name in my_dataset:
        if my_dataset[name]['poi']:
            for val in selected_features_list:
                if my_dataset[name][val] != 'NaN':
                    my_dataset[name][val + '_error'] = abs(my_dataset[name][val] - ivalues[val])
                else:
                    my_dataset[name][val + '_error'] = abs( 0 - ivalues[val])
        else:
            for val in selected_features_list:
                if my_dataset[name][val] != 'NaN':
                    my_dataset[name][val + '_error'] = abs(my_dataset[name][val] - ivalues[val])
                else:
                    my_dataset[name][val + '_error'] = abs( 0 - ivalues[val])

    max_names = []
    for val in selected_features_list:
        max_value = 0
        max_name = 'Blank'
        for name in my_dataset:
            if not my_dataset[name]['poi']:
                if my_dataset[name][val + '_error'] > max_value:
                    max_value = my_dataset[name][val + '_error']
                    max_name = name
        max_names.append([max_name, max_value])
    # max_names = sorted(max_names, key= lambda x: x[1])
    # delete added columns:
    for val in selected_features_list:
        for name in my_dataset:
            del my_dataset[name][val + '_error']
            
    for x in max_names:
        if my_dataset[x[0]]['poi']:
            max_names.remove(x)
            
    return max_names


### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI """
    fraction = 0.
    if all_messages != 0 and all_messages != "NaN":
        fraction = poi_messages / float(all_messages)
    return fraction

def addFeatures(my_dataset):
    # go through dictionary of enron employees.
    for name in my_dataset:
        # Each data point is a enron employee dictionary of features and their corresponding values.
        # compute new feature values for datapoint, and add to dictionary of employees
        data_point = my_dataset[name]
        fraction_from_poi = computeFraction( data_point["from_poi_to_this_person"], data_point["to_messages"] )
        data_point["fraction_from_poi"] = fraction_from_poi
        fraction_to_poi = computeFraction( data_point["from_this_person_to_poi"], data_point["from_messages"] )
        data_point ["fraction_to_poi"] = fraction_to_poi


### Task 4: Try a variety of classifiers

# naivebayes
from sklearn.naive_bayes import GaussianNB

# k nearest neighbors
from sklearn import neighbors

# decision tree
from sklearn.tree import DecisionTreeClassifier

# svm
from sklearn import svm 


### Task 5: Tune classifier to achieve better than .3 precision and recall 
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV

# no parameter tunes for Naive Bayes

# k nearest neighbors parameters to tune
k_params = {'n_neighbors': [2, 4, 6, 8, 10], 'weights': ['uniform', 'distance']}

# decision tree parameters to tune
d_params = {'min_samples_split': [ 2, 3, 4, 5, 6, 7, 8]}

# support vector machine parameter tune
svm_params = {'C': [1, 1e3, 1e4], 'gamma': [ 0, 0.1, 0.01, 1, 10]}


#Load dictionary containing the dataset of Enron employees and their features.
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
my_dataset = data_dict
print 'Number of data points: ', len(my_dataset) # get number of data points.

# Get a list of all available features in dataset. 
features_list = my_dataset[my_dataset.keys()[0]].keys()
print 'Number of features: ', len(features_list)

# Add features: TASK 3
addFeatures(my_dataset)
features_list = my_dataset[my_dataset.keys()[0]].keys() # update feature list after adding features
features_list.remove('email_address') # remove unselectable feature (identifier feature) 

from sklearn.preprocessing import StandardScaler, MinMaxScaler # will need to rescale features: try both scalers
from sklearn.pipeline import make_pipeline # will need scale and then test - use pipeline

# Remove Outliers: TASK 2 up to about 10% of outliers. 
# I tested outlier removal code below to see if classifiers performed better. They never 
# bested the best result without removing outliers. The code was not used to get best classifier.
# but I did see that there was errant data in data set. By going through the outlier set generated
# by the outlierRemoval function. 

"""
features_list.remove('poi')
features_list.insert(0, 'poi')
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scale = StandardScaler()
features = scale.fit_transform(features)
features_list = featureSelection(8, features, labels) # select features: TASK 1
# remove no n-poi greatest outlier for each feature selected
to_remove = outlierRemoval(features_list, my_dataset) # remove about 5% of non-poi outlier values
print to_remove
for name in to_remove:
    if name[0] in my_dataset:
        del my_dataset[name[0]]
features_list = my_dataset[my_dataset.keys()[0]].keys() # update feature list
features_list.remove('email_address')"""

# remove errant data that does not represent an individual.
del my_dataset['TOTAL']

# in loop: each iteration will specify number of features to select
# get features: SelectKBest features withn Number specified by loop iteration: Task 1
# try a variety of classifiers: Naive_Bays, KNN, Decision Trees, SVM: Task 4
# tune parameters of classifiers: parameters specified above: Task 5

'''
for i in range(1, 10, 1):
    
    # featureformat and targetfeaturesplit functions require our label to 
    # be at front of features list. 'poi' is our label.
    features_list.remove('poi')
    features_list.insert(0, 'poi')
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scale = StandardScaler()
    features = scale.fit_transform(features)
    features_list = featureSelection(i, features, labels) # select features: TASK 1
    print 
    print 'GaussianNB: '
    clf = make_pipeline(StandardScaler(), GaussianNB())
    test_classifier(clf, my_dataset, features_list)
    print 
    print 'K nearest Neighbors: '
    grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid = k_params, cv = 3)
    clf = make_pipeline(StandardScaler(), grid)
    test_classifier(clf, my_dataset, features_list)
    print
    print 'Decision Trees: '
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid = d_params, cv = 3) # no need to scale
    test_classifier(clf, my_dataset, features_list)
    print
    print 'SVM: '
    grid = GridSearchCV(svm.SVC(), param_grid = svm_params, cv = 3)
    clf = make_pipeline(StandardScaler(), grid)
    test_classifier(clf, my_dataset, features_list)     
    features_list = my_dataset[my_dataset.keys()[0]].keys() # update feature list
    features_list.remove('email_address') 
'''

# after going through up to 10 selected features (code above), first dumping outliers
# and then not dumping outliers it was determined one selected feature
# on a Decision Tree algorithm works best without dumping outliers. 

features_list.remove('poi')
features_list.insert(0, 'poi')
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scale = StandardScaler()
features = scale.fit_transform(features)
features_list = featureSelection(1, features, labels) # select features: TASK 1
clf = GridSearchCV(DecisionTreeClassifier(), param_grid = d_params, cv = 3)
dump_classifier_and_data(clf, my_dataset, features_list)

# just curious: hyperparameter value: 7
"""test_classifier(clf, my_dataset, features_list)
selectKfeatures = SelectKBest(k=1) # get the k best features
Kfeature = selectKfeatures.fit_transform(features, labels)
clf.fit(Kfeature, labels)
print clf.best_estimator_ """



# In[ ]:




# In[ ]:



