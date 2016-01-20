# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:20:29 2016

@author: Brandon Bogan and Sarah Cooper
"""
## Import Modeules
import pandas as pd
import os
import sys
import pylab as P
import string
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

## Local modules
sys.path.append("..")
from DataQualityTool import DataQualityTool
import DataAnalyzer as da




class TitanicSurivalModel:
    ## Initialize the model
    def __init__(self):
        os.chdir("Data")
        self.train = pd.read_csv("train.csv", index_col= 0)
        self.test = pd.read_csv("train.csv", index_col= 0)
        self.dqTool = DataQualityTool(self.train)
        self.dataAnalyzer = da.DataAnalyzer()
        self.regressionTool = da.RegressionTool(self.train, 0)
        
    ##Explortory Data Analysis
    ##Graphing each column to explore distrubtions and survivor rates
    def graphAge(self):
        self.train['Age'].hist()
        P.show()
        
    ## Get the training data set
    def getTrain(self):
        return self.train
        
    ## Set the training data set
    def setTrain(self, data):
        self.train = data
        
    ## Get the test data set of this model
    def getTest(self):
        return self.test
        
    ## Set the test data set
    def setTest(self, data):
        self.test = data
        
    ## Checks data for missing values
    def analyzeDataQuality(self):
        return self.dqTool.analyze()
        
    ## Takes in a string and a list of strings, and returns the first string 
    ## from the list that is found in the first argument, or null if none of 
    ## the listed strings are found. 
    def substrings_in_string(self, big_string, substrings):
        for substring in substrings:
            if string.find(big_string, substring) != -1:
                return substring
        print "Null value! None of the substrings were found!"
        print big_string
        return np.nan

    ## Converts titles to mr, mrs, miss, master, if applicables
    ## Even checks for gender of doctors :)    
    def replace_titles(self, data):
        title=data['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if data['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title
            
    ## Returns the given dataset, but with a title field, where a title is one
    ## of "Mr", "Mrs", or "Miss" 
    def calcTitles(self, data):
        title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
        data['Title']=data['Name'].map(lambda x: self.substrings_in_string(x, title_list))
        data['Title']=data.apply(self.replace_titles, axis=1)
        return data
    
    ## Calculates the family size for each person in the data set by adding the
    ## number of sibblines, number of parents, and the person themselves, then 
    ## returns the updated data set. 
    def calcFamilySize(self, data):
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
        return data
   
   #Calculating Fare per person using Family Size
   ##acts fare per person to data and returns data
    def farePerPerson(self, data):
       data["Fare_Per_Person"]=data["Fare"]/(data["FamilySize"]+1)
       return data
   
   #Turning cabin number into Deck
   #adds deck as feature and then returns data
    def cabintoDeck(self,data):
        cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
        data['Deck']=data['Cabin'].map(lambda x: self.substrings_in_string(str(x), cabin_list))
        ## Replace nulls with "Unknown"        
        data['Deck'] = data['Deck'].fillna("Unknown")
        return data
        
    def missingAge(self,data):
        age_glm = smf.glm(formula = 'Age \~ Title + FamilySize + Sex', df =data, 
                      datafamily=sm.families.Binomial()).fit()
        print age_glm.summary()

    ## Gets Correlation Matrix and returns of all columns
    def getCorr(self,data):
        corrMatrix = data.corr()
        return corrMatrix
    
    ## Adds family size, fare per family member, deck, and title to the given
    ## data set, then returns it    
    def addFeatures(self,data):
        data = self.calcFamilySize(data)
        data = self.farePerPerson(data)
        data = self.cabintoDeck(data)
        data = self.calcTitles(data)
        data['Embarked'].fillna('S')
        return data
    
    
    ## Removes Fare, Cabin, Name, and Ticket from the given data set, then 
    ## returns the dataset. These variables are removed because they are either 
    ## arbitrary strings, or represented in other variables. 
    def deleteFeatures(self, data):
        return data.drop("Fare",1).drop("Cabin",1).drop("Name",1).drop("Ticket",1).drop("SibSp",1).drop("Parch",1)
        
    
    ## Returns a list containing the column names or nominal variables
    def getNominalNames(self, data):
        Nominal  = self.dataAnalyzer.getNameOfNoms(data)
        return Nominal
        
    ## Takes in a Pandas DataFrame and a list of column names to be treated as 
    ## nominal variables, and returns a Pandas DataFrame. 
    ## For each column in the list, dummy variables will be added to the 
    ## DataFrame for each value in the column, then the column will be removed.
    def nominaltoDummy(self, data):
        nominals  = self.getNominalNames(data)
        rt = da.RegressionTool(data, 0)
        return rt.convertCatsToDummies(data, nominals)
        
    ## Completely preps the data for building models
    ## Takes one argument, a Pandas DataFrame
    ## Returns a DataFrame
    def prepData(self, data):
        if not isinstance(data, pd.DataFrame):
            print "\n" + "ERROR! Wrong type of data!" + "\n"
        else:
            data = self.addFeatures(data)
            data = self.deleteFeatures(data)
            data = self.nominaltoDummy(data)
            return data
        

          
################################################################################          
 ##############################Modelling#######################################         

    #Looking at Feature Importance using Entropy and Information Gain 
    #Given by Modelling with a RandomForest
    def featureImportance(self, data):
        ##make all inputs
        features_list = data.columns.values[1::]
        predictors = data.values[:, 1::]
        response = data.values[:, 0]
        fi_threshold = 15
        
        # Fit a tree to determine feature importance 
        ## add in to use Extra tree or Random Forest
        forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
        forest.fit(predictors, response)
        feature_importance = forest.feature_importances_
 
        # make importances relative to max importance
        #Get the indexes of all features over the importance threshold
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        important_idx = np.where(feature_importance > fi_threshold)[0]
        
           # Create a list of all the feature names above the importance threshold
        important_features = features_list[important_idx]
        print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):\n", \
            important_features

        ##get std for the graphing
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
        std =std*100
        # Get the sorted indexes of important features
        sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
        for f in range(sorted_idx.shape[0]):
            print("%d. feature %d (%s)" % (f + 1, sorted_idx[f], important_features[sorted_idx[f]]))
     
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importance")
        plt.bar(range(sorted_idx.shape[0]), feature_importance[important_idx][sorted_idx[::-1]],
            color="r", yerr=std[sorted_idx[::-1]], align="center")
        plt.xticks(range(sorted_idx.shape[0]), sorted_idx[::-1])
        plt.xlim([-1, sorted_idx.shape[0]])
        plt.show()
          
      
  #############Recursive Feature Elimination##############

    def RecurrciveFE(self, data):
        data= data.dropna()
        features_list = data.columns.values[1::]
        predictors = np.asarray(data.values[:, 1::])
        response = np.asarray(data.values[:, 0])
        estimator = SVC(kernel="linear")
        
        ###using cross validation to determine nooffeatures
        rfecv = RFE(estimator, step=1, cv=StratifiedKFold(response, 2), scoring = 'accuracy')
        rfecv.fit(predictors, response)
        RFE( )
        print("Optimal number of features : %d" % rfecv.n_features_)
        
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()        
        
        ##label as optimal #of features
        noffeatures = rfecv.n_features_  
        
        ##use rfe to determine top features
        selector = RFE(estimator,noffeatures , step=1)
        selector = selector.fit(predictors, response)
        ##creat index to get names
        index1 = np.where(selector.support_ == False)[0]
        index = np.argsort(selector.ranking_[index1])[::-1]
        feature_list_imp = features_list[index]

        for f in range(index.shape[0]):
            print("%d. feature %d (%s)" % (f + 1, index[f], feature_list_imp[index[f]]))
        print(selector.support_)
        print(selector.ranking_)
        
   

        
    
        

        
        