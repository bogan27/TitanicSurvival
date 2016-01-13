# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:20:29 2016

@author: brandonbogan
"""
## Import Modeules
import pandas as pd
import os
import sys
import pylab as P
import string
import numpy as np
sys.path.append("..")
from DataQualityTool import DataQualityTool


class TitanicSurivalModel:
    ## Initialize the model
    def __init__(self):
        os.chdir("/Users/brandonbogan/Desktop/DataScience/Python /Titanic/Data")
        self.train = pd.read_csv("train.csv", index_col= 0)
        self.test = pd.read_csv("train.csv", index_col= 0)
        self.dqTool = DataQualityTool(self.train)
        
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
       data["Fare_Per_Person"]=data["Fare"]/(data["Family_Size"]+1)
       return data
   
   #Turning cabin number into Deck
   #adds deck as feature and then returns data
    def cabintoDeck(self,data):
        cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
        data['Deck']=data['Cabin'].map(lambda x: self.substrings_in_string(x, cabin_list))
        return data

    ##gets Correlation Matrix and returns of all columns
    def getCorr(self,data):
        corrMatrix = data.corr()
        return corrMatrix
        
    def addFeatures(self,data):
        data = self.calcFamilySize(data)
        data = self.farePerPerson(data)
        data = self.cabintoDeck(data)
        data = self.calcTitles(data)
        return data
       
        
          
################################################################################          
 ##############################Modelling#######################################         
          
          
          
          
          
#class Passenger:
#    ## Initialize a passenger
#    def __init__(self, survived, pClass, name, sex, age, sibSp, parch, 
#                 fare, embarked):
#        self.survived = survived
#        self.pClass = pClass
#        self.name = name
#        self.sex = sex
#        self.age = age
#        self.sibSp = sibSp
#        self.parch = parch
#        self.fare = fare
#        self.embarked = embarked
#        self.title = ""
#        self.familySize = self.sibSp + self.parch
        
    
        
    
        

        
        