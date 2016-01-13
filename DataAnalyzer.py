# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:12:16 2016

@author: brandonbogan
"""

# Import Modules
import numpy as np
import pandas as pd
import DataQualityTool as dqt
from sklearn import linear_model


# DataAnalyzer class directs the types of analysis to be done based on
# user input
class DataAnalyzer:
    
    ## Instantiate the class
    def __init__(self):
        self.dataUrl = 'http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv'
        self.index = 0
        self.response = 3

        self.data = pd.read_csv(self.dataUrl, index_col= self.index)
        
    ## Set the data for analysis
    def setData(self, data):
        self.data = data
      
    ## Determine what type of analysis the user wants, instantiate an object 
    ## that can perform that analysis, and execute it
    def analyze(self, analysisType):
        if analysisType == "regression":
            regModel = LinRegModel(self.data, self.response)
            print regModel.analyze()
            
    ## Given a DataFrame, returns a list containing the names of all
    ## variables determined to be nominal. A variable is considered nominal if
    ## its values are strings, and if the ratio of unique values to total 
    ## values falls below the threshold, which is 25% by default, but can be 
    ## specified by an optional second argument. 
    def getNameOfNoms(self, data, threshold=.25):
        uniquePercentages = dqt.DataQualityTool(data).analyze().loc[['unique_percent'],]      
        answer = []
        for column in data:
            curCol = data[column]
            isNominal = True
            if curCol.size < 1:
                isNominal = False
            elif type(curCol.iat[curCol.first_valid_index() - 1]) != str:
                isNominal = False
            elif uniquePercentages[column][0] >= threshold:
                isNominal  = False
            if isNominal:
                answer.append(column)
        return answer
                
       
       
## A class for performing linear regression
class LinRegModel:
    
    ## Instantiate the class
    def __init__(self, data, response):
        self.data = data
        self.response = response
        
    ## Returns a Panda DataFrame showing the coefficient, MSE, and variance 
    ## score for each feature
    def analyze(self):
        
        ## Get the target variable 
        responseData = self.data.iloc[:,self.response]
        rowCount = len(self.data.index)
        responseTrain = responseData[0:rowCount/2]
        responseTest = responseData[(rowCount / 2):rowCount]
        
        ## Set up the df to store the response in
        resultsDF = pd.DataFrame()
        
        ## Set the counting variable to 0 and loop through the variables
        currentCol = 0
        for col in self.data.columns:
            ## If the current column is the index or the target variable, 
            ## increase the counter and move on
            if currentCol == self.response:
                currentCol += 1 
            ## Otherwise, perform the regression analysis
            else:
                currentVar = self.data.iloc[:,currentCol]
                currentTrain = currentVar[0:(rowCount / 2)]
                currentTest = currentVar[(rowCount / 2):rowCount]                
                # Create linear regression object
                regr = linear_model.LinearRegression()
                # Train the model using the training sets
                regr.fit(currentTrain[:,np.newaxis], responseTrain)
                ## Calc the response metric values
                coef = regr.coef_
                ## For formatting
                coef = coef[0]
                mse = np.mean((regr.predict(currentTest[:,np.newaxis]) - responseTest) ** 2)                
                variance = regr.score(currentTest[:,np.newaxis], responseTest)                
                
                ## Log the metrics in the response df
                colName = list(self.data.columns.values)[currentCol]
                resultsDF[colName] = pd.Series([coef, mse, variance])
                print "Done analyzing " + colName
                
                ## Increment the column
                currentCol += 1
                print ""
                print ""
        
        metrics = pd.Series(["Coefficient", "MSE", "Variance Score"])
        resultsDF.index = metrics
        return resultsDF
        

