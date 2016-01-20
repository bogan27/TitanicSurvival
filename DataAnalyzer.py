# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:12:16 2016

@author: brandonbogan and Sarah Cooper
"""

# Import Modules
import numpy as np
import pandas as pd
from pandas.stats.api import ols
import statsmodels.api as sm
import DataQualityTool as dqt
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
import sklearn.feature_selection 
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt


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
    
    ## Return the data currently being used by this tool
    def getData(self):
        return self.data
      
    ## Determine what type of analysis the user wants, instantiate an object 
    ## that can perform that analysis, and execute it
    def analyze(self, analysisType):
        ## For linear regression
        if analysisType == "lr":
            regTool = RegressionTool(self.data, self.response)
            print regTool.analyzeLinReg()
            
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
                
##############################Build Feature Importance Tools##################       
    def univariateFeatureselection(self, data, scoringfunction, nooffeatures, response):
         X_new = SelectKBest(scoringfunction, k=nooffeatures).fit_transform(data, response)
         return X_new
         
         
    def recurrsiveFeatureselction(self, data, response):
        # Create the RFE object and compute a cross-validated score.
        svc = SVC(kernel="linear")
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(response, 2),
              scoring='accuracy')
        rfecv.fit(data, response)
        print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
        
        
        
## A class for performing linear regression
class RegressionTool:
    
    ## Instantiate the class with a dataset and the col index of the response
    ## variable of that set
    ## Argument for data should be a Pandas DataFrame, and the argument for 
    ## response should be a non-negative integer
    def __init__(self, data, response):        
        self.data = data
        self.response = response
        
    ## Return the current model's data, as a Pandas DataFrame
    def getData(self):
        return self.data
        
    ## Set the DataFrame that should be used for this model.
    ## If not a DataFrame, the new data will not be used. 
    def setData(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            print 'WARNING! DATA REJECTED!'
            print 'Argument for data must be a pandas DataFrame'
        
    ## Returns a Panda DataFrame showing the coefficient, MSE, and variance 
    ## score for each feature
    def analyzeLinReg(self):
        
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
        
    ## Takes in a Pandas DataFrame and a list of column names to be treated as 
    ## nominal variables, and returns a Pandas DataFrame. 
    ## For each column in the list, dummy variables will be added to the 
    ## DataFrame for each value in the column, then the column will be removed.
    def convertCatsToDummies(self, data, cols):
        if isinstance(data, pd.DataFrame) and isinstance(cols, list):
            for col in cols:
                dummies = pd.get_dummies(data[col])
                dummyNames = []
                for d in dummies:
                    name = col + "_" + d
                    dummyNames.append(name)
                dummies.columns = dummyNames
                data = pd.concat([data, dummies], axis = 1)
                data = data.drop(col, 1)
            return data
        else:
            print ""
            print "ERROR! Invalid data types of arguments."
            print ""
            
    ## Builds a regression model based on the given data set, which must be a 
    ## Pandas DataFrame, and the column name of the dependent variable
    def buildModel(self, data, depenVar):
        if not isinstance(data, pd.DataFrame) or not isinstance(depenVar, str):
            print '\n' + "ERROR! Wrong Data Types! Takes a DataFrame and a string." + '\n'
        else:
            data = data[np.isfinite(data[depenVar])]
            dependent = data[depenVar]
            independents = data.drop(depenVar,1)
            regr = linear_model.LinearRegression()
            return regr.fit(independents, dependent)
            
    ## Fill in missing values with the predicted values
    ## Takes in a DataFrame, the name of the column to fill in, and an optional
    ## code for a distribution family (defaiults to Gaussian) for the dependent
    ## variable. The following are codes that correspond to available families:
    ##      1 - Binomial
    ##      2 - Gamma
    ##      3 - Gaussian
    ##      4 - Inverse Gaussian
    ##      5 - Negative Binomial
    ##      6 - Poisson
    def predictNullsGLM(self, d, depenVar, familyCode=3, roundTo=2):
        if not isinstance(d, pd.DataFrame) or not isinstance(depenVar, str):
            print '\n' + "ERROR! Wrong Data Types! Takes a DataFrame and a string." + '\n'
        else:
            distFams = [sm.families.Binomial(), sm.families.Gamma(),
                        sm.families.Gaussian(), sm.families.InverseGaussian(),
                        sm.families.NegativeBinomial(), sm.families.Poisson()]
            fam = distFams[familyCode - 1]
            dClean = d.dropna()
            dat = np.asarray(dClean[depenVar])
            model = sm.GLM(dat, exog= np.asarray(dClean.drop(depenVar, 1)), 
                           family=fam)
            model =  model.fit()
            estimates = model.predict(exog = d.drop(depenVar,1))
            estimateDF = d.copy()
            estimateDF[depenVar] = estimates
            estimateDF[depenVar] = estimateDF[depenVar].round(roundTo)
            return d.combine_first(estimateDF)
            
    ## Fill in the missing values with the predicted values
    ## Takes in a PD DataFrame and the String name of the dependent variable
    ## Note: Only works on numeric values
    def predictNullsOLS(self, data, depenVar):
            model = ols(y=data[depenVar], x=data.drop(depenVar,1))
            estimates = model.predict()
            estimateDF = data.copy()
            estimateDF[depenVar] = estimates
            return data.combine_first(estimateDF)
                    
                
        
        

