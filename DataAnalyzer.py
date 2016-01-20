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
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold


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
                
##############################Build Feature Importance Tools#################################       

#################################1##########################################
    def univariateFeatureselection(self, data, scoringfunction, nooffeatures):
          """
        Univariate feature selection works by selecting the best features based on univariate
        statistical tests. It can be seen as a preprocessing step to an estimator.

        Parameters
        ----------
        data : DataFrame
            Input data, for which categorical variables should be converted
            response should be in 0 column, predictors in additional
        scoring function : object
            For regression: f_regression
            For classification: chi2 or f_classif 
            returns the univariate pvalue
        nooffeatures : Integer
            Selects the best specified number of features from predictors 
        
        Returns
        -------
        out : List
            Returns list of best predictors
          
        """
        predictors = data.values[:, 1::]
        response = data.values[:, 0]
        X_new = SelectKBest(scoringfunction, k=nooffeatures).fit_transform(predictors, response)
        return X_new
    
#################################2##########################################
    #Looking at Feature Importance using Entropy and Information Gain 
    #Given by Modelling with a RandomForest
    def featureImportance(self, data, fi_threshold):
        """
        Models data with a Extra tree classifer and using information
        importance feature to help aid dimensionality reduction. 

        Parameters
        ----------
        data : DataFrame
            Input data, for which categorical variables should be converted
            response should be in 0 column, predictors in additional
        fi_threshold : Integer
            The number you want to set as the % threshold for features
        
        Returns
        -------
        out : Plot
            A plot of the featues of most importance determined by the set 
            threshold given and standard deviation tick marks
          
        """
        
        ##make all inputs
        features_list = data.columns.values[1::]
        predictors = data.values[:, 1::]
        response = data.values[:, 0]
    
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
        
#################################3########################################## 
 #############Recursive Feature Elimination##############

    def recurrciveFE(self, data):
        """
        Uses Recurrcise Feature Elimination to determine the write number of 
        features before adding additional leads to overfitting &
         It works by recursively removing attributes and building a model on those 
         attributes that remain. It uses the model accuracy to identify 
         which attributes (and combination of attributes) contribute the 
         most to predicting the target attribute.

        Parameters
        ----------
        data : DataFrame
            Input data, for which categorical variables should be converted
            response should be in 0 column, predictors in additional

        Returns
        -------
        out : Plot
            A plot with the number of optimal number of features,
            which is then used to determine features of most
            importance returned in a print out to console
          
        """
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
 
 
 #####################################4#######################################
     def interactionV(self,data):
         from minepy import MINE
         m = MINE()
         m.compute_score(data, x**2)
         print m.mic()

##########################################################################################       
## A class for performing linear regression
class RegressionTool:
    
    def __init__(self, data, response):
        """
        Initialize a RegressionTool instance

        Parameters
        ----------
        data : DataFrame
            The default set of data to be used for analysis in this tool's methods
        response : int
            The index of the column containing the response variable. 
            
        """
        self.data = data
        self.response = response
        
    def getData(self):
        """Return the current model's data, as a Pandas DataFrame"""
        return self.data
        
    ## If not a DataFrame, the new data will not be used. 
    def setData(self, data):
        """
        Set the DataFrame that should be used for this model.
        
        Parameters
        ----------
        data : DataFrame
            The default set of data to be used for analysis in this tool's methods
            
        Returns
        -------
        out : *None*
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            print 'WARNING! DATA REJECTED!'
            print 'Argument for data must be a pandas DataFrame'
        
    ## Returns a Panda DataFrame showing the coefficient, MSE, and variance 
    ## score for each feature
    def analyzeLinReg(self, data=None, response=None):
        """ 
        Provides statistics on the variables of this instances data.
        
        More specifically, calculates the coefficient, MSE, and Variance Score
        for each independent variable.
        
        Parameters
        ----------
        data : DataFrame
            Default value is self.data, but an optional DataFrame can be passed
            to be analyzed instead, without having any affect on self.data. 
        TODO: Finish this document
        """
        if data is None:
            data = self.data
        if response is None:
            response = self.response
        
        ## Get the target variable 
        responseData = data.iloc[:,response]
        rowCount = len(data.index)
        responseTrain = responseData[0:rowCount/2]
        responseTest = responseData[(rowCount / 2):rowCount]
        
        ## Set up the df to store the response in
        resultsDF = pd.DataFrame()
        
        ## Set the counting variable to 0 and loop through the variables
        currentCol = 0
        for col in data.columns:
            ## If the current column is the index or the target variable, 
            ## increase the counter and move on
            if currentCol == response:
                currentCol += 1 
            ## Otherwise, perform the regression analysis
            else:
                currentVar = data.iloc[:,currentCol]
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
                colName = list(data.columns.values)[currentCol]
                resultsDF[colName] = pd.Series([coef, mse, variance])
                print "Done analyzing " + colName
                
                ## Increment the column
                currentCol += 1
                print ""
                print ""
        
        metrics = pd.Series(["Coefficient", "MSE", "Variance Score"])
        resultsDF.index = metrics
        return resultsDF
        
    def convertCatsToDummies(self, data, cols):
        """
        Converts specified catagorical variables into binary dummy variables,
        then drops the categorical variables. 

        Parameters
        ----------
        data : DataFrame
            Input data, for which categorical variables should be converted
        cols : List[str]
            A list of column names to convert from categorical to dummy

        Returns
        -------
        out : DataFrame
            The original data frame but without any columns names in the ``cols``
            argument, which are instead represented by the newly created dummy
            variables. 

        Example
        -------
        Suppose you have a DataFrame ``d`` that contains a column "Gender". You
        could use this method to convert the Gender column to two new variables, 
        gender_male and gender_female by calling:

        >>> categoricalD = convertCatsToDummies(d, ['Gender'])
        >>> categoricalD.columns
        ['Gender_male', 'Gender_Female']
          
        """
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
    def predictNullsGLM(self, d, depenVar, familyCode=3, link=None, roundTo=2):
        if not isinstance(d, pd.DataFrame) or not isinstance(depenVar, str):
            print '\n' + "ERROR! Wrong Data Types! Takes a DataFrame and a string." + '\n'
        else:
            distFams = [sm.families.Binomial(), sm.families.Gamma(),
                        sm.families.Gaussian(), sm.families.InverseGaussian(),
                        sm.families.NegativeBinomial(), sm.families.Poisson()]
            fam = distFams[familyCode - 1]
            if link is not None:
                fam._setlink(link)
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
                    
                
  
