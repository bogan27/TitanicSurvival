# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:41:15 2016

@author: brandonbogan and Sarah Cooper
"""
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from decimal import Decimal

class DataQualityTool: 
    def __init__(self, data):
        self.data = data
        
    ## Set the data to be used for this tool's analysis, if data is a DataFrame
    def setData(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            print( '\n' + "ERROR! Data must be a Pandas DataFrame!" + '\n' )
            
    ## Return the data currently being used by this tool
    def getData(self):
        return self.data
    
    ## Returns table with the following info for each column:
    ## Number of unique values, % of vallues that are unique, 
    ## number of NANs/blanks, percent of values that are null/missing
    def analyze(self):
        # Create the accumulator variable
        summaryDF = pd.DataFrame()
        labels = pd.Series(["unique_count", "unique_percent", "missing_count",
                            "missing_percent"])
                            
        ## Determine number of rows in the data
        totalCount = len(self.data.index)
        TWOPLACES = Decimal(10) ** -2
        totalCount = Decimal(totalCount).quantize(TWOPLACES)
        
        ## Iterate through the columns, accumulate, then return the statistics 
        ## on each column
        for col in self.data.columns:
            colData = self.data[col]
            uniqueCount = self.countUniques(colData)
            uniquePercent = uniqueCount / totalCount
            badValues = self.countBadValues(colData)
            badValPercent = badValues / totalCount
            
            ## Format percent values
            uniquePercent = Decimal(uniquePercent).quantize(TWOPLACES)
            badValPercent = Decimal(badValPercent).quantize(TWOPLACES)
            
            stats = pd.Series([uniqueCount, uniquePercent, badValues, 
                               badValPercent])
            summaryDF[col] = stats
        
        summaryDF.index = labels
        return summaryDF
        
    ## Count the number of unique values in the given data
    def countUniques(self, x):
        return len(x.unique())
        
    ## Count the number of missing or bad values
    def countBadValues(self, x):
        return x.isnull().sum()
        
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
            print( "" )
            print("ERROR! Invalid data types of arguments.")
            print( "" )
            
        
    ## Removes variables with low variance, such as boolean variables that are
    ## the same more than the threshold % of the time
    ## Takes in a DataFrame, and an optional threshold
    ## between 0 and 1 (default is .8)
    ## Returns a DataFrame
        
        ## NOT FINISHED!
        ## Look into using .var() method for pd dfs
#    def removeLowVarianceVars(self, data, threshold = 0.9):
#        if not isinstance(data, pd.DataFrame):
#            print "\n" + "ERROR! Wrong type of data - must be DataFrame." + "\n"
#        if threshold <= 0 or threshold >= 1:
#            print "\n" + "ERROR! Threshold must be between 0 and 1." + "\n"      
#        selector  = VarianceThreshold(threshold=(threshold * (1 - threshold)))
#        answer = selector.fit_transform(data.as_matrix)
        
        
        
        
        