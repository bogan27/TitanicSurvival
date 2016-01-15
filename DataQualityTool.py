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
        
        
        
        
        