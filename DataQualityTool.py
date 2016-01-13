# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:41:15 2016

@author: brandonbogan and Sarah Cooper
"""
import pandas as pd
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
        
        
        
        