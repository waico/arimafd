from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.diagnostic import het_arch
import numpy as np
from numpy import linalg
import pandas as pd
from sympy import diff, symbols, sympify, Symbol, poly
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AR
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time


class diff_integ:
    def __init__(self,seasons):
        """
        seasons - list of lag including del trend
        
        Comments: 1) The algorithm is not optimized, in terms of adding new element in dictionary instead adding new dictionary
        """
        self.seasons=seasons
    

            
    def fit_transform(self,data,return_diff=True):
        """
        return array of difference time series
        data - numpy
        """
        self.data=np.array(data)
        data=np.array(data)
        if (len(data)-sum(self.seasons) <= sum(self.seasons)) or (len(self.seasons) < 1):
            print('Error: too small lengths of the initial array')
        else:
            self.Minuend={}
            self.Difference={}
            self.Subtrahend={}
            self.Sum_insstead_Minuend={}
            self.additional_term={}

            # process of differentiation 
            self.Minuend[0]=data[self.seasons[0]:]
            self.Subtrahend[0]=data[:-self.seasons[0]]              
            self.Difference[0]=self.Minuend[0]-self.Subtrahend[0]

            self.additional_term[0]=data[-self.seasons[0]] 
            for i in range(1,len(self.seasons)):
                self.Minuend[i]=self.Difference[i-1][self.seasons[i]:]
                self.Subtrahend[i]=self.Difference[i-1][:-self.seasons[i]]

                self.Difference[i]=self.Minuend[i]-self.Subtrahend[i]

                self.additional_term[i]=self.Difference[i-1][-self.seasons[i]]

            if return_diff:
                return self.Difference[len(self.seasons)-1]
           
    def transform(self,point):
        return self.fit_transform(np.append(self.data,point),return_diff=True)[-1]
        
                
    def inverse_fit_transform0(self):
        """
        Return inital data for check
        """
        self.Sum_insstead_Minuend[len(self.seasons)]=self.Difference[len(self.seasons)-1]
        j=0
        for i in range(len(self.seasons)-1,-1,-1):
            self.Sum_insstead_Minuend[i]=self.Sum_insstead_Minuend[i+1]+self.Subtrahend[i][sum(self.seasons[::-1][:j]):]
            j+=1
        return self.Sum_insstead_Minuend[0]

    def inverse_transform(self,new_value):
        """
        may return only 1 element

        new_value = np.array !!!
        """
        self.new_value=new_value
        self.Sum_insstead_Minuend[len(self.seasons)]=self.new_value
        for i in range(len(self.seasons)-1,-1,-1):
            self.Sum_insstead_Minuend[i]= self.Sum_insstead_Minuend[i+1]+ self.additional_term[i]      

        new_value1=float(self.Sum_insstead_Minuend[0])
        # для того чтобы не выполнять регулярное fit_transform исполним хитрость тут
        self.fit_transform(np.append(self.data,new_value1),return_diff=False)
        return new_value1


