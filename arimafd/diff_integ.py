import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AR
import statsmodels.api as sm
from time import time


class diff_integ:
    def __init__(self,seasons):
        """
        Differentiation and Integration Module

        This class is needed to bring series to stationarity 
        and perform inverse operation.


        Parameters
        ----------
        Seasons : list of int
            List of lags for differentiation. 


        Returns
        -------
        self : object


        Exampless
        --------
        >>> import arimafd as oa
        >>> dif = oa.diff_integ([1])
        >>> my_array=[1,2,3,4,5]
        >>> dif.fit_transform(my_array)
        array([1, 1, 1, 1])
        >>> dif.transform(6)
        1
        >>> dif.inverse_transform(1)
        7.0
        """
        
        #Comments: 1) The algorithm is not optimized, in terms of adding new element in dictionary instead adding new dictionary
        self.seasons=seasons
       
    

            
    def fit_transform(self,data,return_diff=True):
        """
        Fit the model and transform data according to the given training data.

        Parameters
        ----------
        data : array-like, shape (n_samples,)
            Training data, where n_samples is the number of samples

        return_diff, optional (default=True)
            Returns the differentiated array 

        Returns
        -------
        If return_diff = True: data_new : array-like, shape (n_samples - sum_seasons,) 
            where sum_seasons is sum of all lags
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
        """
        Differentiation to the series data that were 
        in method fit_transform and plus all the points that 
        were in this method.
        
        Parameters
        ----------
        point : float
            Add new point to self.data
        
        Returns
        -------
        Array-like, shape (n_samples + n*n_points - sum_seasons,) 
        """
        return self.fit_transform(np.append(self.data,point),return_diff=True)[-1]
        
                
    def inverse_fit_transform0(self):
        """
        Return inital data for check class
        """
        self.Sum_insstead_Minuend[len(self.seasons)]=self.Difference[len(self.seasons)-1]
        j=0
        for i in range(len(self.seasons)-1,-1,-1):
            self.Sum_insstead_Minuend[i]=self.Sum_insstead_Minuend[i+1]+self.Subtrahend[i][sum(self.seasons[::-1][:j]):]
            j+=1
        return self.Sum_insstead_Minuend[0]

    def inverse_transform(self,new_value):
        """
        Return last element after integration. 
        (Forecasting value in initial dimension)
        
        Parameters
        ----------
        new_value : float
            New value in differentiated series
        
        Returns
        -------
        Integrated value, float
       """
        self.new_value=new_value
        self.Sum_insstead_Minuend[len(self.seasons)]=self.new_value
        for i in range(len(self.seasons)-1,-1,-1):
            self.Sum_insstead_Minuend[i]= self.Sum_insstead_Minuend[i+1]+ self.additional_term[i]      

        new_value1=float(self.Sum_insstead_Minuend[0])
        # для того чтобы не выполнять регулярное fit_transform исполним хитрость тут
        self.fit_transform(np.append(self.data,new_value1),return_diff=False)
        return new_value1


