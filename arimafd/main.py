import numpy as np
from numpy import linalg
import pandas as pd
from sympy import diff, symbols, sympify, Symbol, poly
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
from .diff_integ import diff_integ
from .find_best_model import find_best_model
from .tanh import Anomaly_detection

class Arima_anomaly_detection(Anomaly_detection):
        """
        Thic class for anomaly detection application of modernized ARIMA model
        
        Returns
        -------
        self : object
        
        Examples
        --------
        >>> import arimafd as oa
        >>> my_array=pd.DataFrame([1,2,3,4,5])
        >>> ad = Anomaly_detection(my_array)
        >>> ad.generate_tensor()
        >>> ad.proc_tensor()
        >>> ad.evaluate_nab([[1,3]])
        """
        
        def __init__(self,ar_order=None):
            """
            
            Parameters
            ----------
            ar_order : float (default=None)
                Order of auoregression
            """
            super().__init__(pd.Series([1]))
            self.ar_order = ar_order
            
            
        def fit(self, history_dataset,
                window=100,
                No_metric=1,
                window_insensitivity=100):
                
                """
                Fit ARIMA Anomaly detection
                
                
                
                 Parameters
                ----------
                
                Parameters
                ----------
                data: pd.Series or pd.DataFrame
                    The researched time series or sequences array. 
                    Desire: dataset without anomalies for computing 
                    appropriate weights.  
                    
                window : int (default=100)
                    Time window for calculating metric.
                    It will be better if it is equal to 
                    half the width of the physical process.
                    
                window_insensitivity : int (default=100)
                    Аfter the new detected changepoint,
                    the following 'window_insensitivity' points 
                    is guaranteed not to be changepoints.
                    
                    

                Returns
                -------
                bin_metric: pandas array, shape (n_samples), float
                    Labeled pandas series, where value 1 is the anomaly,
                    0 is not the anomaly.

                
                Attributes
                -------
                self.metric: pandas array, shape (n_samples), float
                    calculated metric for data
                self.ucl: float, upper control limit for self.metric
                self.lcl: float, lower control limit for self.metric
                self.bin_metric: pandas array, shape (n_samples), float
                    Labeled pandas series, where value 1 is the anomaly,
                    0 is not the anomaly.               
                """
                
            if max([window,window_insensitivity]) >= len(history_dataset):
                print("Width of window is grater then len(data), Use batch")
            self.data = history_dataset
            self.indices = history_dataset.index
            self.generate_tensor(self.ar_order)
            bin = self.proc_tensor(window=window,
                                   No_metric=No_metric,
                                   window_insensitivity=window_insensitivity)
            return bin
        
        def predict(self,data,
                        window=100,
                        No_metric=1,
                        window_insensitivity=100):
                        
            if max([window,window_insensitivity]) >= len(data):
                print("Width of window is grater then len(data), Use batch")
            
            self.data = data
            self.indices = data.index
            
            data=self.ss.transform(data.copy())

            tensor = np.zeros((data.shape[0],
                               data.shape[1],
                               self.ar_order+1))
            j=0
            for i in range(data.shape[1]):
                for value in data[:,i]:
                    new_val = self.diffrs[i].transform(value)
                    self.models[i].predict(new_val)

                tensor[:,i,:] = self.models[i].dif_w.values[-len(data[:,i]):]
            self.tensor = tensor
            
            bin = self.proc_tensor(window=window,
                                   No_metric=No_metric,
                                   window_insensitivity=window_insensitivity)
            return bin
        
            