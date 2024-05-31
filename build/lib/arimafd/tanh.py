import numpy as np
from numpy import linalg
import pandas as pd
from sympy import diff, symbols, sympify, Symbol, poly
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
from .diff_integ import diff_integ
from .find_best_model import find_best_model

#=============================================================
def projection(w,circle=1.01):
    """
    Function for projection weights
    
    Parameters
    ----------
    w : array-like, shape (n_weights,)
        List of initial weights, where n_weights is the number of weights 

    Returns
    -------
    new_w : array-like, shape (n_weights,)
        List of weights resolved solution area,
        where n_weights is the number of weights.

    """
    w=w[::-1] # due to using in function body
    # find coeff of poly if we have roots 
    def c_find(roots):
        x = Symbol('x')
        whole =1
        for root in roots:
            whole *=(x-root)
    #     print('f(x) =',whole.expand())
        p = poly(whole, x)
        return np.array(p.all_coeffs()).astype(float)
    
    roots = np.roots(w)
    l1 = linalg.norm(roots)
    #print(l1)

    if l1 < circle:
        print('Projection')
        scale = circle/l1
        new_roots = roots*scale
        new_w=c_find(new_roots)[::-1]
        return new_w
    else:
        return w[::-1]
 #================================================================

          
class online_tanh:
    def __init__(self, order=4, lrate=0.001, random_state=42, soft_grad=False,project=True):
        """
        A class for online arima with stochastic gradient
        descent and log-cosh loss function

        Parameters
        ----------
        order : array-like, shape (default=4)
            Order of autoregression
            
        lrate : float
            Value of gradint descent rate
        
        random_state : int, (default=42)
            Random_state is the random number generator

        soft_grad, optional (default=False)
            Rate of gradient descent is redused with every iteration
            
        project, optional (default=True)
            If True, make projection on resolved solution

        Returns
        -------
        self : object

        Examples
        --------
        >>> import arimafd as oa
        >>> pr = oa.online_tanh()
        >>> my_array=[1,2,3,4,5]
        >>> pr.fit(my_array)
        >>> pr.predict(predict_size=4)
        array([0.13778082, 0.11579774, 0.08165416, 0.0298824 ])
        """
        self.soft_grad = soft_grad
        self.order=order
        self.lrate=lrate
        self.random_state=random_state
        self.project = project
        
        if soft_grad:
            def fun_w(i):
                return 1/ np.sqrt(i+1)  #намерено опустил  член -order, из-за небольшой погрешности допущения
        else:
            def fun_w(i):
                return 1
        self.fun_w = fun_w
    
    def fit(self, data, init_w=None):
        """
        Fit the AR model according to the given historical data. 
        It will be better if data represent normal operation mode
        
        
        Parameters
        ----------
        data : array-like, shape (n_samples,)
            Training data, where n_samples is the number of samples

        init_w : array-like, shape (n_weight,), (default=None)
            Initial array of weights, where n_weight is the number of weights
            If None the weights are initialized randomly

        Returns
        -------
        self : object
        """
        
        data=np.array(data)
        self.data=data
        np.random.seed(self.random_state)
        self.pred = np.zeros(data.shape[0] + 1)*np.nan
        self.w = np.random.rand(self.order+1)*0.01 if init_w is None else init_w.copy()
        self.ww=pd.DataFrame([self.w])
        self.diff=np.zeros(len(self.w))
        # create pandas diffrent of w 
        self.dif_w = pd.DataFrame([self.w])
        for i in range(self.order, data.shape[0]):
            self.pred[i] = self.w[:-1] @ data[i-self.order:i] + self.w[-1]          
            self.diff[:-1]= np.tanh(self.pred[i] - data[i])*data[i-self.order:i]
            self.diff[-1] = np.tanh(self.pred[i] - data[i])# свободный член
            self.w -= self.lrate * self.diff * self.fun_w(i)
            
            if self.project:
                self.w = projection(self.w)
            self.ww = pd.concat([self.ww, pd.DataFrame([self.w])], ignore_index=True)
            self.dif_w = pd.concat([self.dif_w, pd.DataFrame([self.diff])], ignore_index=True)
        self.iii=i
        # реальные предсказания 
        # это нужно для дальнейшей работы алгоритма: 1 точка
        self.pred[-1]=self.w[:-1] @ data[-self.order:] + self.w[-1]                
    
    def predict(self, point_get=None, predict_size=1,return_predict=True):
        """
        Make forecasting series from data to predict_size points
       
        Parameters
        ----------
        point_get : float (default=None)
            Add new for next iteration of stochastic gradiend descent
        
        predict_size: float
            The number of out of sample forecasts from the end of the sample
            
        return_predict, optional (default=True)
            Returns array of forecasting values

        Returns
        -------
        If return_diff = True: data_new : array-like, shape (n_samples - sum_seasons,) 
            where sum_seasons is sum of all lags
        
        self : object           
        """
        # часть отвечающая за онлайн
        if point_get is not None:
            self.data=np.append(self.data,point_get)            
            self.diff[:-1]= np.tanh(self.pred[-1] - self.data[-1])*self.data[-self.order-1:-1]
            self.diff[-1] = np.tanh(self.pred[-1] - self.data[-1])# свободный член
            self.w -= self.lrate * self.diff * self.fun_w(self.iii)
            if self.project:
                self.w = projection(self.w)
            
            self.ww = pd.concat([self.ww, pd.DataFrame([self.w])], ignore_index=True)
            
            self.pred=np.append(self.pred,np.nan)
            self.dif_w = pd.concat([self.dif_w, pd.DataFrame([self.diff])], ignore_index=True)
            self.pred[-1]=self.w[:-1] @ self.data[-self.order:] + self.w[-1]

            
                    
        if predict_size > 1:
            data_p=np.append(self.data[-self.order:],np.zeros(predict_size)*np.nan)
            
            for i in range(self.order,self.order+predict_size):
                data_p[i]=self.w[:-1] @ data_p[i-self.order:i] + self.w[-1]
            if return_predict:
                return data_p[self.order:]
        elif predict_size==1 and return_predict:
            return self.pred[-1]



class Anomaly_detection:
    """
    This class for anomaly detection application of modernized ARIMA model
    
    Examples:
    ----------
   
    >>> import pandas as pd
    >>> import numpy as np
    >>> import arimafd as oa
    >>> my_array = np.random.normal(size=1000) # init array
    >>> my_array[-3] = 1000 # init anomaly
    >>> ts = pd.DataFrame(my_array,
    >>>                   index=pd.date_range(start='01-01-2000',
    >>>                                       periods=1000,
    >>>                                       freq='H'))
    >>> ad = oa.anomaly_detection(ts) #init anomaly detection algorithm
    >>> ad.generate_tensor(ar_order=3) #it compute weights of ARIMA on history 
    >>> ts_anomaly = ad.proc_tensor() #processing of weights. 
    >>> ts_anomaly
    """

    def __init__(self,data):
        """
       
        Parameters
        ----------
        data: pandas array, float
            The reseached array 
        
        Returns
        -------
        self : object
        
        """
        self.indices = data.index
        self.data = data
        



    def generate_tensor(self,ar_order=None,verbose=True):
        """
        Generation tensor of weights for outlier detection

        Parameters
        ----------
        ar_order : float (default=None)
            Order of auoregression

        Returns
        -------
        tensor : array-like, shape (n_samples,n_features,ar_order)
            Tensor of weights (see higher for scale) where 
            n_samples - number of samples
            n_features - number of features
            ar_order - number of weights
        
        Attributes
        -------
        self.tensor: array-like, shape (n_samples,n_features,ar_order)
            Tensor of weights (see higher for scale) where 
            n_samples - number of samples
            n_features - number of features
            ar_order - number of weights
        """
        data = self.data.copy()


        if ar_order is None:
            ar_order=int(len(data)/5)
        self.ar_order = ar_order

        self.ss = StandardScaler()
        #mms = MinMaxScaler()
        
        data=self.ss.fit_transform(data.copy())

        tensor = np.zeros((data.shape[0]-ar_order,data.shape[1],ar_order+1))
        j=0
        self.models = []
        self.diffrs = []
        for i in range(data.shape[1]):
            t1=time()
            kkk=0

            diffr=diff_integ([1])
            dif = diffr.fit_transform(data[:,i])
            self.diffrs.append(diffr)
            
            model=online_tanh(ar_order)
            model.fit(dif)
            
            self.models.append(model)
            t2=time()
            if verbose:
                print('Time seconds:', t2-t1)

            tensor[:,i,:] = model.dif_w.values
        self.tensor = tensor
        return tensor

    def proc_tensor(self,window=100,No_metric=1,window_insensitivity=100):
        """
        Processing tensor of weights and calcute metric
        and binary labels  

        Parameters
        ----------
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
        
        tensor = self.tensor.copy()
        df = pd.DataFrame(tensor.reshape(len(tensor),-1),index=self.indices[-len(tensor):])
        # mean and min in end is the euthrestic 
#         if len(self.data.iloc[:,i].unique())/len(self.data.iloc[:,i]) > 0.99:
        if No_metric == 1:
            metric = (df.rolling(window).max().abs()/df.rolling(window).std().abs()).mean(axis=1)
#             print('Metric1')
        elif No_metric == 2:
            metric = df.abs().max(axis=1)
        elif No_metric == 3:
            metric = np.sqrt(np.square(df.diff(1)).sum(axis=1))
            metric[0] = np.nan
        elif No_metric == 4:
            member1 = df.abs().rolling(window).max().drop(df.index[-1],axis=0).values
            member2 = df.abs().drop(df.index[0],axis=0).values
            metric = np.abs(member2 - member1)
            sc = StandardScaler()
            metric = sc.fit_transform(metric).max(axis=1)
            metric = pd.Series(np.append(np.nan*np.zeros(1), metric),index=df.index)
        elif No_metric == 5:
            metric = df.sum(axis=1)*np.nan
            for i in range(window,len(df)):
                try:
                    cov = np.linalg.inv(np.cov(df[i-window:i].T))
                except:
                    cov = np.linalg.pinv(np.cov(df[i-window:i].T))
                mean = np.array(df[i:i+window].mean(axis=0)).reshape(1,-1)
                x = np.array(df.iloc[i]).reshape(1,-1)
                metric.iloc[i]=np.dot(np.dot((x-mean),cov),(x-mean).T)
        elif No_metric == 6:
            koef = 3
            metric = df.rolling(window).apply(lambda x: int(  np.abs( (x.values[-1]-np.mean(x))/np.std(x) > koef*np.std(x) ))).sum(axis=1)
        
        
            
#             print('Metric2')
        ucl = metric.mean() + 3*metric.std()
        lcl = metric.mean() - 3*metric.std()
        self.metric = metric
        self.ucl = ucl
        self.lcl = lcl            
        bin_metric = ((metric > ucl) | (metric < lcl)).astype(int)
        # filtering atreh him, after 1 all 100 digits will be zero 
        winn = window_insensitivity
        for i in range(len(bin_metric)-winn):
            if ((bin_metric.iloc[i] == 1.0) & (bin_metric[i:i+winn].sum()>1.0)):
                bin_metric[i+1:i+winn]=np.zeros(winn-1)
        self.bin_metric = bin_metric        
        return bin_metric


    
