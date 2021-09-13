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
            self.ww=self.ww.append([self.w], ignore_index=True)
            self.dif_w = self.dif_w.append([self.diff], ignore_index=True)
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
            
            self.ww=self.ww.append([self.w], ignore_index=True)
            
            self.pred=np.append(self.pred,np.nan)
            self.dif_w = self.dif_w.append([self.diff], ignore_index=True)
            self.pred[-1]=self.w[:-1] @ self.data[-self.order:] + self.w[-1]

            
                    
        if predict_size > 1:
            data_p=np.append(self.data[-self.order:],np.zeros(predict_size)*np.nan)
            
            for i in range(self.order,self.order+predict_size):
                data_p[i]=self.w[:-1] @ data_p[i-self.order:i] + self.w[-1]
            if return_predict:
                return data_p[self.order:]
        elif predict_size==1 and return_predict:
            return self.pred[-1]



class anomaly_detection:

    def __init__(self,data):
        """
        Thic class for anomaly detection aplication of modernized ARIMA model
        
        Parameters
        ----------
        data: pandas array, float
            The reseached array 
        
        Returns
        -------
        self : object
        
        Examples
        --------
        >>> import arimafd as oa
        >>> my_array=pd.DataFrame([1,2,3,4,5])
        >>> ad = oa.anomaly_detection(my_array)
        >>> ad.generate_tensor()
        >>> ad.proc_tensor()
        >>> ad.evaluate_nab([[1,3]])
        """
        self.indices = data.index
        self.data = data
        



    def generate_tensor(self,ar_order=None):
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

        ss = StandardScaler()
        mms = MinMaxScaler()

        data=ss.fit_transform(data.copy())

        tensor = np.zeros((data.shape[0]-ar_order,data.shape[1],ar_order+1))
        j=0
        for i in range(data.shape[1]):
            t1=time()
            kkk=0

            diffr=diff_integ([1])
            dif = diffr.fit_transform(data[:,i])

            model=online_tanh(ar_order)
            model.fit(dif)
            t2=time()
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


    def evaluate_nab(self,anomaly_list,table_of_coef=None):
        """
        Scoring labeled time series by means of
        Numenta Anomaly Benchmark methodics

        Parameters
        ----------
        anomaly_list: list of list of two float values
            The list of lists of left and right boundary indices
            for scoring results of labeling
        table_of_coef: pandas array (3x4) of float values
            Table of coefficients for NAB score function
            indeces: 'Standart','LowFP','LowFN'
            columns:'A_tp','A_fp','A_tn','A_fn'
            

        Returns
        -------
        Scores: numpy array, shape of 3, float
            Score for 'Standart','LowFP','LowFN' profile 
        Scores_null: numpy array, shape 3, float
            Null score for 'Standart','LowFP','LowFN' profile             
        Scores_perfect: numpy array, shape 3, float
            Perfect Score for 'Standart','LowFP','LowFN' profile  
        """
        if table_of_coef is None:
            table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                 [1.0,-0.22,1.0,-1.0],
                                  [1.0,-0.11,1.0,-2.0]])
            table_of_coef.index = ['Standart','LowFP','LowFN']
            table_of_coef.index.name = "Metric"
            table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']

        alist = anomaly_list.copy()
        bin_metric = self.bin_metric.copy()
        
#         bin_metric = bin_metric.reset_index().drop_duplicates().set_index(bin_metric.index.name)
        Scores,Scores_perfect,Scores_null=[],[],[]
        for profile in ['Standart','LowFP','LowFN']:       
            A_tp = table_of_coef['A_tp'][profile]
            A_fp = table_of_coef['A_fp'][profile]
            A_fn = table_of_coef['A_fn'][profile]
            #TODO make 10% window if not known boundary
            #if len(list(al.values())[0])
            def sigm_scale(y,A_tp,A_fp,window=1):
                return (A_tp-A_fp)*(1/(1+np.exp(5*y/window))) + A_fp

            #First part
            score = 0
            if len(alist)>0:
                score += bin_metric[:alist[0][0]].sum()*A_fp
            else:
                score += bin_metric.sum()*A_fp
            #second part
            for i in range(len(alist)):
                if i<=len(alist)-2:
                    win_space = bin_metric[alist[i][0]:alist[i+1][0]].copy()
                else:
                    win_space = bin_metric[alist[i][0]:].copy()
                win_fault = bin_metric[alist[i][0]:alist[i][1]]
                slow_width = int(len(win_fault)/4)
                
                if len(win_fault) + slow_width >= len(win_space):
#                    не совсем так правильно лелать
                    print('большая ширина плавного переходы сигмойды')
                    win_fault_slow = win_fault.copy()
                else:
                    win_fault_slow= win_space[:len(win_fault)  +  slow_width]
                
                win_fp = win_space[-len(win_fault_slow):]
                
                if win_fault_slow.sum() == 0:
                    score+=A_fn
                else:
                    #берем первый индекс
                    tr = pd.Series(win_fault_slow.values,index = range(-len(win_fault),len(win_fault_slow)-len(win_fault)))
                    tr_values= tr[tr==1].index[0]
                    tr_score = sigm_scale(tr_values, A_tp,A_fp,slow_width)
                    score += tr_score
                    score += win_fp.sum()*A_fp
            Scores.append(score)
            Scores_perfect.append(len(alist)*A_tp)
            Scores_null.append(len(alist)*A_fn)
        self.Scores,self.Scores_null,self.Scores_perfect = np.array(Scores), np.array(Scores_null) ,np.array(Scores_perfect)
        return np.array(Scores), np.array(Scores_null) ,np.array(Scores_perfect)


    def evaluate_nab(self,anomaly_list,table_of_coef=None):
        """
        Scoring labeled time series by means of
        Numenta Anomaly Benchmark methodics

        Parameters
        ----------
        anomaly_list: list of list of two float values
            The list of lists of left and right boundary indices
            for scoring results of labeling
        table_of_coef: pandas array (3x4) of float values
            Table of coefficients for NAB score function
            indeces: 'Standart','LowFP','LowFN'
            columns:'A_tp','A_fp','A_tn','A_fn'
            

        Returns
        -------
        Scores: numpy array, shape of 3, float
            Score for 'Standart','LowFP','LowFN' profile 
        Scores_null: numpy array, shape 3, float
            Null score for 'Standart','LowFP','LowFN' profile             
        Scores_perfect: numpy array, shape 3, float
            Perfect Score for 'Standart','LowFP','LowFN' profile  
        """
        if table_of_coef is None:
            table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                 [1.0,-0.22,1.0,-1.0],
                                  [1.0,-0.11,1.0,-2.0]])
            table_of_coef.index = ['Standart','LowFP','LowFN']
            table_of_coef.index.name = "Metric"
            table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']

        alist = anomaly_list.copy()
        bin_metric = self.bin_metric.copy()
        
#         bin_metric = bin_metric.reset_index().drop_duplicates().set_index(bin_metric.index.name)
        Scores,Scores_perfect,Scores_null=[],[],[]
        for profile in ['Standart','LowFP','LowFN']:       
            A_tp = table_of_coef['A_tp'][profile]
            A_fp = table_of_coef['A_fp'][profile]
            A_fn = table_of_coef['A_fn'][profile]
            #TODO make 10% window if not known boundary
            #if len(list(al.values())[0])
            def sigm_scale(y,A_tp,A_fp,window=1):
                return (A_tp-A_fp)*(1/(1+np.exp(5*y/window))) + A_fp

            #First part
            score = 0
            if len(alist)>0:
                score += bin_metric[:alist[0][0]].sum()*A_fp
            else:
                score += bin_metric.sum()*A_fp
            #second part
            for i in range(len(alist)):
                if i<=len(alist)-2:
                    win_space = bin_metric[alist[i][0]:alist[i+1][0]].copy()
                else:
                    win_space = bin_metric[alist[i][0]:].copy()
                win_fault = bin_metric[alist[i][0]:alist[i][1]]
                slow_width = int(len(win_fault)/4)
                
                if len(win_fault) + slow_width >= len(win_space):
#                    не совсем так правильно лелать
                    print('большая ширина плавного переходы сигмойды')
                    win_fault_slow = win_fault.copy()
                else:
                    win_fault_slow= win_space[:len(win_fault)  +  slow_width]
                
                win_fp = win_space[-len(win_fault_slow):]
                
                if win_fault_slow.sum() == 0:
                    score+=A_fn
                else:
                    #берем первый индекс
                    tr = pd.Series(win_fault_slow.values,index = range(-len(win_fault),len(win_fault_slow)-len(win_fault)))
                    tr_values= tr[tr==1].index[0]
                    tr_score = sigm_scale(tr_values, A_tp,A_fp,slow_width)
                    score += tr_score
                    score += win_fp.sum()*A_fp
            Scores.append(score)
            Scores_perfect.append(len(alist)*A_tp)
            Scores_null.append(len(alist)*A_fn)
        self.Scores,self.Scores_null,self.Scores_perfect = np.array(Scores), np.array(Scores_null) ,np.array(Scores_perfect)
        return np.array(Scores), np.array(Scores_null) ,np.array(Scores_perfect)
                
        
def get_score(list_metrics):
    """
    Get full score for algorithm
    from several datasets
    """
    sum1 = np.zeros((3,3))
    for i in range(len(list_metrics)):
        sum1 += list_metrics[i]
    desc = ['Standart','LowFP','LowFN']    
    for t in range(3):
        print(desc[t],' - ', 100*(sum1[0,t]-sum1[1,t])/(sum1[2,t]-sum1[1,t]))
