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

class find_best_model:
    def __init__(self,data,max_ar,max_ma,verbose=True,find_online_ar=True,criterion='mae'):

        if type(max_ar) == int:
            # издержки питона
            max_ar+=1
            ar_=range(1,max_ar)
        else:
            ar_=max_ar
        
        if type(max_ma) == int:
            max_ma+=1
            ma_=range(0,max_ma)
        else:
            ma_=max_ma

        table_aic=pd.DataFrame(index=ar_,columns=ma_)
        table_aic.index.name='AIC AR\MA'
        table_bic=table_aic.copy()
        table_bic.index.name='BIC AR\MA'
        table_mae=table_aic.copy()
        table_mae.index.name='MAE AR\MA'

        for ar in ar_:
            for ma in ma_:
                if ma > ar:
                    continue

                arma = sm.tsa.SARIMAX(endog=data, order=(ar,0,ma))
#                 arma = ARMA(endog=data, order=(ar,ma))
                try:
                    results=arma.fit()
                except:
                    if verbose:
                        print('not solve for model ',ar,ma)
                    continue
                table_aic.loc[ar][ma]=results.aic
                table_bic.loc[ar][ma]=results.bic
                table_mae.loc[ar][ma]=np.mean(np.abs(results.resid))
                if verbose:
                    print(ar,ma)
                del arma, results

        ar_aic=table_aic[table_aic==np.nanmin(table_aic.values)].dropna(axis=1,how='all').dropna(how='all').index.item()
        ma_aic=table_aic[table_aic==np.nanmin(table_aic.values)].dropna(axis=1,how='all').dropna(how='all').columns.item()

        ar_bic=table_bic[table_bic==np.nanmin(table_bic.values)].dropna(axis=1,how='all').dropna(how='all').index.item()
        ma_bic=table_bic[table_bic==np.nanmin(table_bic.values)].dropna(axis=1,how='all').dropna(how='all').columns.item()

        ar_mae=table_mae[table_mae==np.nanmin(table_mae.values)].dropna(axis=1,how='all').dropna(how='all').index.item()
        ma_mae=table_mae[table_mae==np.nanmin(table_mae.values)].dropna(axis=1,how='all').dropna(how='all').columns.item()

        if verbose:
            print('\r\n')
            print(table_aic)
            print('the best model for aic (AR/MA) is:',ar_aic,ma_aic)

            print('\r\n')
            print(table_bic)
            print('the best model for aic (AR/MA) is:',ar_bic,ma_bic)


            print('\r\n')
            print(table_mae)
            print('the best model for aic (AR/MA) is:',ar_mae,ma_mae)

        # назначение лучшей модели
        if criterion == 'mae':
            self.best_model=(ar_mae,ma_mae)
        if criterion == 'aic':
            self.best_model=(ar_aic,ma_aic)        
        if criterion == 'bic':
            self.best_model=(ar_bic,ma_bic)
            
        print('WE CHOOSE THE BEST MODEL IS:',self.best_model[0],self.best_model[1])

            # приблизительный метод из статьи на замене q на m
#             number=1/het_arch(np.nanmin(table_mae.values))[0]
#             base=1/(ma_mae)
#             m = math.log(number, base)

        # tatsmodels.tsa.arima_process.arma2ma(ar, ma, lags=100, **kwargs)[source]
        # эвристичечкий метод имени Славы
        if find_online_ar:
            err=np.nanmin(np.nanmin(table_mae.values))
            for i in range(20):
                ar= AR(data)
                res=ar.fit(ar_mae+i)
                if np.mean(np.abs(res.resid)) < err:
                    break
            self.best_model_ar=i+ar_mae
            print('WE CHOOSE THE BEST ONLINE AR MODEL IS:',self.best_model_ar)
