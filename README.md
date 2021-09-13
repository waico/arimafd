[![Downloads](https://pepy.tech/badge/arimafd)](https://pepy.tech/project/arimafd) [![Downloads](https://pepy.tech/badge/arimafd/month)](https://pepy.tech/project/arimafd) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/waico/arimafd/blob/master/LICENSE.txt) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/online-forecasting-and-anomaly-detection/anomaly-detection-on-numenta-anomaly)](https://paperswithcode.com/sota/anomaly-detection-on-numenta-anomaly?p=online-forecasting-and-anomaly-detection)

# About arimafd

Arimafd is a Python package that provides algorithms
for online prediction and anomaly detection. One of the
applications of this package can be the early detection
of faults in technical systems.



# Main Features


- Differentiation and integration of series including seasonal components
- Finding best hyperparametrs for ARIMA model
- Online forecasting based on ARIMA model
- Anomaly detection 
- Evaluating score of anomaly detection algorithms

# How to get it
The master branch on GitHub 

https://github.com/waico/arimafd


Binaries and source distributions are available from PyPi

https://pypi.org/project/arimafd/

# Get started

**Installation** through [PyPi](https://pypi.org/project/tsad): 

`pip install -U arimafd`

1. Example

```python
import pandas as pd
import numpy as np
import arimafd as oa

my_array = np.random.normal(size=1000) # init array
my_array[-3] = 1000 # init anomaly
ts = pd.DataFrame(my_array,
                  index=pd.date_range(start='01-01-2000',
                                      periods=1000,
                                      freq='H'))
ad = oa.anomaly_detection(ts) #init anomaly detection algorithm
ad.generate_tensor(ar_order=3) #it compute weights of ARIMA on history 
ts_anomaly = ad.proc_tensor() #processing of weights. 
# ad.ebeluate_nab() # function for evaluating results of algorithms
ts_anomaly
```

Actually, labeling time series on anomaly and not an anomaly have already been performed by proc_tensor function (it returns labeled time series, where 1 is an anomaly, 0 - not anomaly). 

[Output]:

```python
2000-01-01 03:00:00    0
2000-01-01 04:00:00    0
2000-01-01 05:00:00    0
2000-01-01 06:00:00    0
2000-01-01 07:00:00    0
                      ..
2000-02-11 11:00:00    0
2000-02-11 12:00:00    0
2000-02-11 13:00:00    0
2000-02-11 14:00:00    1
2000-02-11 15:00:00    1
Freq: H, Length: 997, dtype: int32
```



# License

MIT
