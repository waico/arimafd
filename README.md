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
- The master branch on GitHub:  
https://github.com/waico/arimafd

- Binaries and source distributions are available from PyPi:  
https://pypi.org/project/arimafd/

- **Installation** through [PyPi](https://pypi.org/project/arimafd):  
`pip install -U arimafd`

# Get started

### Example #1

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

my_arima = oa.Arima_anomaly_detection(ar_order=3)
my_arima.fit(ts[:500])
ts_anomaly = my_arima.predict(ts[500:])


# or you can use for streaming:
# bin_metric = []
# for i in range(len(df)):
#     bin_metric.append(my_arima.predict(df[i:i+1]))
# bin_metric = pd.concat(bin_metric)
# bin_metric
ts_anomaly
```

[Output]:

```python
2000-01-21 20:00:00    0
2000-01-21 21:00:00    0
2000-01-21 22:00:00    0
2000-01-21 23:00:00    0
2000-01-22 00:00:00    0
                      ..
2000-02-11 11:00:00    0
2000-02-11 12:00:00    0
2000-02-11 13:00:00    0
2000-02-11 14:00:00    1
2000-02-11 15:00:00    1
Freq: H, Length: 997, dtype: int32
```

Actually, labeling time series on anomaly and not an anomaly have already been performed by proc_tensor function (it returns labeled time series, where 1 is an anomaly, 0 - not anomaly). 

For evaluating results you can use https://tsad.readthedocs.io/en/latest/Evaluating.html 

### Example #2

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
ad = oa.Anomaly_detection(ts) #init anomaly detection algorithm
ad.generate_tensor(ar_order=3) #it compute weights of ARIMA on history 
ts_anomaly = ad.proc_tensor() #processing of weights. 
ts_anomaly
```

# License

MIT
