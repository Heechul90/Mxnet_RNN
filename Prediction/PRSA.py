###
import pandas as pd
import numpy as np
import mxnet as mx

### Dataset description:
# No:     row number                    # 넘버
# year:   year of data in this row      # 년도
# month:  month of data in this row     # 월
# day:    day of data in this row       # 일
# hour:   hour of data in this row      # 시간
# pm2.5:  PM2.5 concentration           #
# DEWP:   Dew Point                     # 이슬점
# TEMP:   Temperature                   # 온도
# PRES:   Pressure                      # 압력
# cbwd:   Combined wind direction       #
# Iws:    Cumulated wind speed          #
# Is:     Cumulated hours of snow       #
# Ir:     Cumulated hours of rain       #


### Load and pre-process th data
data = pd.read_csv('dataset/pollution/PRSA_data_2010.1.1-2014.12.31.csv',
                   sep=',')
data
