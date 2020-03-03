##### Time Series Anomaly Detection with LSTM and MXNet
### 함수, 모듈 불러오기
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import mxnet as mx
from mxnet import nd, autograd, gluon

from sklearn import preprocessing
from sklearn.metrics import f1_score


# 데이터 불러오기

df1 = pd.read_csv('dataset/통신구/crack1_변형.csv')
df2 = pd.read_csv('dataset/통신구/crack2_변형.csv')
df3 = pd.read_csv('dataset/통신구/strain1.csv')
df4 = pd.read_csv('dataset/통신구/strain2.csv')
df5 = pd.read_csv('dataset/통신구/strain3.csv')
df3.plot()
df4.plot()
df5.plot()


len(df1), len(df2)

df2 = df2[:len(df1)]
len(df1), len(df2)


df1['time'] = pd.to_datetime(df1['time'])
df1.head()
df1.tail()


df2['time'] = df1['time']
df2.head()


def convert_timestamps(data_frame):
    data_frame['time_epoch'] = data_frame['time'].astype(np.int64)

convert_timestamps(df1)
convert_timestamps(df2)

def labeling(data_frame):
    data_frame['anomaly_label'] =  list(map(lambda x: 1 if x > 0.094 else 0, data_frame['value']))

labeling(df1)
len(df1[df1['anomaly_label'] == 1])

labeling(df2)
len(df2[df2['anomaly_label'] == 1])

# When plotting the data we mark anomalies with green dots:

def prepare_plot(data_frame):
    fig, ax = plt.subplots()
    ax.scatter(data_frame['time_epoch'], data_frame['value'], s=8, color='blue')  # scatter 산포그래프

    labled_anomalies = data_frame.loc[data_frame['anomaly_label'] == 1, ['time_epoch', 'value']]
    ax.scatter(labled_anomalies['time_epoch'], labled_anomalies['value'], s=100, color='green')

    return ax

figsize(16, 7)
prepare_plot(df1)
plt.show()



# The visualization of the training and test datasets look like this:
# visualization

figsize(16, 7)
prepare_plot(df2)
plt.show()


# Preparing a dataset
df1['value_no_anomaly'] = df1[df1['anomaly_label'] == 0]['value']

df1.loc[df1['anomaly_label'] == 1, ['value_no_anomaly']]


########################################################################################################################
# nan값 채우기

# 결측값을 특정 값으로 채우기:   df.fillna(0)
# 결측값을 앞 방향으로 채우기:   df.fillna(method: 'ffill' or 'pad')
# 결측값을 뒷 방향으로 채우기:   df.fillna(method: 'bill' or 'backfill')
# 결측값을 채우는 회수 제한하기: df.fillna(limit=1)
# 결측값을 평균값으로 채우기:    df.fillna(df.mean())
########################################################################################################################

df1['value_no_anomaly'] = df1['value_no_anomaly'].fillna(method='ffill') # method 앞 값으로 채우기

df1['value'] = df1['value_no_anomaly']
features = ['value']

feature_count = len(features)

########################################################################################################################
# scikit-learn에서는 다음과 같은 스케일링 클래스를 제공한다.
#
# StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
# RobustScaler(X):   중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
# MinMaxScaler(X):   최대값이 각각 1, 최소값이 0이 되도록 변환
# MaxAbsScaler(X):   0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환
########################################################################################################################

data_scaler = preprocessing.StandardScaler()
data_scaler.fit(df1[features].values.astype(np.float32))

training_data = data_scaler.transform(df1[features].values.astype(np.float32))

rows = len(training_data)

split_factor = 0.8

# 교육 및 검증 데이터 준비
training = training_data[0:int(rows * split_factor)]
validation = training_data[int(rows * split_factor):]
len(training)
len(validation)

### Choosing a Model(모델 정의)

########################################################################################################################
# gluon.nn.Sequential() : 순차적으로 블럭을 쌓는다
# model.add : 스택위로 블럭을 추가한다
# gluon.rnn.LSTM(n) : LSTM layer with n-output dimensionality. In our situation, we used an LSTM layer without dropout at the layer output. Commonly, dropout layers are used for preventing the overfitting of the model. It’s just zeroed the layer inputs with the given probability
# gluon.nn.Dense(n, activation=’tanh’) : densely-connected NN layer with n-output dimensionality and hyperbolic tangent activation function

########################################################################################################################
model = mx.gluon.nn.Sequential()

with model.name_scope():
    model.add(mx.gluon.rnn.LSTM(feature_count))
    model.add(mx.gluon.nn.Dense(feature_count, activation='tanh'))


### Training & Evaluation
# loss 함수 선택
L = gluon.loss.L2Loss() # L2 loss: (실제값 - 예측치)제곱해서 더한 값, L1 loss: (실제값 - 예측치)절대값해서 더한 값

# 평가
def evaluate_accuracy(data_iterator, model, L):
    loss_avg = 0.
    for i, data in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 1, feature_count))
        output = model(data)
        loss = L(output, data)
        loss_avg = (loss_avg * i + nd.mean(loss).asscalar()) / (i + 1)
    return loss_avg



# cpu or gpu
ctx = mx.cpu()

# batch_size가 작으면 학습시간이 늘어난다
# 여기서는 batch_size를 실험을 통해서 48이 적당하다는 것을 알아냄
# 데이터의 값은 5분마다 발생하고 48은 4시간과 같습니다
batch_size = 60

training_data_batches = mx.gluon.data.DataLoader(training, batch_size, shuffle=False)
validation_data_batches = mx.gluon.data.DataLoader(validation, batch_size, shuffle=False)

# Xavier는 모든 layers에서 gradient scale이 거의 동일하게 유지하도록 설계됨
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# 이 모델에는 sgd 최적화 함수, 학습률은 0.01이 가장 최적으로 보임
# sgd에 비해 너무 작지 않고, 최적화에 시간이 오래 걸리지 않으며 너무 크지 않는다
# 그래서 loss function의 최소값을 넘지 않는다
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})


### Let’s run the training loop and plot MSEs
epochs = 15
training_mse = []        # 평균 제곱 오차를 기록
validation_mse = []

for epoch in range(epochs):
    print(str(epoch+1))
    for i, data in enumerate(training_data_batches):
        data = data.as_in_context(ctx).reshape((-1, 1, feature_count))

        with autograd.record():
            output = model(data)
            loss = L(output, data)

        loss.backward()
        trainer.step(batch_size)

    training_mse.append(evaluate_accuracy(training_data_batches, model, L))
    validation_mse.append(evaluate_accuracy(validation_data_batches, model, L))



### prediction
# (input value, predicted output value)가 각각 쌍으로 autoencoder로 사용할 때
# reconstruction error를 가질 수 있다
# training set으로 reconstruction error를 구할 수 있다.
# reconstruction error가 전체 training set에서 값에서 멀리 벗어날 경우를 비정상이라고 본다
# 이 경우에는, 3-sigma approach를 사용하는 것이 좋다
# reconstruction error가 3-sigma deviation보다 높으면 비정상이다
def calculate_reconstruction_errors(input_data, L):
    reconstruction_errors = []
    for i, data in enumerate(input_data):
        input = data.as_in_context(ctx).reshape((-1, feature_count, 1))
        predicted_value = model(input)
        reconstruction_error = L(predicted_value, input).asnumpy().flatten()
        reconstruction_errors = np.append(reconstruction_errors, reconstruction_error)

    return reconstruction_errors


all_training_data = mx.gluon.data.DataLoader(training_data.astype(np.float32), batch_size, shuffle=False)

training_reconstruction_errors = calculate_reconstruction_errors(all_training_data, L)

# 3*sigma: 평균에서 3*표준편차를 더한 값
reconstruction_error_threshold = np.mean(training_reconstruction_errors) + 3 * np.std(training_reconstruction_errors)

test_data = data_scaler.fit_transform(df2[features].values.astype(np.float32))

test_data_batches = mx.gluon.data.DataLoader(test_data, batch_size, shuffle=False)

test_reconstruction_errors = calculate_reconstruction_errors(test_data_batches, L)



predicted_test_anomalies = list(map(lambda v: 1 if v > reconstruction_error_threshold else 0, test_reconstruction_errors))

df2['anomaly_predicted'] = predicted_test_anomalies


figsize(16, 7)

ax = prepare_plot(df2)

predicted_anomalies = df2.loc[df2['anomaly_predicted'] == 1, ['time_epoch', 'value']]
ax.scatter(predicted_anomalies['time_epoch'], predicted_anomalies['value'], s=50, color='red')

plt.show()


test_labels = df2['anomaly_label'].astype(np.float32)

score = f1_score(test_labels, predicted_test_anomalies)
print('F1 score: ' + str(score))