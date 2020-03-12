##### 시계열 데이터 이상탐지

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

# Let’s describe all paths to datasets and labels:
nab_path = './dataset/nab'
nab_data_path = nab_path + '/data/'

labels_filename = '/labels/combined_labels.json'
training_file_name = 'realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv'
test_file_name = 'realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv'

# Anomaly labels are stored separately from the data values. Let’s load the train and test datasets and label the values with pandas:
# 이미 만들어진 label을 불러옵니다
labels_file = open(nab_path + labels_filename, 'r')
labels = json.loads(labels_file.read())
labels_file.close()

# label을 trainset과 testset에 컬럼으로 만들어 줍니다
def load_data_frame_with_labels(file_name):
    data_frame = pd.read_csv(nab_data_path + file_name)
    data_frame['anomaly_label'] = data_frame['timestamp'].isin(labels[file_name]).astype(int)
    return data_frame

########################################################################################################################
# a = [1, 2, 3]
# b = [4, 5, 6]
# c = [7, 8, 9]
# dfdf = pd.DataFrame({'a': a,
#                      'b': b,
#                      'c': c})
# type(dfdf)
# dfdf['mean'] = dfdf.mean(axis=1)
# dfdf['max'] = dfdf.max(axis=1)
# dfdf['min'] = dfdf.min(axis=1)
# dfdf['sum'] = dfdf.sum(axis=1)
# dfdf['sum'] = dfdf[['a', 'b', 'c']].sum(axis=1)
# dfdf['sum1'] = dfdf[dfdf.columns[:-4]].sum(axis=1)
# dfdf
########################################################################################################################
# df = pd.read_csv(nab_data_path + training_file_name)
# df['anomaly_label'] = df['timestamp'].isin(labels[training_file_name]).astype('int')
# df[df['anomaly_label'] == 1]
# df['anomaly_label'][946]


training_data_frame = load_data_frame_with_labels(training_file_name)
test_data_frame = load_data_frame_with_labels(test_file_name)
len(training_data_frame)
len(test_data_frame)
training_data_frame.value.plot()
test_data_frame.value.plot()



# Check the dataset head:

training_data_frame.head()
training_data_frame.describe()
training_data_frame[training_data_frame['anomaly_label'] == 1]
training_data_frame[training_data_frame['value'] >= 25.11]


# As we can see, it contains a timestamp, a CPU utilization value, and labels noting if this value is an anomaly.

# The next step is a visualization of the dataset with pyplot, which requires converting timestamps to time epochs:

def convert_timestamps(data_frame):
    data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'])
    data_frame['time_epoch'] = data_frame['timestamp'].astype(np.int64)

convert_timestamps(training_data_frame)
convert_timestamps(test_data_frame)

training_data_frame.head()
training_data_frame.tail()


# When plotting the data we mark anomalies with green dots:

def prepare_plot(data_frame):
    fig, ax = plt.subplots()
    ax.scatter(data_frame['time_epoch'], data_frame['value'], s=8, color='blue')  # scatter 산포그래프

    labled_anomalies = data_frame.loc[data_frame['anomaly_label'] == 1, ['time_epoch', 'value']]
    ax.scatter(labled_anomalies['time_epoch'], labled_anomalies['value'], s=200, color='green')

    return ax

# def prepare_plot(data_frame):
# #     fig, ax = plt.subplots()
# #     ax.scatter(data_frame['timestamp'], data_frame['value'], s=8, color='blue')  # scatter 산포그래프
# #
# #     labled_anomalies = data_frame.loc[data_frame['anomaly_label'] == 1, ['timestamp', 'value']]
# #     ax.scatter(labled_anomalies['timestamp'], labled_anomalies['value'], s=200, color='green')
# #
# #     return ax


figsize(16, 7)
prepare_plot(training_data_frame)
plt.show()



# The visualization of the training and test datasets look like this:
# visualization

figsize(16, 7)
prepare_plot(test_data_frame)
plt.show()


# Preparing a dataset
training_data_frame['value_no_anomaly'] = training_data_frame[training_data_frame['anomaly_label'] == 0]['value']

training_data_frame.loc[training_data_frame['anomaly_label'] == 1, ['value_no_anomaly']]
training_data_frame[training_data_frame['value_no_anomaly'].isnull()]


########################################################################################################################
# nan값 채우기

# 결측값을 특정 값으로 채우기:   df.fillna(0)
# 결측값을 앞 방향으로 채우기:   df.fillna(method: 'ffill' or 'pad')
# 결측값을 뒷 방향으로 채우기:   df.fillna(method: 'bill' or 'backfill')
# 결측값을 채우는 회수 제한하기: df.fillna(limit=1)
# 결측값을 평균값으로 채우기:    df.fillna(df['value'].mean())
########################################################################################################################
training_data_frame['value_no_anomaly'][945]
training_data_frame['value_no_anomaly'][946]
training_data_frame['value_no_anomaly'][947]

training_data_frame[['value_no_anomaly', 'anomaly_label']][940:950]

training_data_frame.columns



training_data_frame['value_no_anomaly'] = training_data_frame['value_no_anomaly'].fillna(method='ffill') # method 앞 값으로 채우기

training_data_frame['value'] = training_data_frame['value_no_anomaly']
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
data_scaler.fit(training_data_frame[features].values.astype(np.float32))
training_data = data_scaler.transform(training_data_frame[features].values.astype(np.float32))

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
batch_size = 48

training_data_batches = mx.gluon.data.DataLoader(training, batch_size, shuffle=False)
validation_data_batches = mx.gluon.data.DataLoader(validation, batch_size, shuffle=False)

for d in training_data_batches:
    break

# Xavier는 모든 layers에서 gradient scale이 거의 동일하게 유지하도록 설계됨
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# 이 모델에는 sgd 최적화 함수, 학습률은 0.01이 가장 최적으로 보임
# sgd에 비해 너무 작지 않고, 최적화에 시간이 오래 걸리지 않으며 너무 크지 않는다
# 그래서 loss function의 최소값을 넘지 않는다
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})


### Let’s run the training loop and plot MSEs
epochs = 90
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

plt.plot(training_mse, color='r')
plt.plot(validation_mse, color='b')

print(training_mse[-1], validation_mse[-1])

len(training_mse)

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


# 원래의 트레이닝 셋을 다시 불러옴
all_training_data = mx.gluon.data.DataLoader(training_data.astype(np.float32), batch_size, shuffle=False)

# 트레이닝셋과 reconstruction의 오차를 구함
training_reconstruction_errors = calculate_reconstruction_errors(all_training_data, L)
len(training_reconstruction_errors)

# 3*sigma: 평균에서 3*표준편차를 더한 값
reconstruction_error_threshold = np.mean(training_reconstruction_errors) + 3 * np.std(training_reconstruction_errors)
reconstruction_error_threshold

# 테스트셋 데이터 표준화
test_data = data_scaler.fit_transform(test_data_frame[features].values.astype(np.float32))

# 테스트셋 배치로 나누기
test_data_batches = mx.gluon.data.DataLoader(test_data, batch_size, shuffle=False)

# 테스트셋 reconstruction errors 구하기
test_reconstruction_errors = calculate_reconstruction_errors(test_data_batches, L)


# 테스트셋의 reconstruction errors들이 3시그마 법칙보다 이상이면 1, 아니면 0값으로 반환
predicted_test_anomalies = list(map(lambda v: 1 if v > reconstruction_error_threshold else 0, test_reconstruction_errors))
len(predicted_test_anomalies)


test_data_frame['anomaly_predicted'] = predicted_test_anomalies
test_data_frame.columns
test_data_frame[['anomaly_label', 'anomaly_predicted']]

test_data_frame[test_data_frame['anomaly_predicted'] == 1]
test_data_frame[test_data_frame['value'] >= 18.3333]


figsize(16, 7)

ax = prepare_plot(test_data_frame)

predicted_anomalies = test_data_frame.loc[test_data_frame['anomaly_predicted'] == 1, ['time_epoch', 'value']]
ax.scatter(predicted_anomalies['time_epoch'], predicted_anomalies['value'], s=50, color='red')

plt.show()


test_labels = test_data_frame['anomaly_label'].astype(np.float32)

score = f1_score(test_labels, predicted_test_anomalies)
print('F1 score: ' + str(score))

########################################################################################################################
##### 이상탐지 사례
## 1. 비즈니스 분석
# 매출이 이전에 비해 떨어진 판매 채널
# 주문량이 이전에 비해서 증가한 거래처
# 구매량이 이전에 비해서 증가 또는 감소한 품목
# 다른 경쟁사로 이탈할 징후가 보이는 쇼핑몰 또는 서비스의 고객
# 그 자체로는 문제가 없지만 다른 상품의 판매에 의해 영향을 받는 제품
# 평소와 다른 공간 이동 패턴을 보이는 차량 또는 모바일 디바이스 사용자

## 2. 컴퓨터 시스템 관리
# 자원 사용량(메모리, 디스크IO, 네트워크IO 등)이 이전에 비해서 증가한 시스템
# 접속이 없었던 IP주소들로부터의 시스템 접근량의 증가 (DDOS의 징후)
# 지금은 문제가 없지만 이대로 간다면 틀림없이 다운될 것 같은 시스템 또는 시스템 클러스터

## 3. 온라인 서비스 관리
# 이전에 비해서 증가한 로그인 실패 기록
# 이전에 발생하지 않은 에러코드의 기록 또는 더 빈발해진 기록
# 다른 사용자와 다른 이동패턴이나 접근 패턴을 보이는 사용자

## 4. 공정 관리
# 불량품이 늘어날 것 같은 조짐이 보이는 공정
# 현재 불량률은 차이가 없지만 후에 불량품이 대량 발생되어 리콜될 것 같은 제품 라인
# 큰 고장이 발생할 것 같은 설비

## 5. 보안
# 기밀문서 불출 횟수가 상대적으로 증가한 직원
# 임계치를 넘지는 않는 범위내에서 이전보다 증가한 보안 접근 실패
# 사람의 접근이 거의 없는 지역에 이전보다 다소 증가한 통행량