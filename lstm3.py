# autoencoder :
# 우리가 구현할 신경망을 autoencoder 라고 합니다 .
# 오토 인코더는 입력 데이터를 중간 상태로 변환 한 다음 입력 피처 수와 비교하여 입력 함수의 근사값을 계산하는 신경망 유형입니다.
# 자동 인코더를 훈련시킬 때 아이디어는 입력 및 출력 값에 따라 일부 메트릭을 최소화하는 것입니다. 우리는이 경우에 MSE를 사용합니다.
# 다른 네트워크 설계와 하이퍼 파라미터의 성능을 비교하기 위해 F1 점수를 사용합니다.
# F1 점수는 정밀도와 리콜 간의 균형을 전달하며 일반적으로 이진 분류에 사용됩니다.

# Deep learning requires large amounts of data for real-world applications.
# But smaller datasets are acceptable for basic study, especially since model training doesn’t take much time.

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

path = './dataset/통신구/'


# Anomaly labels are stored separately from the data values. Let’s load the train and test datasets and label the values with pandas:

raw_data = pd.read_csv(path + 'water_level_fixed.csv')
data = raw_data.copy()

data.head()
data.set_index('time', inplace = True)

# label
for thres in data.columns:
    data.loc[data[thres] >= 0.443, "label"] = -1
    data.loc[data[thres] <= 0.17, "label"] = -1

data[data['label'].isnull]
data.loc[data['label'] != -1, "label"] = 1
data.loc[data['value'] <= 0.17, "value"]
data.loc[data['value'] >= 0.443, "value"]

data.head()
data.value.plot()
data.label.plot()

########################################################################################################################

#####
test = pd.read_csv(path + 'level.csv')
test.set_index('time', inplace = True)
test.head()
test.value1.plot()
test.label.plot()

# trainin data
training_data_frame = data[['time', 'value', 'label']]

training_data_frame.set_index('time', inplace = True)
training_data_frame.value.plot()
training_data_frame.label.plot()

#
# for thres in training_data_frame.columns:
#     training_data_frame.loc[training_data_frame[thres] >= 0.443, "label"] = -1
#     training_data_frame.loc[training_data_frame[thres] <= 0.17, "label"] = -1
#
# training_data_frame.loc[training_data_frame['label'] != -1, "label"] = 1
# training_data_frame.loc[training_data_frame['label'].isnull(), ['value']]


# test data
test_data_frame = data[['time', 'value', 'label']]
test_data_frame.set_index('time', inplace = True)
test_data_frame.value.plot()
test_data_frame.label.plot()

# for thres in test_data_frame.columns:
#     test_data_frame.loc[test_data_frame[thres] >= 0.443, "label"] = -1
#     test_data_frame.loc[test_data_frame[thres] <= 0.17, "label"] = -1
#
# test_data_frame.loc[test_data_frame['label'] != -1, "label"] = 1
# test_data_frame.loc[test_data_frame['label'].isnull(), ['value']]
# test_data_frame.reset_index().info()

# As we can see, it contains a timestamp, a CPU utilization value, and labels noting if this value is an anomaly.

# The next step is a visualization of the dataset with pyplot, which requires converting timestamps to time epochs:

def convert_timestamps(data_frame):
    data_frame.reset_index(inplace = True)
    data_frame['time'] = pd.to_datetime(data_frame['time'])
    data_frame['time_epoch'] = data_frame['time'].astype(np.int64)

convert_timestamps(training_data_frame)
convert_timestamps(test_data_frame)
training_data_frame
test_data_frame

# When plotting the data we mark anomalies with green dots:

def prepare_plot(data_frame):
    fig, ax = plt.subplots()
    ax.scatter(data_frame['time_epoch'],
               data_frame['value'], s=8, color='blue')

    labled_anomalies = data_frame.loc[data_frame['label'] == 1, ['time_epoch', 'value']]
    ax.scatter(labled_anomalies['time_epoch'],
               labled_anomalies['value'], s=200, color='green')

    return ax

figsize(16, 7)
prepare_plot(training_data_frame)
plt.show()

figsize(16, 7)
prepare_plot(test_data_frame)
plt.show()



# Preparing a dataset
# training_data_frame['value'] =
features = ['value']
feature_count = len(features)

data_scaler = preprocessing.StandardScaler()
data_scaler.fit(training_data_frame[features].values.astype(np.float32))

training_data = data_scaler.transform(training_data_frame[features].values.astype(np.float32))

rows = len(training_data)

split_factor = 0.8

training = training_data[0:int(rows * split_factor)]
validation = training_data[int(rows * split_factor):]


# Choosing a Model
model = mx.gluon.nn.Sequential()

with model.name_scope():
    model.add(mx.gluon.rnn.LSTM(feature_count))
    model.add(mx.gluon.nn.Dense(feature_count, activation='tanh'))


# Training & Evaluation
L = gluon.loss.L2Loss()

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


batch_size = 48

training_data_batches = mx.gluon.data.DataLoader(training, batch_size, shuffle=False)
validation_data_batches = mx.gluon.data.DataLoader(validation, batch_size, shuffle=False)


model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})


epochs = 15
training_mse = []
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

training_mse
validation_mse


def calculate_reconstruction_errors(input_data, L):
    reconstruction_errors = []
    for i, data in enumerate(input_data):
        input = data.as_in_context(ctx).reshape((-1, feature_count, 1))
        predicted_value = model(input)
        reconstruction_error = L(predicted_value, input).asnumpy().flatten()
        reconstruction_errors = np.append(
            reconstruction_errors, reconstruction_error)

    return reconstruction_errors


all_training_data = mx.gluon.data.DataLoader(training_data.astype(np.float32), batch_size, shuffle=False)

training_reconstruction_errors = calculate_reconstruction_errors(all_training_data, L)
reconstruction_error_threshold = np.mean(training_reconstruction_errors) + 3 * np.std(training_reconstruction_errors)



test_data = data_scaler.fit_transform(test_data_frame[features].values.astype(np.float32))

test_data_batches = mx.gluon.data.DataLoader(test_data, batch_size, shuffle=False)

test_reconstruction_errors = calculate_reconstruction_errors(test_data_batches, L)



predicted_test_anomalies = list(map(lambda v: 1 if v > reconstruction_error_threshold else -1, test_reconstruction_errors))

test_data_frame['anomaly_predicted'] = predicted_test_anomalies



figsize(16, 7)

ax = prepare_plot(test_data_frame)

predicted_anomalies = test_data_frame.loc[test_data_frame['anomaly_predicted'] == 1, ['time_epoch', 'value']]
ax.scatter(predicted_anomalies['time_epoch'], predicted_anomalies['value'], s=50, color='red')

plt.show()


test_labels = test_data_frame['anomaly_label'].astype(np.float32)

score = f1_score(test_labels, predicted_test_anomalies)
print('F1 score: ' + str(score))