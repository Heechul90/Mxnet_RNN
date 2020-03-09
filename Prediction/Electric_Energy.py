##### 전력량 예측하기

### 함수
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import mxnet as mx
from mxnet import nd, autograd, gluon

from sklearn import preprocessing
from sklearn.metrics import f1_score

### 하이퍼파라미터


### 데이터 불러오기
raw_data = pd.read_csv('../dataset/UCI/electricity.csv', sep=',', header=None)

df = raw_data.copy()


########################################################################################################################
# from sklearn.preprocessing import StandardScaler
#
# data_scaler = preprocessing.StandardScaler()
# data_scaler.fit(df.values.astype(np.float32))
#
# training_data = data_scaler.transform(df.values.astype(np.float32))
# training_data.shape
# rows = len(training_data)


window = 24*7
horizon = 24

data = []
label = []

for i in range(0, len(df) - (window+horizon)):
    x = df[i:i + window]
    y = df[i + window + horizon]

    data.append(x)
    label.append(y)

type(data[0])
len(data)
len(label)

data = mx.nd.array(data)
label = mx.nd.array(label)

batch_size = 48
training_data_batches = mx.gluon.data.DataLoader([data, label], batch_size, shuffle=False)


for d, l in training_data_batches:
    break

d.shape
l.shape

### 학습용/테스트용 데이터 생성
# 전체 70%를 학습용 데이터로 사용
train_size = int(len(data) * 0.8)

# 나머지(30%)를 테스트용 데이터로 사용
test_size = len(data) - train_size

# 데이터를 잘라 학습용 데이터 생성
trainX = np.array(data[0:train_size])
trainY = np.array(label[0:train_size])


# 데이터를 잘라 테스트용 데이터 생성
testX = np.array(data[train_size:len(data)])
testY = np.array(label[train_size:len(label)])


########################################################################################################################
### 텐서플로우 플레이스홀더 생성
# tf.placeholder(dtype, [shape], name)
# dtype : 데이터 타입을 의미하며 반드시 적어주어야 한다.
# shape : 입력 데이터의 형태를 의미한다. 상수 값이 될 수도 있고 다차원 배열의 정보가 들어올 수도 있다. ( 디폴트 파라미터로 None 지정 )
# name  : 해당 placeholder의 이름을 부여하는 것으로 적지 않아도 된다.  ( 디폴트 파라미터로 None 지정 )
# 입력 X, 출력 Y를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)

predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)


# 모델(LSTM 네트워크) 생성
def lstm_cell():
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  셀 스테이트에서 어떤 정보를 버릴지 선택하는 과정
    #               이 결정은 'forget gate laye'라고 불리는 시그모이드 레이어로 만들어짐
    #               이 게이트에서는 0과 1의 입력값을 받는다(0: 완전히 이 값을 버려라, 1: 완전히 이 값을 유지해라)
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


# Multi Layer Perceptron RNN: layer 연결
# 입력층과 출력층 사이에 하나 이상의 중간층(hidden layer: 은닉층)이 존재하는 신경망
# 네트워크는 입력층, 은닉층, 출력층 방향으로 연결
# 각 층내의 연결과 출력층에서 입력층으로의 직접적인 연결은 존재하지 않은 전방향 (Feedforward) 네트워크
# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

# [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
# 과거 여러일수의 전력량을 이용해서 다음날의 전력량 1개를 에측하기때문에 many-to-one 형태이다
# one to one: Vanilla Neural Networks
# one to many: Image Captioning - 한장의 이미지에 대해 여러개의 문장으로 해석하는 형태, '소년이 사과를 고르고 있다'
# many to one: Sentiment Classification - 여러개의 문장으로 구성된 글을 해석하여, 감정상태를 나타내는 형태, '긍정', '부정'
# many to many: Machine Translation - 여러개의 문장에서 각각의 문장들을 다른 언어로 해석해주는 형태, 'Hello' -> '안녕'
# many to many: Video classification - 여러개의 이미지에 대해 여러개의 설명, 번역을 하는 형태

hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

# 손실함수로 평균제곱오차를 사용한다
loss = tf.reduce_sum(tf.square(hypothesis - Y))

# 최적화함수로 AdamOptimizer를 사용한다
# optimizer = tf.train.RMSPropOptimizer(learning_rate) # LSTM과 궁합 별로임
optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)

# RMSE(Root Mean Square Error)
# 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균 오차가 나온다
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록한다
test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
test_predict = ''  # 테스트용데이터로 예측한 결과

### 세션 정의
# 세션 생성: Session 객체 생성. 분산 환경에서는 계산 노드와의 연결을 만든다.
# 세션 사용: run 메서드에 그래프를 입력하면 출력 값을 계산하여 반환한다. 분산 환경에서는 계산 노드로 그래프를 보내 계산을 수행한다.
# 세션 종료: close 메서드. with 문을 사용하면 명시적으로 호출하지 않아도 된다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def build_accuracy(predictions, labels_):
    """
    Create accuracy
    """
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

# # 학습
start_time = datetime.datetime.now()  # 시작시간을 기록한다
print('학습을 시작합니다...')

for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
        # 학습용데이터로 rmse오차를 구한다
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)

        # 테스트용데이터로 rmse오차를 구한다
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)

        # 현재 오류를 출력한다
        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1,
                                                                                 train_error,
                                                                                 test_error,
                                                                                 test_error - train_error))



print(build_accuracy(test_predict, testY))

# 하이퍼파라미터 출력
print('input_data_column_cnt:', input_data_column_cnt, end='')
print(',output_data_column_cnt:', output_data_column_cnt, end='')

print(',seq_length:', seq_length, end='')
print(',rnn_cell_hidden_dim:', rnn_cell_hidden_dim, end='')
print(',forget_bias:', forget_bias, end='')
print(',num_stacked_layers:', num_stacked_layers, end='')
print(',keep_prob:', keep_prob, end='')

print(',epoch_num:', epoch_num, end='')
print(',learning_rate:', learning_rate, end='')

print(',train_error:', train_error_summary[-1], end='')
print(',test_error:', test_error_summary[-1], end='')
print(',min_test_error:', np.min(test_error_summary))

# 결과 그래프 출력
plt.figure(1)
plt.plot(train_error_summary, 'red')
plt.plot(test_error_summary, 'blue')
plt.xlabel('Epoch(x1000)')
plt.ylabel('Root Mean Square Error')

plt.figure(2)
plt.plot(testY, 'red')
plt.plot(test_predict, 'blue')
plt.xlabel('Time Period')
plt.ylabel('elec')
plt.show()




# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
recent_data = np.array([x[len(x) - seq_length-1:-1]])
print("recent_data.shape: ", recent_data.shape)
print("recent_data: ", recent_data)

# 내일 전력량을 예측해본다
test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

print("test_predict: ", test_predict[0])
test_predict = reverse_min_max_scaling(elec, test_predict)  # 전력량 데이터를 다시 역정규화
print("Tomorrow's elec price: ", test_predict[0])             # 내일 전력량 데이터

# test_predict:  [0.5484068]
# Tomorrow's elec price:  [788970.2]

# test_predict [0.5380935]
# Tomorrow's elec price [787583.9]