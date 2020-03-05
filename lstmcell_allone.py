#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.gluon.data import ArrayDataset, DataLoader
import random

mx.random.seed(128)
np.random.seed(128)
random.seed(128)

##hyperparameter
splits = [0.80, 0.10]
batch_size = 32
dropout = 0.5
lr = 0.003
epochs = 5
ctx = mx.gpu(0)
optimizer_ = "sgd"


def getData():
    data = nd.random.normal(shape=(100, 5, 10))
    label = nd.random.normal(shape=(100, 1))
    return data, label


def split_load_Data(data, label, splits):
    training_examples = int(data.shape[0] * splits[0])
    valid_examples = int(data.shape[0] * splits[1])
    x_train, y_train = data[:training_examples], label[:training_examples]
    x_valid, y_valid = data[training_examples:training_examples + valid_examples], label[
                                                                                   training_examples:training_examples + valid_examples]
    x_test, y_test = data[training_examples + valid_examples:], label[training_examples + valid_examples:]
    print("train data shape: ", x_train.shape, y_train.shape)
    print("validation data shape: ", x_valid.shape, y_valid.shape)
    print("Test data shape: ", x_test.shape, y_test.shape)
    train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(x_valid, y_valid), batch_size=batch_size)
    test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test, y_test), batch_size=batch_size)

    return train_iter, val_iter, test_iter


def evaluate_accuracy(model, dataloader):
    eval_metrics_1 = mx.metric.MAE()
    eval_metrics_2 = mx.metric.MSE()
    eval_metrics_3 = mx.metric.RMSE()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metrics_1, eval_metrics_2, eval_metrics_3]:
        eval_metrics.add(child_metric)
    for i, (data, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        preds = model(data)
        eval_metrics.update(labels=label, preds=preds)
    return eval_metrics.get()


class NET(gluon.Block):
    def __init__(self, **kwargs):
        super(NET, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = rnn.LSTMCell(hidden_size=20)
            self.batchnorm = nn.BatchNorm(axis=2)
            self.dense = nn.Dense(1, flatten=True)

    def forward(self, inputs):
        enout, (next_h, next_c) = self.encoder.unroll(inputs=inputs, length=5, merge_outputs=True)
        enout = self.batchnorm(enout)
        enout = self.dense(enout)
        return (enout)


def fit(model):
    for e in range(epochs):
        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                _out = model(data)
                L = loss1(_out, label)
            L.backward()
            trainer.step(data.shape[0])

        val_err = evaluate_accuracy(model, val_iter)
        print("Epoch %s. Valid MAE: %s " % (e + 1, val_err))

    return model


data, label = getData()
print("data shape is:", data.shape, "\t", "label shape is", label.shape)

train_iter, val_iter, test_iter = split_load_Data(data, label, splits)
data, label = None, None

net = NET()
print(net)
net.collect_params().initialize(mx.init.Xavier(rnd_type="gaussian", factor_type="avg", magnitude=3), ctx=ctx)
loss1 = mx.gluon.loss.L1Loss()
trainer = mx.gluon.Trainer(net.collect_params(), optimizer_,
                           {'learning_rate': lr, 'multi_precision': True, 'wd': 0.0005, 'momentum': 0.9})

print("Loss before training:  ", evaluate_accuracy(net, test_iter))
net = fit(net)
print("Test : ", evaluate_accuracy(net, test_iter))








