from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)


with open("../data/nlp/timemachine.txt") as f:
    time_machine = f.read()
time_machine = time_machine[:-38083]