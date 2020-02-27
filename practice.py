import mxnet as mxnet

from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon


class UnRolledRNN_Model(Block):
    def __init__(self, vocab_size, num_embed, num_hidden, **kwargs):
        super(UnRolledRNN_Model, self).__init__(**kwargs)
        self.num_embed = num_embed
        self.vocab_size = vocab_size

        # use name_scope to give child Blocks appropriate names.
        # It also allows sharing Parameters between Blocks recursively.
        with self.name_scope():
            self.encoder = nn.Embedding(self.vocab_size, self.num_embed)
            self.dense1 = nn.Dense(num_hidden, activation='relu', flatten=True)
            self.dense2 = nn.Dense(num_hidden, activation='relu', flatten=True)
            self.dense3 = nn.Dense(vocab_size, flatten=True)

    def forward(self, inputs):
        emd = self.encoder(inputs)
        # print( emd.shape )
        # since the input is shape (batch_size, input(3 characters) )
        # we need to extract 0th, 1st, 2nd character from each batch
        character1 = emd[:, 0, :]
        character2 = emd[:, 1, :]
        character3 = emd[:, 2, :]
        # green arrow in diagram for character 1
        c1_hidden = self.dense1(character1)
        # green arrow in diagram for character 2
        c2_hidden = self.dense1(character2)
        # green arrow in diagram for character 3
        c3_hidden = self.dense1(character3)
        # yellow arrow in diagram
        c1_hidden_2 = self.dense2(c1_hidden)
        addition_result = F.add(c2_hidden, c1_hidden_2)  # Total c1 + c2
        addition_hidden = self.dense2(addition_result)  # the yellow arrow
        addition_result_2 = F.add(addition_hidden, c3_hidden)  # Total c1 + c2
        final_output = self.dense3(addition_result_2)
        return final_output


vocab_size = len(chars) + 1  # the vocabsize
num_embed = 30
num_hidden = 256
# model creation
simple_model = UnRolledRNN_Model(vocab_size, num_embed, num_hidden)
# model initilisation
simple_model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(simple_model.collect_params(), 'adam')
loss = gluon.loss.SoftmaxCrossEntropyLoss()



hidden_state_at_t = tanh(WI x input + WH x previous_hidden_state + bias)


# total of characters in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)

# maps character to unique index e.g. {a:1,b:2....}
char_indices = dict((c, i) for i, c in enumerate(chars))
# maps indices to character (1:a,2:b ....)
indices_char = dict((i, c) for i, c in enumerate(chars))

# mapping the dataset into index
idx = [char_indices[c] for c in text]


# input for neural network(our basic rnn has 3 inputs, n samples)
cs = 3
c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
# the output of rnn network (single vector)
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]

# stacking the inputs to form (3 input features )
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

# the output (1 X N data points)
y = np.stack(c4_dat[:-2])

col_concat = np.array([x1, x2, x3])
t_col_concat = col_concat.T
print(t_col_concat.shape)



# Set the batchsize as 32, so input is of form 32 X 3
# output is 32 X 1
batch_size = 32


def get_batch(source, label_data, i, batch_size=32):
    bb_size = min(batch_size, source.shape[0] - 1 - i)
    data = source[i: i + bb_size]
    target = label_data[i: i + bb_size]
    # print(target.shape)
    return data, target.reshape((-1, ))



# prepares rnn batches
# The batch will be of shape is (num_example * batch_size) because of RNN uses sequences of input     x
# for example if we use (a1,a2,a3) as one input sequence , (b1,b2,b3) as another input sequence and (c1,c2,c3)
# if we have batch of 3, then at timestep '1'  we only have (a1,b1.c1) as input, at timestep '2' we have (a2,b2,c2) as input...
# hence the batchsize is of order
# In feedforward we use (batch_size, num_example)
def rnn_batch(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data



#get the batch

def get_batch(source, i, seq):
    seq_len = min(seq, source.shape[0] - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len]
    return data, target.reshape((-1,))



class GluonRNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, **kwargs):
        super(GluonRNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer=mx.init.Uniform(0.1))

            if mode == 'lstm':
                #  we create a LSTM layer with certain number of hidden LSTM cell and layers
                #  in our example num_hidden is 1000 and num of layers is 2
                #  The input to the LSTM will only be passed during the forward pass (see forward function below)
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                #  we create a GRU layer with certain number of hidden GRU cell and layers
                #  in our example num_hidden is 1000 and num of layers is 2
                #  The input to the GRU will only be passed during the forward pass (see forward function below)
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                #  we create a vanilla RNN layer with certain number of hidden vanilla RNN cell and layers
                #  in our example num_hidden is 1000 and num of layers is 2
                #  The input to the vanilla will only be passed during the forward pass (see forward function below)
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            self.decoder = nn.Dense(vocab_size, in_units=num_hidden)
            self.num_hidden = num_hidden

    #  define the forward pass of the neural network
    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        #  emb, hidden are the inputs to the hidden
        output, hidden = self.rnn(emb, hidden)
        #  the ouput from the hidden layer to passed to drop out layer
        output = self.drop(output)
        #  print('output forward',output.shape)
        #  Then the output is flattened to a shape for the dense layer
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    # Initial state of RNN layer
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)




def trainGluonRNN(epochs, train_data, seq=seq_length):
    for epoch in range(epochs):
        total_L = 0.0
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, seq_length)):
            data, target = get_batch(train_data, i, seq)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target) # this is total loss associated with seq_length
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and seq_length to balance it.
            gluon.utils.clip_global_norm(grads, clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % log_interval == 0 and ibatch > 0:
                cur_L = total_L / seq_length / batch_size / log_interval
                print('[Epoch %d Batch %d] loss %.2f', epoch + 1, ibatch, cur_L)
                total_L = 0.0
        model.save_params(rnn_save)



hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)

sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T, ctx=context)

output, hidden = model(sample_input, hidden)
        index = mx.nd.argmax(output, axis=1)
        index = index.asnumpy()
        count = count + 1



new_string = new_string + indices_char[index[-1]]
input_string = input_string[1:] + indices_char[index[-1]]


import sys

# a nietzsche like text generator
def generate_random_text(model, input_string, seq_length, batch_size, sentence_length):
    count = 0
    new_string = ''
    cp_input_string = input_string
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    while count < sentence_length:
        idx = [char_indices[c] for c in input_string]
        if(len(input_string) != seq_length):
            print(len(input_string))
            raise ValueError('there was a error in the input ')
        sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T, ctx=context)
        output, hidden = model(sample_input, hidden)
        index = mx.nd.argmax(output, axis=1)
        index = index.asnumpy()
        count = count + 1
        new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
    print(cp_input_string + new_string)