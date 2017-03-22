import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class FullyConnectedNN(object):
    '''
    construct a simple NNs with fully-connected layers that support 2D input: N x D
        hidden_dims:  size of neuron in each hidden layers
        input_dim:    D input's size, so for image, we flatten it to a long vector of pixel
        num_classes:  number of class that we want to classify
        reg:          l2 regulization on weights
        learning_rate:step size when doing gradient descent update
        lr_decay:     decay step size after each epoch
        weight_scale: scale init Gaussian for weight-init
        act_fn:       activation function default tanh
        dtype:        data-type of tensor default single precision tf.float32
    '''
    def __init__(self, hidden_dims, input_dim = 2, num_classes=2, reg = 0.0,
                 learning_rate = 1e-3, lr_decay = 1.0, weight_scale = 1e-2,
                 act_fn = tf.nn.tanh, dtype = tf.float32):

        self.hidden_dims_   = hidden_dims
        self.input_dim_     = input_dim
        self.num_classes_   = num_classes
        self.reg_           = reg
        self.initial_lr_    = learning_rate
        self.lr_decay_      = lr_decay

        dims = [input_dim] + hidden_dims
        self.params_ = {}
        self.layers_ = {}

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, input_dim])
            self.y = tf.placeholder(tf.int32, [None])

        h = None
        l2_reg = None
        for i in range(len(hidden_dims)):
            with tf.name_scope('hidden_layer_{}'.format(i+1)):
                self.params_['Wh_{}'.format(i+1)] = tf.Variable(tf.truncated_normal([dims[i], dims[i+1]]) * weight_scale, name = 'Wh_{}'.format(i+1))
                self.params_['bh_{}'.format(i+1)] = tf.Variable(tf.zeros(dims[i+1]), name = 'bh_{}'.format(i+1))

                if (i == 0):
                    h = tf.matmul(self.x, self.params_['Wh_{}'.format(i+1)]) + self.params_['bh_{}'.format(i+1)]
                    l2_reg = self.reg_ * tf.nn.l2_loss(self.params_['Wh_{}'.format(i+1)])
                else:
                    h = tf.matmul(h, self.params_['Wh_{}'.format(i + 1)]) + self.params_['bh_{}'.format(i + 1)]
                    l2_reg += self.reg_ * tf.nn.l2_loss(self.params_['Wh_{}'.format(i + 1)])

                self.layers_['fc_{}'.format(i+1)] = h
                h = act_fn(h)
                self.layers_['afc_{}'.format(i + 1)] = h

        with tf.name_scope('out_layer'):
            self.params_['Wo'] = tf.Variable(tf.truncated_normal([dims[-1], num_classes]) * weight_scale, name='Wo')
            self.params_['bo'] = tf.Variable(tf.zeros(num_classes), name='bo')
            self.logits = tf.matmul(h, self.params_['Wo']) + self.params_['bo']
            self.layers_['logits'] = self.logits

        with tf.name_scope('train'):
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                      labels=self.y)) + l2_reg
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate= self.lr).minimize(self.cost)

        with tf.name_scope('validation'):
            pred = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, self.y), tf.float32))

    '''
    train model given training-data and validation-data
        inputs:     training inputs of size N x D
        labels:     labels input of size N
        batch_size: number of sample per mini-batch
        epochs:     number of iteration through the data
        val_inputs: validation inputs of size N_val x D
        val_labels: validation labeles of size N_val

    function returns fitted_params: key -> fitted_weights/fitted_biases
    '''
    def train(self, inputs, labels, batch_size, epochs,
                    val_inputs, val_labels,
                    print_every=20):
        nb_samples = inputs.shape[0]
        iter_per_epoch = nb_samples // batch_size;
        nb_iters = iter_per_epoch * epochs
        fitted_params = {}
        learning_rate = self.initial_lr_
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(nb_iters):
                idx = np.random.choice(nb_samples, batch_size)
                _ = sess.run(self.train_op,
                             feed_dict={self.x : inputs[idx],
                                        self.y : labels[idx],
                                        self.lr : learning_rate})

                if (i+1) % iter_per_epoch==0:
                    learning_rate *= self.lr_decay_

                if (i % print_every == 0) or ((i+1) == nb_iters):
                    loss, acc = sess.run([self.cost, self.accuracy],
                                         feed_dict={self.x : val_inputs, self.y : val_labels})
                    print('{:>5d} iter loss = {:>10.4f} acc = {:>5.2f}'.format(i, loss, acc*100.))

            for k in self.params_:
                fitted_params[k] = self.params_[k].eval()

        return fitted_params

    '''
    predict new inputs given fitted params
        fitted_params:  trained parameters
        inputs:         new inputs that we want to classify

    function returns prediction of inputs' labels
    '''
    def predict(self, fitted_params, inputs):
        with tf.Session() as sess:
            # assign fitted weights & biases
            for k in fitted_params:
                sess.run(tf.assign(self.params_[k], fitted_params[k]))

            # forward pass to layer that we want to visualize
            outputs = self.forward(sess, 'logits', inputs)
            return np.argmax(outputs, axis=1)

    '''
    run forward step to specific layer given current session and inputs
        sess:   a tf.Session where variables has been assigned with fitted params
        layer:  name of layer e.g fc1 => hidden layer 1
        inputs: inputs to be feeded NNs to obtain layer's ouputs

    return layer's outputs
    '''
    def forward(self, sess, layer, inputs):
        return sess.run(self.layers_[layer], feed_dict = {self.x : inputs})

    '''
    simple visualize layer's output in 2D/3D
        layer:          name of layer that we want to visualize
        fitted_params:  trained parameters
        inputs:         new inputs
        labels:         inputs' label so that we can color (only support 2-class: 0/1)
        cut_params:     weights of the next layer
    '''
    def visualize(self, layer, fitted_params, inputs, labels, cut_params = None):
        with tf.Session() as sess:
            # assign fitted weights & biases
            for k in fitted_params:
                sess.run(tf.assign(self.params_[k], fitted_params[k]))

            # forward pass to layer that we want to visualize
            outputs = self.forward(sess, layer, inputs)

            # visualize layer if it 2D/3D
            colors = np.array(['b']*len(labels))
            colors[labels==1]= 'r'
            if (outputs.shape[1] == 2):
                ax = plt.subplot()
                ax.scatter(outputs[:, 0], outputs[:,1], c = colors)
                if cut_params is not None:
                    cut_w = cut_params['w'][:, 1] - cut_params['w'][:, 0]
                    cut_b = cut_params['b'][1] - cut_params['b'][0]
                    x = np.linspace(np.min(outputs[:,0]), np.max(outputs[:,0]), 20)
                    y = (-cut_b - cut_w[0] * x )/cut_w[1]
                    ax.plot(x, y, c='black')

            elif (outputs.shape[1] == 3):
                ax = plt.subplot(projection='3d')
                ax.view_init(30, 150)
                ax.scatter(outputs[:, 0], outputs[:,1], outputs[:,2], c = colors)

                if cut_params is not None:
                    cut_w = cut_params['w'][:, 1] - cut_params['w'][:, 0]
                    cut_b = cut_params['b'][1] - cut_params['b'][0]
                    x = np.linspace(np.min(outputs[:, 0]), np.max(outputs[:, 0]), 10)
                    y = np.linspace(np.min(outputs[:, 1]), np.max(outputs[:, 1]), 10)
                    X, Y = np.meshgrid(x, y)
                    Z = - (cut_w[0] * X + cut_w[1] * Y + cut_b)/cut_w[2]
                    ax.plot_surface(X, Y, Z, alpha=0.8, color='grey')

            else:
                raise Exception('visualize currently supports only 2D or 3D output')