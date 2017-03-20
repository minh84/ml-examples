import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class FullyConnectedNN(object):
    def __init__(self, hidden_dims, input_dim = 2, num_classes=2, reg = 0.0, learning_rate = 1e-3,
                 weight_scale = 1e-2, dtype = tf.float32, act_fn = tf.nn.tanh):
        self.hidden_dims_ = hidden_dims
        self.input_dim_ = input_dim
        self.num_classes_ = num_classes
        self.reg_ = reg

        dims = [input_dim] + hidden_dims
        self.params = {}

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.int32, [None])
        self.layers = {}
        h = None
        for i in range(len(hidden_dims)):
            with tf.name_scope('hidden_layer_{}'.format(i+1)):
                self.params['Wh_{}'.format(i+1)] = tf.Variable(tf.truncated_normal([dims[i], dims[i+1]]) * weight_scale, name = 'Wh_{}'.format(i+1))
                self.params['bh_{}'.format(i+1)] = tf.Variable(tf.zeros(dims[i+1]), name = 'bh_{}'.format(i+1))

                if (i == 0):
                    h = tf.matmul(self.x, self.params['Wh_{}'.format(i+1)]) + self.params['bh_{}'.format(i+1)]
                else:
                    h = tf.matmul(h, self.params['Wh_{}'.format(i + 1)]) + self.params['bh_{}'.format(i + 1)]

                self.layers['fc_{}'.format(i+1)] = h
                h = act_fn(h)
                self.layers['afc_{}'.format(i + 1)] = h

        with tf.name_scope('out_layer'):
            self.params['Wo'] = tf.Variable(tf.truncated_normal([dims[-1], num_classes]) * weight_scale, name='Wo')
            self.params['bo'] = tf.Variable(tf.zeros(num_classes), name='bo')
            self.logits = tf.matmul(h, self.params['Wo']) + self.params['bo']
            self.layers['logits'] = self.logits

        with tf.name_scope('train'):
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1).minimize(self.cost)

        with tf.name_scope('validation'):
            pred = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, self.y), tf.float32))

    def train(self, inputs, labels, batch_size, epochs, print_every=20):
        nb_samples = inputs.shape[0]
        iter_per_epoch = nb_samples // batch_size;
        nb_iters = iter_per_epoch * epochs
        fitted_params = {}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(nb_iters):
                idx = np.random.choice(nb_samples, batch_size)
                _ = sess.run(self.train_op,
                             feed_dict={self.x : inputs[idx], self.y : labels[idx]})
                if (i%print_every==0 or i+1==nb_iters):
                    loss, acc = sess.run([self.cost, self.accuracy],
                                         feed_dict={self.x : inputs, self.y : labels})
                    print('{:>5d} iter loss = {:>10.4f} acc = {:>5.2f}'.format(i, loss, acc*100.))

            for k in self.params:
                fitted_params[k] = self.params[k].eval()

        return fitted_params

    def forward(self, sess, layer, inputs):
        return sess.run(self.layers[layer], feed_dict = {self.x : inputs})

    def visualize(self, layer, fitted_params, inputs, labels, cut_params = None):
        with tf.Session() as sess:
            # assign fitted weights & biases
            for k in fitted_params:
                sess.run(tf.assign(self.params[k], fitted_params[k]))

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
                ax.view_init(180, 50)
                ax.scatter(outputs[:, 0], outputs[:,1], outputs[:,2], c = colors)

                if cut_params is not None:
                    cut_w = cut_params['w'][:, 1] - cut_params['w'][:, 0]
                    cut_b = cut_params['b'][1] - cut_params['b'][0]
                    x = np.linspace(np.min(outputs[:, 0]), np.max(outputs[:, 0]), 10)
                    y = np.linspace(np.min(outputs[:, 1]), np.max(outputs[:, 1]), 10)
                    X, Y = np.meshgrid(x, y)
                    Z = - (cut_w[0] * X + cut_w[1] * Y + cut_b)/cut_w[2]
                    ax.plot_surface(X, Y, Z, alpha=0.5)

            else:
                raise Exception('visualize currently supports only 2D or 3D output')