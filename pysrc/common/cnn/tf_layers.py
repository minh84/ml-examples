import tensorflow as tf

def conv_fwd_tf(X, W, b, stride, pad):
    '''
    An implementation of the forward pass for convolution layer using tensorflow, it does
        .) pad input with zeros
         X -> [batch, in_H + 2*pad, in_W + 2*pad, in_channels]
        .) sliding windows with step-size=stride and right-multiplies the filter matrix + biases
         X[i, u:u+filter_H, v:v+filter_W, in_channels] * W + b

    The output dimension has following formula
        out_H = (in_H + 2*pad - filter_H) / stride + 1
        out_W = (in_W + 2*pad - filter_W) / stride + 1        

    It takes input
    :param X: input data of shape [batch, in_H, in_W, in_channels]
    :param W: filter of shape     [filter_H, filter_W, in_channels, out_channels]
    :param b: biases of shape     [out_channels]
    :param stride: step of sliding windows 
    :param pad: padding of input (add zero around input)
    :return: 
        a data of shape [batch, out_H, out_W, out_channels] 
    '''

    tf.reset_default_graph()

    # note that conv2d only support tf.float32
    # check this link: https://github.com/tensorflow/tensorflow/issues/5539
    pX = tf.placeholder(tf.float32, X.shape, name = 'input_X')

    vW = tf.Variable(W, dtype=tf.float32, name = 'weights')
    vb = tf.Variable(b, dtype=tf.float32, name = 'biases')

    paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
    strides = [1, stride, stride, 1]

    xPad = tf.pad(pX, paddings, mode='CONSTANT', name = 'x_padded')
    conv = tf.nn.conv2d(xPad, vW, strides=strides, padding='VALID') + vb

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(conv, feed_dict={pX:X})

def maxpool_fwd_tf(X, pH, pW, pS):
    vX = tf.placeholder(tf.float32, X.shape)

    ksize   = [1, pH, pW, 1]
    strides = [1, pS, pS, 1]
    maxpool = tf.nn.max_pool(vX, ksize=ksize, strides=strides, padding='VALID')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(maxpool, feed_dict={vX: X})
