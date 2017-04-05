import numpy as np

def conv_fwd_naive(X, W, b, stride, pad):
    '''
    A naive implementation of the forward pass for convolution layer, it does
        .) pad input with zeros
         X -> [batch, in_W + 2*pad, in_H + 2*pad, in_channels]
        .) sliding windows with step-size=stride and right-multiplies the filter matrix + biases
         X[i, u:u+filter_W, v:v+filter_H, in_channels] * W + b
    
    The output dimension has following formula
        out_W = (in_W + 2*pad - filter_W) / stride + 1
        out_H = (in_H + 2*pad - filter_H) / stride + 1
        
    It takes input
    :param X: input data of shape [batch, in_W, in_H, in_channels]
    :param W: filter of shape     [filter_W, filter_H, in_channels, out_channels]
    :param b: biases of shape     [out_channels]
    :param stride: step of sliding windows 
    :param pad: padding of input (add zero around input)
    :return: 
        a data of shape [batch, out_W, out_H, out_channels] 
    '''

    batch, in_W, in_H, in_channels = X.shape
    filter_W, filter_H, _, out_channels = W.shape

    out_W = (in_W + 2 * pad - filter_W) // stride + 1
    out_H = (in_H + 2 * pad - filter_H) // stride + 1

    output = np.zeros([batch, out_W, out_H, out_channels], dtype=np.float64)

    X_padded = np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='constant')

    W_reshaped = np.reshape(W, [-1, out_channels])
    b_reshaped = np.reshape(b, [1, out_channels])
    j_outW = 0
    for j in range(out_W):
        k_outH = 0
        for k in range(out_H):
            X_in = np.reshape(X_padded[:, j_outW : j_outW + filter_W, k_outH : k_outH + filter_H, :], [batch, -1])
            # X_in has shape [batch, filter_W * filter_H * in_channel]
            # W_reshape has shape [filter_W * filter_H * in_channel, out_channels]
            output[:, j, k, :] += X_in.dot(W_reshaped) + b_reshaped
            k_outH += stride
        j_outW += stride

    return output

def test_input():
    X = np.zeros([5, 5, 3], dtype=np.float64)
    X[:, :, 0] = np.array([[2, 2, 2, 0, 0],
                           [1, 1, 1, 0, 2],
                           [1, 2, 1, 2, 1],
                           [1, 2, 0, 2, 2],
                           [0, 0, 0, 0, 2]])

    X[:, :, 1] = np.array([[2, 0, 0, 0, 2],
                           [2, 2, 1, 0, 2],
                           [0, 1, 0, 1, 2],
                           [0, 0, 2, 2, 0],
                           [0, 2, 2, 2, 0]])

    X[:, :, 2] = np.array([[0, 0, 0, 2, 1],
                           [0, 0, 2, 0, 1],
                           [1, 1, 2, 0, 0],
                           [0, 0, 1, 1, 2],
                           [1, 0, 0, 1, 0]])


    W = np.zeros([3, 3, 3, 2], dtype=np.float64)
    W[:, :, 0, 0] = np.array([[-1, 1, 1],
                              [1, -1, 1],
                              [1, 1, -1]])
    W[:, :, 1, 0] = np.array([[0, 0, 0],
                              [0, 1, -1],
                              [0, -1, -1]])
    W[:, :, 2, 0] = np.array([[0, 0, -1],
                              [0, 0,  0],
                              [0, 1, -1]])

    W[:, :, 0, 1] = np.array([[1, 0, 1],
                              [1, -1, -1],
                              [-1, -1, -1]])
    W[:, :, 1, 1] = np.array([[-1, -1, 1],
                              [0, 0, 0],
                              [-1, -1, 0]])
    W[:, :, 2, 1] = np.array([[1, 1, 0],
                              [-1, 0, 0],
                              [0, -1, 0]])

    b = np.array([1, 0], dtype=np.float64)
    return np.array([X]), W, b