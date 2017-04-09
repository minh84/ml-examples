import numpy as np

def conv_fwd_naive(X, W, b, stride, pad):
    '''
    A naive implementation of the forward pass for convolution layer, it does
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

    batch, in_H, in_W, in_channels = X.shape
    filter_H, filter_W, _, out_channels = W.shape

    out_H = (in_H + 2 * pad - filter_H) // stride + 1
    out_W = (in_W + 2 * pad - filter_W) // stride + 1


    output = np.zeros([batch, out_H, out_W, out_channels], dtype=X.dtype)

    X_padded = np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='constant')

    W_reshaped = np.reshape(W, [-1, out_channels])
    b_reshaped = np.reshape(b, [1, out_channels])

    k_outH = 0
    for k in range(out_H):
        j_outW = 0
        for j in range(out_W):
            X_in = np.reshape(X_padded[:, k_outH : k_outH + filter_H, j_outW : j_outW + filter_W, :], [batch, -1])
            # X_in has shape [batch, filter_W * filter_H * in_channel]
            # W_reshape has shape [filter_W * filter_H * in_channel, out_channels]
            output[:, k, j, :] += X_in.dot(W_reshaped) + b_reshaped
            j_outW += stride
        k_outH += stride


    return output

def maxpool_fwd_naive(X, pH, pW, pS):
    batch, in_H, in_W, in_C = X.shape

    out_H = (in_H - pH) // pS + 1
    out_W = (in_W - pW) // pS + 1

    output = np.zeros([batch, out_H, out_W, in_C], dtype=X.dtype)

    X_tranpose = np.transpose(X, [0, 3, 1, 2])

    iH = 0
    for i in range(out_H):
        jW = 0
        for j in range(out_W):
            X_in = np.reshape(X_tranpose[:, :, iH : iH + pH, jW : jW + pW], [batch, in_C, -1])
            output[:, i, j, :] = np.max(X_in, axis=2)
            jW += pS
        iH += pS

    return output

def conv_test_input(dtype = np.float32):
    X = np.zeros([5, 5, 3], dtype=dtype)
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


    W = np.zeros([3, 3, 3, 2], dtype=dtype)
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

    b = np.array([1, 0], dtype=dtype)
    return np.array([X]), W, b

def maxpool_test_input(dtype = np.float32):
    x_shape = (2, 4, 4, 3)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape), dtype = dtype).reshape(x_shape)
    return x