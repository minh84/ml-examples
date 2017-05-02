import numpy as np

def rnn_step_fwd(x, prev_h, W_xh, W_hh, b_h):
    '''
    run forward pass for single timestep of a rnn model:
            h = tanh(x.W_xh + prev_h.W_hh + b_h)
    :param x: 
    :param prev_h: 
    :param Wxh: 
    :param Whh: 
    :param b_h: 
    :return: 
    '''
    h = np.tanh(x.dot(W_xh) + prev_h.dot(W_hh) + b_h)
    cache = (x, prev_h, h, W_xh, W_hh)
    return h, cache

def rnn_step_bwd(dh, cache):
    '''
    run backward pass for single timestep of a rnn model:
        given dh and h is defined as 
             h = tanh(x.W_xh + prev_h.W_hh + b_h)
    the goal is to compute
        dx, dprev_h, dW_xh, dW_hh, db_h
    :param dh:    is gradient of dloss/dh which shoud have same shape as h 
    :param cache: 
    :return: 
    '''
    x, prev_h, h, W_xh, W_hh = cache

    # using chain rule and we know that dtanh(x)/dx = 1 - tanh(x)^2
    tmp = dh * (1.0 - h * h) # this has shape [hidden_dim]

    dx = np.dot(tmp, W_xh.T)
    dprev_h = np.dot(tmp, W_hh.T)
    db_h = np.sum(tmp, 0)
    dW_xh = np.dot(x.T, tmp)
    dW_hh = np.dot(prev_h.T, tmp)

    return dx, dprev_h, db_h, dW_xh, dW_hh