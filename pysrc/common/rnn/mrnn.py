import numpy as np
import contextlib

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"


@contextlib.contextmanager
def _checked_scope(cell, scope, reuse=None, **kwargs):
  if reuse is not None:
    kwargs["reuse"] = reuse
  with vs.variable_scope(scope, **kwargs) as checking_scope:
    scope_name = checking_scope.name
    if hasattr(cell, "_scope"):
      cell_scope = cell._scope  # pylint: disable=protected-access
      if cell_scope.name != checking_scope.name:
        raise ValueError(
            "Attempt to reuse RNNCell %s with a different variable scope than "
            "its first use.  First use of cell was with scope '%s', this "
            "attempt is with scope '%s'.  Please create a new instance of the "
            "cell if you would like it to use a different set of weights.  "
            "If before you were using: MultiRNNCell([%s(...)] * num_layers), "
            "change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  "
            "If before you were using the same cell instance as both the "
            "forward and reverse cell of a bidirectional RNN, simply create "
            "two instances (one for forward, one for reverse).  "
            "In May 2017, we will start transitioning this cell's behavior "
            "to use existing stored weights, if any, when it is called "
            "with scope=None (which can lead to silent model degradation, so "
            "this error will remain until then.)"
            % (cell, cell_scope.name, scope_name, type(cell).__name__,
               type(cell).__name__))
    else:
      weights_found = False
      try:
        with vs.variable_scope(checking_scope, reuse=True):
          vs.get_variable(_WEIGHTS_VARIABLE_NAME)
        weights_found = True
      except ValueError:
        pass
      if weights_found and reuse is None:
        raise ValueError(
            "Attempt to have a second RNNCell use the weights of a variable "
            "scope that already has weights: '%s'; and the cell was not "
            "constructed as %s(..., reuse=True).  "
            "To share the weights of an RNNCell, simply "
            "reuse it in your second calculation, or create a new one with "
            "the argument reuse=True." % (scope_name, type(cell).__name__))

    # Everything is OK.  Update the cell's scope and yield it.
    cell._scope = checking_scope  # pylint: disable=protected-access
    yield checking_scope

class BasicMRNNCell(RNNCell):
    '''
    Implement RNNs in this paper http://www.icml-2011.org/papers/524_icmlpaper.pdf
    
    '''
    def __init__(self, num_units, num_factors, activation=tanh, reuse=None):
        # hidden dim
        self._num_units = num_units

        # factor dim
        self._num_factors = num_factors

        # activation
        self._activation  = activation

        # flag to control reuse variable or not (in-case of sharing)
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def factor_size(self):
        return self._num_factors

    def __call__(self, inputs, state, scope = None):
        '''
        Multiplicative RNNs structure to model following dynamics
            f_t = diag(W_fx . x_t) . W_fh . prev_h
            h_t = tanh(W_hf . f_t + W_hx . x_t + b_h)
        The above is for single sample, for mini-batches 
        the situation is a bit more complicated
        
        :param inputs: 2D tensor of shape [batch_size, input_dim]
        :param state:  2D tensor of shape [batch_size, hidden_dim] 
        :param scope:  scope for sharing variable
        :return: 
        '''
        with _checked_scope(self,
                            scope or "multiplicative_rnn_cell",
                            reuse=self._reuse):
            input_shape = inputs.get_shape()
            dtype = inputs.dtype
            if input_shape.ndims != 2:
                raise ValueError("BasicMRNNCell is expecting 2D inputs, but inputs has shape: %s" % input_shape)

            state_shape = state.get_shape()
            if state_shape.ndims != 2:
                raise ValueError("BasicMRNNCell is expecting 2D state, but state has shape: %s" % state_shape)

            batch_size, input_dim = input_shape.as_list()


            # create w_fx, since initializer = None,
            # we will use Xavier initilizer (also called glorot_uniform_initializer)
            w_xf = vs.get_variable('%s_xf'%_WEIGHTS_VARIABLE_NAME,
                                   [input_dim, self._num_factors],
                                   dtype = dtype)

            w_hf = vs.get_variable('%s_hf'%_WEIGHTS_VARIABLE_NAME,
                                   [self._num_units, self._num_factors],
                                   dtype = dtype)

            w_xfh = vs.get_variable('%s_xfh' % _WEIGHTS_VARIABLE_NAME,
                                   [input_dim + self._num_factors, self._num_units],
                                   dtype=dtype)

            biases = vs.get_variable(_BIAS_VARIABLE_NAME,
                                     [self._num_units],
                                     dtype=dtype,
                                     initializer=init_ops.constant_initializer(0.,
                                                                               dtype=dtype))

            # multiplicative RNNs with batch-size
            xt_mul_w_xf     = math_ops.matmul(inputs, w_xf, name = 'xt_mul_w_xf')
            state_mul_w_hf  = math_ops.matmul(state, w_hf,  name = 'ht_mul_w_hf')

            ft = math_ops.multiply(xt_mul_w_xf, state_mul_w_hf, name = 'ft')
            xft = array_ops.concat([inputs, ft], 1)
            output = self._activation(math_ops.matmul(xft, w_xfh) + biases)
            return output, output