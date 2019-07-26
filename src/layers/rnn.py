import numpy as np


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    forward = x.dot(Wx) + prev_h.dot(Wh) + b
    # squeeze with tanh
    next_h = np.tanh(forward)
    cache = x, prev_h, Wx, Wh, forward
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    x, prev_h, Wx, Wh, forward = cache

    # derivative(tanh x) = 1 - tanh^2 x
    dforward = (1 - np.tanh(forward)**2) * dnext_h
    dx = dforward.dot(Wx.T)
    dprev_h = dforward.dot(Wh.T)
    dWx = x.T.dot(dforward)
    dWh = prev_h.T.dot(dforward)
    db = np.sum(dforward, axis=0)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    N, T = x.shape[0], x.shape[1]
    H = h0.shape[1]

    cache = []
    h = np.zeros((N, T, H))
    prev_h = h0
    for i in range(T):
        prev_h, cache_current = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
        h[:, i, :] = prev_h
        # cache_current[-1] -> forward value in each step
        cache.append(cache_current)
    return h, cache


def rnn_backward(dh, cache):
    """
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    dx = np.zeros((N, T, D))
    db = np.zeros(H)

    # Weights are added in each gradient calculation because the same weights are
    # applied on each input.
    dh_prev = np.zeros((N, H))
    for i in reversed(range(T)):
        dh_current = dh[:,i,:] + dh_prev
        dx[:, i, :], dh_prev, dWx_current, dWh_current, db_current = \
                                            rnn_step_backward(dh_current, cache[i])
        dWx += dWx_current
        dWh += dWh_current
        db += db_current
    dh0 = dh_prev

    return dx, dh0, dWx, dWh, db