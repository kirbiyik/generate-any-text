import numpy as np

def char_embedding_forward(x, W):
    """
    Inputs:
    - x: Integer array of shape (N, T) giving indices of chars. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving chars vectors for all chars.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving char vectors for all input chars.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    x = x.astype('int')
    out = W[x,:]
    cache = x, W
    return out, cache


def char_embedding_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of char embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    # shape of x -> (N, T)
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW