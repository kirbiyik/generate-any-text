import unittest
import sys

import numpy as np

# tiny hack to ensure that always change directory to base path of project
# no matter this python file called from
project_path = '/'.join(__file__.split('/')[:-1] + ['../'])
sys.path.append(project_path)

from src.layers.rnn import rnn_step_forward, rnn_step_backward, rnn_forward, rnn_backward
from src.layers.temporal_softmax import temporal_softmax_loss
from src.layers.fully_connected import temporal_affine_backward, temporal_affine_forward
from src.layers.char_embedding import char_embedding_forward, char_embedding_backward


class TestLayers(unittest.TestCase):
    def eval_numerical_gradient(self, f, x, verbose=True, h=0.00001):
        """
        a naive implementation of numerical gradient of f at x
        - f should be a function that takes a single argument
        - x is the point (numpy array) to evaluate the gradient at
        """

        fx = f(x)  # evaluate function value at original point
        grad = np.zeros_like(x)
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h  # increment by h
            fxph = f(x)  # evalute f(x + h)
            x[ix] = oldval - h
            fxmh = f(x)  # evaluate f(x - h)
            x[ix] = oldval  # restore

            # compute the partial derivative with centered formula
            grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
            if verbose:
                print(ix, grad[ix])
            it.iternext()  # step to next dimension

        return grad

    def eval_numerical_gradient_array(self, f, x, df, h=1e-5):
        """
        Evaluate a numeric gradient for a function that accepts a numpy
        array and returns a numpy array.
        """
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index

            oldval = x[ix]
            x[ix] = oldval + h
            pos = f(x).copy()
            x[ix] = oldval - h
            neg = f(x).copy()
            x[ix] = oldval

            grad[ix] = np.sum((pos - neg) * df) / (2 * h)
            it.iternext()
        return grad

    def rel_error(self, x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    def test_rnn_step_forward(self):

        N, D, H = 3, 10, 4

        print('Checking RNN Step forward')
        x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
        Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
        Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
        b = np.linspace(-0.2, 0.4, num=H)

        next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
        expected_next_h = np.asarray([
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [0.66854692,  0.79562378,  0.87755553,  0.92795967],
            [0.97934501,  0.99144213,  0.99646691,  0.99854353]])

        self.assertTrue(self.rel_error(expected_next_h, next_h) < 1e-5,
                        "Relative error should be less than 1e-5")

    def test_rnn_step_backward(self):
        print('Checking RNN Step backward')

        np.random.seed(231)
        N, D, H = 4, 5, 6
        x = np.random.randn(N, D)
        h = np.random.randn(N, H)
        Wx = np.random.randn(D, H)
        Wh = np.random.randn(H, H)
        b = np.random.randn(H)

        out, cache = rnn_step_forward(x, h, Wx, Wh, b)

        dnext_h = np.random.randn(*out.shape)

        def fx(x): return rnn_step_forward(x, h, Wx, Wh, b)[0]

        def fh(prev_h): return rnn_step_forward(x, h, Wx, Wh, b)[0]

        def fWx(Wx): return rnn_step_forward(x, h, Wx, Wh, b)[0]

        def fWh(Wh): return rnn_step_forward(x, h, Wx, Wh, b)[0]

        def fb(b): return rnn_step_forward(x, h, Wx, Wh, b)[0]

        dx_num = self.eval_numerical_gradient_array(fx, x, dnext_h)
        dprev_h_num = self.eval_numerical_gradient_array(fh, h, dnext_h)
        dWx_num = self.eval_numerical_gradient_array(fWx, Wx, dnext_h)
        dWh_num = self.eval_numerical_gradient_array(fWh, Wh, dnext_h)
        db_num = self.eval_numerical_gradient_array(fb, b, dnext_h)

        dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
        self.assertTrue(self.rel_error(dx_num, dx) < 1e-5,
                        "dx elative error should be less than 1e-5")
        self.assertTrue(self.rel_error(dprev_h_num, dprev_h) < 1e-5,
                        "dprev_h relative error should be less than 1e-5")
        self.assertTrue(self.rel_error(dWx_num, dWx) < 1e-5,
                        "dWx relative error should be less than 1e-5")
        self.assertTrue(self.rel_error(dWh_num, dWh) < 1e-5,
                        "dWh relative error should be less than 1e-5")
        self.assertTrue(self.rel_error(db_num, db) < 1e-5,
                        "db relative error should be less than 1e-5")

    def test_vanilla_rnn_forward(self):
        print('Checking Vanilla RNN: forward')
        N, T, D, H = 2, 3, 4, 5
        x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D)
        h0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
        Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)
        Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)
        b = np.linspace(-0.7, 0.1, num=H)

        h, _ = rnn_forward(x, h0, Wx, Wh, b)
        expected_h = np.asarray([
            [
                [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
                [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
                [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
            ],
            [
                [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
                [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
                [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])

        self.assertTrue(self.rel_error(expected_h, h) < 1e-5,
                        "rnn_forward relative error should be less than 1e-5")

    def test_vanilla_rnn_backward(self):
        print('Checking Vanilla RNN: backward')
        np.random.seed(231)

        N, D, T, H = 2, 3, 10, 5

        x = np.random.randn(N, T, D)
        h0 = np.random.randn(N, H)
        Wx = np.random.randn(D, H)
        Wh = np.random.randn(H, H)
        b = np.random.randn(H)

        out, cache = rnn_forward(x, h0, Wx, Wh, b)

        dout = np.random.randn(*out.shape)

        dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

        def fx(x): return rnn_forward(x, h0, Wx, Wh, b)[0]

        def fh0(h0): return rnn_forward(x, h0, Wx, Wh, b)[0]

        def fWx(Wx): return rnn_forward(x, h0, Wx, Wh, b)[0]

        def fWh(Wh): return rnn_forward(x, h0, Wx, Wh, b)[0]

        def fb(b): return rnn_forward(x, h0, Wx, Wh, b)[0]

        dx_num = self.eval_numerical_gradient_array(fx, x, dout)
        dh0_num = self.eval_numerical_gradient_array(fh0, h0, dout)
        dWx_num = self.eval_numerical_gradient_array(fWx, Wx, dout)
        dWh_num = self.eval_numerical_gradient_array(fWh, Wh, dout)
        db_num = self.eval_numerical_gradient_array(fb, b, dout)

        self.assertTrue(self.rel_error(dx_num, dx) < 1e-5,
                        "dx relative error should be less than 1e-5")

        self.assertTrue(self.rel_error(dh0_num, dh0) < 1e-5,
                        "dh0 relative error should be less than 1e-5")

        self.assertTrue(self.rel_error(dWx_num, dWx) < 1e-5,
                        "dWx relative error should be less than 1e-5")

        self.assertTrue(self.rel_error(dWh_num, dWh) < 1e-5,
                        "dWh relative error should be less than 1e-5")

        self.assertTrue(self.rel_error(db_num, db) < 1e-5,
                        "db relative error should be less than 1e-5")

    def test_char_embedding_forward(self):
        print('Checking Character Embedding Forward')

        N, T, V, D = 2, 4, 5, 3

        x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
        W = np.linspace(0, 1, num=V*D).reshape(V, D)

        out, _ = char_embedding_forward(x, W)
        expected_out = np.asarray([
            [[0.,          0.07142857,  0.14285714],
             [0.64285714,  0.71428571,  0.78571429],
                [0.21428571,  0.28571429,  0.35714286],
                [0.42857143,  0.5,         0.57142857]],
            [[0.42857143,  0.5,         0.57142857],
             [0.21428571,  0.28571429,  0.35714286],
                [0.,          0.07142857,  0.14285714],
                [0.64285714,  0.71428571,  0.78571429]]])

        self.assertTrue(self.rel_error(expected_out, out) < 1e-5,
                        "char_embedding forward relative error should be less than 1e-5")

    def test_char_embedding_backward(self):
        np.random.seed(231)
        N, T, V, D = 50, 3, 5, 6
        x = np.random.randint(V, size=(N, T))
        W = np.random.randn(V, D)

        out, cache = char_embedding_forward(x, W)
        dout = np.random.randn(*out.shape)
        dW = char_embedding_backward(dout, cache)

        def f(W): return char_embedding_forward(x, W)[0]
        dW_num = self.eval_numerical_gradient_array(f, W, dout)
        self.assertTrue(self.rel_error(dW, dW_num) < 1e-5,
                        "char_embedding_forward relative error should be less than 1e-5")

    def test_temporal_affine_backward(self):
        np.random.seed(231)
        print('Checking Temporal Affine Layer Backward')

        # Gradient check for temporal affine layer
        N, T, D, M = 2, 3, 4, 5
        x = np.random.randn(N, T, D)
        w = np.random.randn(D, M)
        b = np.random.randn(M)

        out, cache = temporal_affine_forward(x, w, b)

        dout = np.random.randn(*out.shape)

        def fx(x): return temporal_affine_forward(x, w, b)[0]

        def fw(w): return temporal_affine_forward(x, w, b)[0]

        def fb(b): return temporal_affine_forward(x, w, b)[0]

        dx_num = self.eval_numerical_gradient_array(fx, x, dout)
        dw_num = self.eval_numerical_gradient_array(fw, w, dout)
        db_num = self.eval_numerical_gradient_array(fb, b, dout)

        dx, dw, db = temporal_affine_backward(dout, cache)

        self.assertTrue(self.rel_error(dx_num, dx) < 1e-5,
                        "dx relative error should be less than 1e-5")
        self.assertTrue(self.rel_error(dw, dw_num) < 1e-5,
                        "dw relative error should be less than 1e-5")
        self.assertTrue(self.rel_error(db_num, db) < 1e-5,
                        "db relative error should be less than 1e-5")

    def test_softmax_forward(self):
        print('Checking softmax loss')

        def check_loss(N, T, V, p, expected):
            x = 0.001 * np.random.randn(N, T, V)
            y = np.random.randint(V, size=(N, T))
            mask = np.random.rand(N, T) <= p
            self.assertTrue(self.rel_error(temporal_softmax_loss(x, y, mask)[0], expected) < 1e-2,
                            "loss relative error should be less than 1e-2")  # not sure about 1e-2

        check_loss(100, 1, 10, 1.0, expected=2.3)   # Should be about 2.3
        check_loss(100, 10, 10, 1.0, expected=23)  # Should be about 23
        check_loss(5000, 10, 10, 0.1, expected=2.3)  # Should be about 2.3

    def test_softmax_backward(self):
        # Gradient check for temporal softmax loss
        N, T, V = 7, 8, 9

        x = np.random.randn(N, T, V)
        y = np.random.randint(V, size=(N, T))
        mask = (np.random.rand(N, T) > 0.5)

        loss, dx = temporal_softmax_loss(x, y, mask, verbose=False)

        dx_num = self.eval_numerical_gradient(
            lambda x: temporal_softmax_loss(x, y, mask)[0], x, verbose=False)

        self.assertTrue(self.rel_error(dx, dx_num) < 1e-5,
                        "softmax backward relative error should be less than 1e-5")


if __name__ == '__main__':
    unittest.main()
