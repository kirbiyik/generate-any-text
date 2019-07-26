import numpy as np

from ..layers.char_embedding import char_embedding_forward, char_embedding_backward
from ..layers.rnn import rnn_step_forward, rnn_step_backward, rnn_forward, rnn_backward
from ..layers.fully_connected import affine_forward, temporal_affine_backward, temporal_affine_forward
from ..layers.temporal_softmax import temporal_softmax_loss


class CharRNN(object):
    def __init__(self, char_to_idx, charvec_dim=128,
                hidden_dim=128, one_hot=False, dtype=np.float32):
        """
        Inputs:
        - char_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - charvec_dim: Dimension W of char vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        self.dtype = dtype
        self.char_to_idx = char_to_idx
        self.idx_to_char = {i: w for w, i in char_to_idx.items()}
        self.params = {}
        self.hidden_dim = hidden_dim
        self.one_hot = one_hot
        vocab_size = len(char_to_idx)
        self.vocab_size = vocab_size
        # null token is ^
        self._null = char_to_idx['^']
        self._start = char_to_idx.get('<START>', None)
        self._end = char_to_idx.get('<END>', None)

        # Initialize char vectors
        self.params['W_embed'] = np.random.randn(vocab_size, charvec_dim)
        self.params['W_embed'] /= 100

        # Initialize parameters for the RNN
        dim_mul = 1
        self.params['Wx'] = np.random.randn(charvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(charvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, chars):
        """
        Inputs:
        - chars: Ground-truth chars; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut chars into two pieces: chars_in has everything but the last char
        # and will be input to the RNN; chars_out has everything but the first
        # char and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce char (t+1)
        # after receiving char t. The first element of chars_in will be the START
        # token, and the first element of chars_out will be the first char.
        chars_in = chars[:, :-1]
        chars_out = chars[:, 1:]

        mask = (chars_out != self._null)

        # char embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}

        # char_vector is (N, T, D)
        initial_hidden = np.zeros((chars_in.shape[0], self.hidden_dim))
        # one hot
        if self.one_hot:
            char_vector_input = np.eye(self.vocab_size)[chars_in]
            print(char_vector_input)
        else:
            char_vector_input, cache2 = char_embedding_forward(
                chars_in, W_embed)
        hidden_states, cache3 = rnn_forward(x=char_vector_input,
                                            h0=initial_hidden, Wx=Wx, Wh=Wh, b=b)
        out, cache4 = temporal_affine_forward(x=hidden_states, w=W_vocab,
                                              b=b_vocab)
        loss, dx = temporal_softmax_loss(x=out, y=chars_out, mask=mask)

        # backward
        dout_affine, dW_vocab, db_vocab = temporal_affine_backward(dx, cache4)
        dout_rnn, dh0, dWx, dWh, db = rnn_backward(dout_affine, cache3)
        dW_embed = char_embedding_backward(dout_rnn, cache2)

        grads.update({'W_embed': dW_embed,
                      'Wx': dWx,
                      'Wh': dWh,
                      'b': db,
                      'W_vocab': dW_vocab,
                      'b_vocab': db_vocab
                      })

        return loss, grads

    def sample(self, temperature=0.5, start_token=None, max_length=60, argmax_sampling=False):
        """
        Inputs:
        - max_length: Maximum length T of generated chars.

        Returns:
        - chars: Array of shape (N, max_length) giving sampled chars,
          where each element is an integer in the range [0, V). The first element
          of chars should be the first sampled char, not the <START> token.
        """
        if not start_token:
            start_token = self._start
        N = 1
        chars = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        # initially start token, then it will be sampled from output
        sampled_chars = start_token * np.ones((N), dtype=np.int32)
        #
        initial_hidden = np.zeros((N, self.hidden_dim))

        prev_h = initial_hidden
        # first tokens are <START> for each element of mini-batch
        chars[:, 0] = sampled_chars
        for i in range(1, max_length):
            x, _ = char_embedding_forward(sampled_chars, W_embed)
            next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
            out, _ = affine_forward(next_h, W_vocab, b_vocab)

            out /= temperature
            # Softmax part
            out = out  # - np.max(out)
            numerator = np.exp(out)
            denominator = np.sum(numerator)
            out = numerator/denominator

            if argmax_sampling:
                sampled_chars = np.argmax(out, axis=1)
            # use multinomial dist instead of argmax sampling
            else:
                sampled_chars = np.random.multinomial(1, out.squeeze())
                sampled_chars = np.where(sampled_chars == 1)[0]
            chars[:, i] = sampled_chars
            prev_h = next_h
        return chars

    def sample_sentence(self, temperature=0.5, start_token=None, max_length=140):
        if start_token is None:
            start_token = self.idx_to_char[0]

        chars_list = [self.idx_to_char[w] for w in self.sample(temperature=0.5,
                                                               start_token=self.char_to_idx[start_token],
                                                               max_length=max_length).squeeze()]
        return ''.join(chars_list)
