import numpy as np
from tqdm import tqdm
import pickle

from ..optimizer.adam import adam
from matplotlib import pyplot as plt


class CharRNNSolver(object):
    def __init__(self, model, data, **kwargs):
        """
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data from load_coco_data

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'adam'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
          rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - save_model: Boolean, save model end of every epoch.
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 50)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.save_model = kwargs.pop('save_model', True)

        self.print_every = kwargs.pop('print_every', 100)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        self.update_rule = eval(self.update_rule)
        self._reset()

    def _sample_batch(self, data, batch_size=100, split='train'):
        mask = np.random.choice(data.shape[0], batch_size)
        captions = data[mask, :]
        return captions

    def _reset(self):
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        # captions (N, T)  where each element is in the range 0 <= y[i, t] < V
        captions = self._sample_batch(self.data,
                                      batch_size=self.batch_size,
                                      split='train')

        # Compute loss and gradient
        loss, grads = self.model.loss(captions)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.data.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        pbar = tqdm(range(num_iterations))
        for t in pbar:
            self._step()
            pbar.set_description('loss: %f' % (
                self.loss_history[-1]))
            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('\n#### New Sample for Iteration {} ####'.format(t))
                print(''.join([self.model.idx_to_char[w] for w in
                               self.model.sample(temperature=0.5,
                                                 start_token=np.random.choice(
                                                     list(self.model.char_to_idx.values())),
                                                 max_length=100).squeeze()]))

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
                if self.save_model:
                    save_path = 'model-files/char_rnn_epoch' + \
                        str(self.epoch) + '.pkl'
                    with open(save_path, 'wb') as f:
                        print("Saving model in {}".format(save_path))
                        pickle.dump(self.model, f)
                self.epoch += 1

