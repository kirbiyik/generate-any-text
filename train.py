import numpy as np
from matplotlib import pyplot as plt
import argparse

from src.model.CharRNN import CharRNN
from src.solver.CharRNNSolver import CharRNNSolver
from src.dataloader.batches_from_txt import generate_batches


parser = argparse.ArgumentParser(description='Training arguments for text generation')
parser.add_argument('-d', '--data', required=True,
                        help='Path for simple corpus file for training')

parser.add_argument('-v', '--visualize', default=False, help='Whether visualize training loss with plots')
parser.add_argument('-e', '--epoch', default=5, type=int, help='Total number of epochs.')
parser.add_argument('-b', '--batch-size', default=32,)
parser.add_argument('-c', '--char-threshold', default=20,
                help='If count of any character is below threshold it is ignored and replaced with null token: ^')
parser.add_argument('-s', '--sequence-length', default=50,
                help='Sequence length of data for one time step of RNN')

args = parser.parse_args()	

# load data and mappings
batch_data, char_to_ix, ix_to_char = generate_batches(args.data, args.char_threshold, args.sequence_length) 

# construct model
rnn_model = CharRNN(
          char_to_idx=char_to_ix,
          hidden_dim=256,
          charvec_dim=50,
        )


rnn_solver = CharRNNSolver(rnn_model, batch_data,
           update_rule='adam',
           num_epochs=args.epoch,
           batch_size=args.batch_size,
           optim_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.95,
           verbose=True, print_every=500,
           save_model=True,
         )

rnn_solver.train()


if args.visualize:
        # Plot the training losses
        plt.plot(rnn_solver.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training loss history')
        plt.show()