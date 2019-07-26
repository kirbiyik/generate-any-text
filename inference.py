import pickle
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Inference arguments for text generation')
    parser.add_argument('-m', '--model', required=True,
                        help='Path for pickle file of RNN model.')
    parser.add_argument('-t', '--temperature', default=0.5,
                        help='Check Readme for more details about temperature.')
    parser.add_argument('-s', '--start-token', required=True,
                        help='Provide initial character to start generation.')
    parser.add_argument('--max-length', default=140,
                        help='Number of chars to be produced.')
    

    args = parser.parse_args()

    rnn_model = None
    with open(args.model, 'rb') as f:
        rnn_model = pickle.load(f)

    print(rnn_model.sample_sentence(args.temperature, args.start_token, args.max_length))
