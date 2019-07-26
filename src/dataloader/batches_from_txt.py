import numpy as np

def generate_batches(data_path, char_threshold, sequence_length):
    """
    Inputs:
        data_path: File path to simple txt file.
        char_threshold: If count of any character is below char_threshold it is
    ignored and replaced with null token: ^'
        sequence_length: Sequence length of data for one time step of RNN.

    """
    data = open(data_path, 'r').read() # should be simple plain text file

    # lower case
    data = data.lower()
    chars = list(set(data))

    # only take frequent ones
    chars = [c for c in chars if data.count(c) > char_threshold]
    data = ''.join([d if d in chars else '^' for d in data])

    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    max_id = max(ix_to_char.keys())
    # null token is ^
    char_to_ix['^']= max_id + 1
    ix_to_char[-1]= '^'

    char_id_data = np.array([char_to_ix[c] for c in data])

    # len of chars
    N = char_id_data.shape[0]
    batch_data = np.zeros((N//sequence_length, sequence_length))
    for i in range(N//sequence_length):
        batch_data[i, :] = char_id_data[i*sequence_length:(i+1)*sequence_length]
    batch_data = batch_data.astype(int)

    return batch_data, char_to_ix, ix_to_char
