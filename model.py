
import os
import torch

current_working_directory = os.getcwd()

input_file_path = os.path.join(current_working_directory, 'Data', 'input.txt')

with open(input_file_path, 'r') as file:
    text =file.read()

chars = sorted(list(set(text)))
str_to_index  = {ch: i for i, ch in enumerate(chars)}
index_to_str = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [str_to_index[c] for c in s]
decode = lambda l: ''.join([index_to_str[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

n =int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]