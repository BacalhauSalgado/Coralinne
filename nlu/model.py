import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower())
    outputs.append('{}\{}'.format(command['entity'], command['action']))


# Processar texto: palavras, caracteres, bytes, sub-palavras

chars = set()

for input in inputs + outputs:
    for ch in input:
        if ch not in chars:
            chars.add(ch)


max_sqe = max([len(x) for x in inputs])


print('Número de chars: ',len(chars))
print('Maior sequencia: ',max_sqe)

# Mapear char-idx

chr2idx = {}
idx2chr = {}

for i, ch in enumerate(chars):
    chr2idx[ch] = i
    idx2chr[i] = ch

