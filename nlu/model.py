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



print('Número de chars: ',len(chars))
