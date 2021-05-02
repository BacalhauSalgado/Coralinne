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


# Mapear char-idx

chr2idx = {}
idx2chr = {}

for i, ch in enumerate(chars):
    chr2idx[ch] = i
    idx2chr[i] = ch


max_sqe = max([len(x) for x in inputs])

print('Número de chars: ',len(chars))
print('Maior sequencia: ',max_sqe)

# Criar o dataset one-rot (números de exemplos, tamanho da sequencia, número de caracteres)
input_data = np.zeros((len(inputs), max_sqe, len(chars)), dtype='int32')

# Criar labels para o classificador

labels = set(outputs)

label2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    label2idx[label] = k
    idx2label[k] = labels

output_data = []

for output in outputs:
    output_data.append(label2idx[output])

output_data = to_categorical(outputs, len(outputs))

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k, chr2idx[ch]] = 1.0


print(output_data[0])


