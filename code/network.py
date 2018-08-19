import sys
import csv
import codecs
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, GRU, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from preprocess import create_data
from testPreprocess import create_test_data

dropout_val1 = 0.5
dropout_val2 = 0.2
output_dim = 1
learning_rate = 0.001
vocab_size = 0
max_seq_len = 0


# Setting up the model
def create_model(num_samples, op_dim, d_val1, d_val2, data, maxlen):
    model = Sequential()
    model.add(LSTM(32, input_shape=(num_samples[1], num_samples[2]), return_sequences=False))
    model.add(Dropout(0.8))
    model.add(Dense(16))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


(tweets, embedded_tweets, labels), handles, hashtags, vocab, max_len = create_test_data()

# comb_data = []
# for i, e in enumerate(embedded_tweets):
#     comb_data.append([tweets[i], e, labels[i]])

# comb_data = np.array(comb_data)
# np.random.shuffle(comb_data)

# tweets = comb_data[:, 0]
# embedded_tweets = comb_data[:, 1]
# labels = comb_data[:, 2]

vocab_size = len(vocab)
max_seq_len = max_len

print "vocab size: ", vocab_size
print "max seq len", max_seq_len

data = pad_sequences(embedded_tweets, maxlen=max_seq_len, dtype='float32')
print "Data Preprocessed"

model = create_model(num_samples=data.shape,
                     op_dim=output_dim,
                     d_val1=dropout_val1,
                     d_val2=dropout_val2,
                     data=embedded_tweets,
                     maxlen=max_seq_len)
print "Model Created"

# model.summary()
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print "Model Compiled"

path = "weights2/weights-improvement-trump-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(data,
          labels,
          batch_size=6,
          epochs=150,
          verbose=1,
          shuffle=True,
          callbacks=callbacks_list,
          validation_split=0.2)

model.save('relevantNet.hdf5')
model.load_weights('relevantNet.hdf5')
#
(test_tweets, test_embedded_tweets, test_labels), test_handles, test_hashtags, test_vocab, test_max_len = create_test_data()

test_data = pad_sequences(test_embedded_tweets, maxlen=max_seq_len, dtype='float32')

print
print
print "PREDICTION PROBABILITIES (0:Irrelevant, 1:Relevant)"

print model.predict(test_data)

pred = model.predict(data[:20])

for n, i in enumerate(pred):
    print i
    print labels[n]
    print tweets[n]
 

def gen_index(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
print
####### Fitting the model with the data #######
for iter in range(1, 150):
    print "Iteration: ", iter
    model.fit(X_train, y_train, batch_size=64, nb_epoch=1, callbacks=callbacks_list, validation_split=0.2)
    start = np.random.randint(0, len(raw_text) - seq_len - 1)
    generated = ''
    sentence = raw_text[start: start + seq_len]
    generated += sentence
    print "Seed: ", sentence
    cnt = 0
    while cnt < 40:
        x = np.zeros((1, seq_len, n_vocab))
        for t, ch in enumerate(sentence):
            if ch in char_to_int:
                x[0, t, char_to_int[ch]] = 1
            else:
                x[0, t, char_to_int[' ']] = 1
        pred = model.predict(x, verbose=0)[0]
        temperature = 0.5
        next_index = int(gen_index(pred, temperature))
        next_char = int_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        # print next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
        cnt += 1
        if next_char in ('.', '?', '!') and cnt > 40:
            break
    print
