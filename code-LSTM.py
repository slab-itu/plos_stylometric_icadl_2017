import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import io



###############3 train
filename = "train.txt"
#raw_text = open(filename).read()
with io.open(filename,'r',encoding='utf8') as f:
    raw_train = f.read()
#raw_train = ''.join([i if ord(i) < 128 else ' ' for i in raw_train])
raw_train = raw_train.lower()
# create mapping of unique chars to integers
chars_train = sorted(list(set(raw_train)))

print (chars_train)

char_to_int = dict((c, i) for i, c in enumerate(chars_train))

n_chars_train = len(raw_train)
n_vocab_train = len(chars_train)
print ("Total Characters: ", n_chars_train)
print ("Total Vocab: ", n_vocab_train)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 20
dataX_train = []
dataY_train = []
for i in range(0, n_chars_train - seq_length, 1):
	seq_in = raw_train[i:i + seq_length]
	seq_out = raw_train[i + seq_length]
	dataX_train.append([char_to_int[char] for char in seq_in])
	dataY_train.append(char_to_int[seq_out])
n_patterns_train = len(dataX_train)
print ("Total Patterns: ", n_patterns_train)
    
# reshape X to be [samples, time steps, features]
X_train = numpy.reshape(dataX_train, (n_patterns_train, seq_length, 1))
# normalize
X_train = X_train / float(n_vocab_train)
# one hot encode the output variable
y_train = np_utils.to_categorical(dataY_train)
    
    
###############3 test
filename = "test.txt"
#raw_text = open(filename).read()
with io.open(filename,'r',encoding='utf8') as f:
    raw_test = f.read()
#raw_test = ''.join([i if ord(i) < 128 else ' ' for i in raw_test])
raw_test=raw_test.lower()
# create mapping of unique chars to integers
chars_test = sorted(list(set(raw_test)))

print (chars_test)

char_to_int = dict((c, i) for i, c in enumerate(chars_test))

n_chars_test = len(raw_test)
n_vocab_test = len(chars_test)
print ("Total Characters: ", n_chars_test)
print ("Total Vocab: ", n_vocab_test)

# prepare the dataset of input to output pairs encoded as integers
#seq_length = 20
dataX_test = []
dataY_test = []
for i in range(0, n_chars_test - seq_length, 1):
	seq_in = raw_test[i:i + seq_length]
	seq_out = raw_test[i + seq_length]
	dataX_test.append([char_to_int[char] for char in seq_in])
	dataY_test.append(char_to_int[seq_out])
n_patterns_test = len(dataX_test)
print ("Total Patterns: ", n_patterns_test)
    
# reshape X to be [samples, time steps, features]
X_test = numpy.reshape(dataX_test, (n_patterns_test, seq_length, 1))
# normalize
X_test = X_test / float(n_vocab_train)
# one hot encode the output variable
y_test = np_utils.to_categorical(dataY_test)   


# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X_test.shape[1], X_test.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])  
model.fit(X_train, y_train, epochs=100, batch_size=128,validation_data=(X_test, y_test), shuffle=True)


#To evaluate use this
#score= model.evaluate(X_eval, y_eval,
#                            verbose=0)

#print("Accuracy: %.2f%%" % (score[1]))

# serialize model to YAML

model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model_author7.h5")
print("Saved model to disk")
