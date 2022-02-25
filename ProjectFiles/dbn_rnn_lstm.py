import tensorflow
import keras
import reconstruction
from reconstruction import my_accuracy_old
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Input
from dbn.tensorflow import SupervisedDBNRegression
from keras.layers import RepeatVector
from keras.layers.wrappers import TimeDistributed
import numpy as np

def DBN_procedure(data, dishonest, honest, hidden_layers_structure=[100,100],learning_rate_rbm=0.01,
                                    lr=0.01,
                                    n_epochs_rbm=15,
                                    n_iter_backprop=300,
                                    batch_size=64,
                                    activation='relu'):
  all_acc_DBN=[]
  for i in range(len(data)):

    regressor = SupervisedDBNRegression(hidden_layers_structure=hidden_layers_structure, 
                                      learning_rate_rbm=learning_rate_rbm,
                                      learning_rate=lr,
                                      n_epochs_rbm=n_epochs_rbm,
                                      n_iter_backprop=n_iter_backprop,
                                      batch_size=batch_size,
                                      activation_function=activation)
    regressor.fit(data[i][0], data[i][2])
    y_pred = regressor.predict(dishonest)
    all_acc_DBN.append(my_accuracy_old(y_pred, honest))
  return all_acc_DBN

def rnn(data,dishonest, honest, units=32, activation ='linear', epochs =10, batch_size=1, verbose=0):
  
  all_pred=[]
  h_total=[]

  for i in range(len(data)):
    X_train = data[i][0]
    X_test = data[i][1]
    y_train = data[i][2]
    y_test = data[i][3]
    allpred=[]
    model = Sequential()
    model.add(keras.layers.Embedding(input_dim=X_train.shape[0], output_dim=9))
    model.add(keras.layers.SimpleRNN(units))

    model.add(keras.layers.Dense(y_train.shape[1], activation=activation))

    model.compile(loss="mse", optimizer="adam")
    h= model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=verbose)
    y_pred=model.predict(dishonest)
    all_pred.append(my_accuracy_old(y_pred, honest))
    h_total.append(h)
  
  return h_total, all_pred



def lstm_1(data,dishonest, honest, units=32, epochs =150, batch_size=32, verbose=0):
  all_pred=[]
  h_total=[]
  dishonest=np.reshape(dishonest, (dishonest.shape[0], 1, dishonest.shape[1]))
  for i in range(len(data)):
    X_train = data[i][0]
    X_test = data[i][1]
    y_train = data[i][2]
    y_test = data[i][3]

    trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    trainX=trainX.astype("float32")
    testX=testX.astype("float32")


    model = Sequential()
    model.add(LSTM(9, return_sequences=True))
    model.add(Dense(9))
    model.compile(loss='mse', optimizer='adam')
    h=model.fit(trainX, y_train, epochs=epochs, batch_size=batch_size,validation_data=(testX, y_test), verbose=verbose)
    y_pred=model.predict(dishonest)
    y_pred=y_pred.reshape(dishonest.shape[0], dishonest.shape[2])
    all_pred.append(my_accuracy_old(y_pred, honest))
    h_total.append(h)
  return h_total, all_pred



def lstm_2(data,dishonest, honest, units=32, epochs =150, verbose=0):
  all_pred=[]
  h_total=[]

  for i in range(len(data)):
    X_train = data[i][0]
    X_test = data[i][1]
    y_train = data[i][2]
    y_test = data[i][3]

    seq_in = X_train
    seq_out=y_train

    features=y_train.shape[1]
    n_in = features
    n_out=features
    
    seq_in = seq_in.reshape((X_train.shape[0], features, 1))
    seq_out = seq_out.reshape((X_train.shape[0], features, 1))

    # define encoder
    visible = Input(shape=(n_in,1))
    encoder = LSTM(100, activation='relu')(visible)

    # define reconstruct decoder
    decoder1 = RepeatVector(n_in)(encoder)
    decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)
    
    # define predict decoder
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(1))(decoder2)
    
    # tie it together
    model = keras.Model(inputs=visible, outputs=[decoder1, decoder2])
    model.compile(optimizer='adam', loss='mse')

    # fit model
    h=model.fit(seq_in, [seq_in,seq_out], epochs=epochs, verbose=verbose)

    testX = dishonest.reshape((dishonest.shape[0], features, 1))
    yhat = model.predict(testX)
    yhat=np.asarray(yhat)
    yhat=yhat[1].reshape(dishonest.shape)
    all_pred.append(my_accuracy_old(yhat, honest))
    h_total.append(h)
  return h_total, all_pred