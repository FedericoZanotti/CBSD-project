import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import reconstruction
from reconstruction import my_accuracy_old



def create_undercomplete_AE(data, epochs=300, batch_size=64, verbose=0):
  X_train = data[0]
  X_test = data[1]
  y_train = data[2]
  y_test = data[3]

  # This is the size of our encoded representations
  encoding_dim = 1  

  input = keras.Input(shape=(9,))
  # "encoded" is the encoded representation of the input

  x = layers.Dense(18, activation='tanh')(input)
  x=layers.Dropout(0.4)(x)  
  x = layers.Dense(36, activation='tanh')(x)
  x=layers.Dropout(0.4)(x)  


  encoded = layers.Dense(encoding_dim, activation='relu')(x)


  y = layers.Dense(36, activation='tanh')(encoded)
  x=layers.Dropout(0.4)(y)  
  y = layers.Dense(18, activation='tanh')(y)
  x=layers.Dropout(0.4)(y)  


  # "decoded" is the lossy reconstruction of the input
  decoded = layers.Dense(9, activation='linear')(y)
  autoencoder = keras.Model(input, decoded)

  autoencoder.compile(optimizer='adam', loss='mean_squared_error')
  h=autoencoder.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_data=(X_test, y_test), verbose=verbose)
  return h, autoencoder

  

encoding_dim = 144  

def create_overcomplete_AE(data, epochs=50, batch_size=16, verbose=0):
  X_train = data[0]
  X_test = data[1]
  y_train = data[2]
  y_test = data[3]  
  encoding_dim=144
  input = keras.Input(shape=(X_train.shape[1],))
  x = layers.Dense(9, activation='tanh')(input)
  x = layers.Dense(18, activation='tanh')(x)
  x = layers.Dense(36, activation='tanh')(x)
  x = layers.Dense(72, activation='tanh')(x)


  encoded = layers.Dense(encoding_dim, activation='relu')(x)


  y = layers.Dense(72, activation='tanh')(encoded)
  y = layers.Dense(36, activation='tanh')(y)
  y = layers.Dense(18, activation='tanh')(y)
  y = layers.Dense(9, activation='tanh')(y)



  decoded = layers.Dense(9, activation='linear')(y)

  autoencoder = keras.Model(input, decoded)
  encoder = keras.Model(input, encoded)
  autoencoder.compile(optimizer='adam', loss='mean_squared_error')
  h=autoencoder.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, y_test), verbose=verbose)
  return h, autoencoder

def get_history_and_prediction_AE(data, dishonest, honest, keys, undercomplete=True, epochs=300, batch_size=64, verbose=0):
  h_total=[]
  pred_total = []
  if undercomplete:
    print("Calculating for UNDERCOMPLETE Autoencoder...")
    for i in range(len(data)):
      h, m=create_undercomplete_AE(data[i], epochs, batch_size, verbose)
      y_pred=m.predict(dishonest)
      pred_total.append(my_accuracy_old(y_pred, honest))
      h_total.append(h)
      print(f"... Done for {keys[i]}... Missing {len(keys)-(i+1)} set of data")
  else:
    print("Calculating for OVERCOMPLETE Autoencoder...")
    for i in range(len(data)):
      h, m=create_overcomplete_AE(data[i], epochs, batch_size, verbose)
      y_pred=m.predict(dishonest)
      pred_total.append(my_accuracy_old(y_pred, honest))
      h_total.append(h)
      print(f"... Done for {keys[i]}... Missing {len(keys)-(i+1)} set of data")


  return h_total, pred_total

def plot_all(h_total,keys):
  plt.figure(figsize=(10, 10))
  for i in range(len(h_total)):
    ax = plt.subplot(2, 3, i + 1)
    ax.plot(h_total[i].history['loss'])
    ax.plot(h_total[i].history['val_loss'])
    ax.get_legend()
    ax.set_title(f"Autoencoder {keys[i]}")