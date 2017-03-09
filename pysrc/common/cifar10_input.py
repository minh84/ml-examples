import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def load_flatten_CIFAR10(ROOT, num_training = 49000, num_test = 5000):
  X_train, y_train, X_test, y_test = load_CIFAR10(ROOT)

  # let's divide train into training set (49000) + validation set (1000)
  mask = range(num_training, X_train.shape[0])
  X_val = X_train[mask]
  y_val = y_train[mask]

  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]

  # flatten input data
  X_train = X_train.reshape(X_train.shape[0], -1)
  X_val = X_val.reshape(X_val.shape[0], -1)
  X_test = X_test.reshape(X_test.shape[0], -1)

  # normalize training data
  mean_images = np.mean(X_train, axis=0)
  X_train -= mean_images
  X_val -= mean_images
  X_test -= mean_images

  # return a dictionary of dataset
  data = {'X_train' : X_train, 'y_train' : y_train, 'mean_images' : mean_images
         ,'X_val'   : X_val,   'y_val'   : y_val
         ,'X_test'  : X_test,  'y_test'  : y_test}
  return data

