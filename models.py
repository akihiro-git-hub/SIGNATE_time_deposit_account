import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from keras.layers import Dense,Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso,Ridge
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import ReLU, PReLU
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

class Model1xgb:

  def __init__(self):
    self.model = None
    self.scaler = None

  def fit (self, tr_x, tr_y, va_x, va_y):
    params = {'booster':'gbtree',
          'objective':'binary:logistic',
          'silent':1,
          'random_state':71,
          'eta': 0.1,}
    num_round = 500
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    tr_x = self.scaler.transform(tr_x)
    va_x = self.scaler.transform(va_x)
    d_train = xgb.DMatrix(tr_x, label = tr_y)
    d_valid = xgb.DMatrix(va_x, label = va_y)
    watchlist = [(d_train, 'train'),(d_valid, 'eval')]
    self.model = xgb.train(params, d_train, num_round, evals = watchlist,early_stopping_rounds=20)

  def predict(self,x):
    x = self.scaler.transform(x)
    data = xgb.DMatrix(x)
    pred = self.model.predict(data)
    return pred


class Model1xgb2:

  def __init__(self):
    self.model = None
    self.scaler = None

  def fit (self, tr_x, tr_y, va_x, va_y):
    params = {'booster':'gbtree',
          'objective':'binary:logistic',
          'silent':1,
          'random_state':71,
          'eta': 0.1,
          'alpha': 0.0003595462667444797,'colsample_bytree': 0.8,
          'gamma': 0.03280399314969842,'lambda': 0.35199860793869975,
         'max_depth': 7,'min_child_weight': 5.189327191467295,
         'subsample': 0.8500000000000001}
    num_round = 500
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    tr_x = self.scaler.transform(tr_x)
    va_x = self.scaler.transform(va_x)
    d_train = xgb.DMatrix(tr_x, label = tr_y)
    d_valid = xgb.DMatrix(va_x, label = va_y)
    watchlist = [(d_train, 'train'),(d_valid, 'eval')]
    self.model = xgb.train(params, d_train, num_round, evals = watchlist,early_stopping_rounds=20)

  def predict(self,x):
    x = self.scaler.transform(x)
    data = xgb.DMatrix(x)
    pred = self.model.predict(data)
    return pred


class Model1NNproba:
  def __init__(self):
    self.model = None
    self.scaler =None
  
  def fit(self, tr_x, tr_y, va_x, va_y):
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    batch_size = 128
    epochs = 100
    tr_x = self.scaler.transform(tr_x)
    va_x = self.scaler.transform(va_x)

    model = Sequential()
    model.add(Dense(256, activation='relu',input_shape=(tr_x.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    optimizer='adam',metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(tr_x,tr_y,
    batch_size=batch_size, epochs=epochs,verbose=1,validation_data = (va_x,va_y),
    callbacks=[early_stopping])
    self.model = model

  def predict(self,x):
    x = self.scaler.transform(x)
    y_pred = self.model.predict_proba(x).reshape(-1)
    return y_pred




class Model1NN2proba:
  def __init__(self):
    self.model = None
    self.scaler =None
  
  def fit(self, tr_x, tr_y, va_x, va_y):
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    batch_size = 128
    epochs = 300
    tr_x = self.scaler.transform(tr_x)
    va_x = self.scaler.transform(va_x)
    model = Sequential()
    model.add(Dropout(0.05, input_shape=(tr_x.shape[1],)))
    model.add(Dense(96,activation=PReLU()))
    model.add(Dropout(0.1))
    model.add(Dense(96,activation=PReLU()))
    model.add(Dropout(0.1))
    model.add(Dense(96,activation=PReLU()))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy',
             optimizer = SGD(lr=0.0016590360920686548, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(tr_x,tr_y,
    batch_size=batch_size, epochs=epochs,verbose=1,validation_data = (va_x,va_y),
    callbacks=[early_stopping])
    self.model = model


  def predict(self,x):
    self.scaler.transform(x)
    y_pred = self.model.predict_proba(x).reshape(-1)
    return y_pred


    

class Model1ramdom:

  def __init__(self):
    self.model = None
    self.scaler = None

  def fit(self, tr_x, tr_y, va_x, va_y):
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    tr_x = self.scaler.transform(tr_x)
    self.model = RandomForestRegressor(random_state=42)
    self.model.fit(tr_x, tr_y)

  def predict(self, x):
    x = self.scaler.transform(x)
    pred = self.model.predict(x)
    return pred


class Model2NN2:
  def __init__(self):
    self.model = None
    self.scaler =None
  
  def fit(self, tr_x, tr_y, va_x, va_y):
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    batch_size = 128
    epochs = 300
    tr_x = self.scaler.transform(tr_x)
    va_x = self.scaler.transform(va_x)
    model = Sequential()
    model.add(Dropout(0.05, input_shape=(tr_x.shape[1],)))
    model.add(Dense(96,activation=PReLU()))
    model.add(Dropout(0.1))
    model.add(Dense(96,activation=PReLU()))
    model.add(Dropout(0.1))
    model.add(Dense(96,activation=PReLU()))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy',
             optimizer = SGD(lr=0.0016590360920686548, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(tr_x,tr_y,
    batch_size=batch_size, epochs=epochs,verbose=1,validation_data = (va_x,va_y),
    callbacks=[early_stopping])
    self.model = model


  def predict(self,x):
    self.scaler.transform(x)
    y_pred = self.model.predict_proba(x).reshape(-1)
    return y_pred




class Model2KMeans:
  def __init__(self):
    self.model = None
    self.scaler = None

  def fit(self, tr_x, tr_y, va_x, va_y):
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    tr_x = self.scaler.transform(tr_x)
    tr_x = pd.DataFrame(tr_x)

    cust_array = []
    for i in range (tr_x.shape[1]):
      list_s = tr_x.iloc[:,1].tolist()
      cust_array.append(list_s)


    cust_array = np.array(cust_array)
    cust_array = cust_array.T
    self.model = KMeans(n_clusters = 2)
    self.model.fit(cust_array)
    
  def predict(self,x):
    x = self.scaler.transform(x)
    x = pd.DataFrame(x)
    cust_test = []
    for c in range(x.shape[1]):
      list_s = x.iloc[:,c].tolist()
      cust_test.append(list_s)
    

    cust_test = np.array(cust_test)
    cust_test = cust_test.T
    pred = self.model.fit_predict(x)
    return pred




class Model2KNN:

  def __init__(self):
    self.model = None
    self.scaler = None

  def fit(self, tr_x, tr_y, va_x, va_y):
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    tr_x = self.scaler.transform(tr_x)
    self.model = KNeighborsClassifier(n_neighbors=32)
    self.model.fit(tr_x, tr_y)

  def predict(self,x):
    self.scaler.transform(x)
    pred = self.model.predict(x)
    return pred


class Model3logistic:

  def __init__(self):
    self.model = None
    self.scaler = None

  def fit(self, tr_x, tr_y, va_x, va_y):
    self.scaler = StandardScaler()
    self.scaler.fit(tr_x)
    tr_x = self.scaler.transform(tr_x)
    self.model = Ridge()
    self.model.fit(tr_x,tr_y)

  def predict(self,x):
    x = self.scaler.transform(x)
    pred = self.model.predict(x)
    return pred

    