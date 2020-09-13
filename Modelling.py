# %% Import Libraries
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import PReLU
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from sklearn.metrics import roc_curve,roc_auc_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import os
import tensorboard
from datetime import datetime 
from xgboost import XGBClassifier
import xgboost as xgb
import graphviz
from sklearn import tree
print(tf.config.list_physical_devices('GPU'))


# %%
!pip install graphviz
print(tf.__version__)

# %%
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
# %% Tensorboard
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# %% Wandb Setup
#!wandb login fecf0f8ecccbce95a5bd0d2ce6b8cfb12b90bd0b --relogin
import wandb
from wandb import util
from wandb.keras import WandbCallback
wandb.init(project="nids")
config = wandb.config
config.learning_rate = 0.0006
config.batch_size = 2048
config.optimizer = 'adam'
config.epochs=300

# %% Load Dataset
df=pd.read_csv("D:/Semester5Project/datasetnorm.csv")
df.drop(columns=["Unnamed: 0"],inplace=True)

# %% Adding column
df['Label']= np.where(df['attack_cat']==0, 0, 1)

# %% Seperating Dataset into X and y
X=df.drop(columns=['attack_cat','Label'])
y=df['Label']
del df
gc.collect()
gc.collect()

# %% scale pos
print(y_train.value_counts().loc[0]/y_train.value_counts().loc[1])

# %% Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
del X
del y
gc.collect()

# %% One Hot Encoding Output
y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)

# %%
#X_train=X_train.values
#y_train=y_train.values
#X_test=X_test.values
#y_test=y_test.values
# %% Decision Tree Model Fitting
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
winsound.Beep(frequency, duration)

# %% Random Forest Classfier
model1=RandomForestClassifier(n_estimators=120,random_state=42,verbose=3,n_jobs=3,)
model1.fit(X_train,y_train)
winsound.Beep(frequency, duration)

# %% XGBoost (test binary logistic and softmax WITHOUT scale)
model=XGBClassifier(predictor='gpu_predictor',
                    objective='multi:softmax',
                    scale_pos_weight=6.89916078386075,
                    n_estimators=300,
                    max_depth=10,
                    num_class=2,
                    verbosity=3)

# %% XGBoost Fitting
model.fit(X_train,y_train,verbose=True)
trainacc=accuracy_score(y_train,model.predict(X_train))
testacc=accuracy_score(y_test,model.predict(X_test))
print(f"\nAccuracy on testing set: {testacc}")
print(f"\nAccuracy on training set: {trainacc}")
winsound.Beep(frequency, duration)

# %% XGBoost Feature Plot
plt.style.use('seaborn')
xgb.plot_importance(model, importance_type='gain',height=5.2)
fig = plt.gcf()
fig.set_size_inches(350, 200)

# %% Tree Plot
xgb.plot_tree(model, num_trees=0,rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(400, 500)
fig.savefig('abc.png')

#%% Grid Search CV
parameters_dt={'splitter':['best'],
            'max_depth':[i for i in range(6,30)],
            'random_state':[42]
            }
parameters_rfc={'max_depth':[i for i in range(8,12)],
                'n_estimators':[i for i in range(110,140,2)]}
parameters_xgb={'n_estimators': [100,250,400, 700, 1000],
                'max_depth': [i for i in range(7,25)],
                }
grid=GridSearchCV(model,parameters_xgb,verbose=10)
grid.fit(X_train,y_train)
winsound.Beep(frequency, duration)

# %% Grid Search CV Results
print(grid.best_params_)
print(grid.best_score_)
winsound.Beep(frequency, duration)
# %% Accuracy Score
trainacc=accuracy_score(y_train,model.predict(X_train))
testacc=accuracy_score(y_test,model.predict(X_test))
print(f"\nAccuracy on testing set: {testacc}")
print(f"\nAccuracy on training set: {trainacc}")
winsound.Beep(frequency, duration)

# %% Keras N.N Model
cvscores=[]
def baseline_model():
    model=models.Sequential()
    model.add(layers.Dense(420,input_shape=X_train.shape,kernel_initializer='LecunNormal'))
    model.add(layers.Dropout(0.22, noise_shape=None, seed=42))
    model.add(layers.Dense(308,activation='selu'))
    model.add(layers.Dense(188,activation='selu'))
    model.add(layers.Dropout(0.12, noise_shape=None, seed=7))
    model.add(layers.Dense(108,activation='selu'))
    model.add(layers.Dense(38,activation='selu'))
    model.add(layers.Dropout(0.095, noise_shape=None, seed=42))
    model.add(layers.Dense(2,activation='softmax'))
    return model


# %% Plotting the Model
model=baseline_model()
plot_model(model, to_file='model.png',show_shapes=True,expand_nested=True,dpi=300)

# %% K-Fold Cross Val
for train_index,test_index in KFold(3).split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    model=baseline_model()
    opt = Adam(learning_rate=0.0022)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train, y_train,epochs=30, batch_size=2048)
    cvscores.append(model.evaluate(X_test, y_test)[1])
    
# %% Normal Fitting
model=baseline_model()
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,restore_best_weights=True)
opt = Adam(learning_rate=config.learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=config.epochs,batch_size=config.batch_size,
          callbacks=[WandbCallback(),earlystop])
#Add tensorboard_callback in callbacks for graph
winsound.Beep(frequency, duration)
    


# %% CV Scores
for i in cvscores:
    print("acc - "+str(i))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# %% Plotting learning loss vs epochs
loss_values = history.history['loss']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %% Neural Model Accuracy
_, accuracy = model.evaluate(X_train, y_train)
_, accuracy1 = model.evaluate(X_test, y_test)
print(model.summary())
print('(Train) Accuracy: %.2f' % (accuracy*100))
print('(Test) Accuracy: %.2f' % (accuracy1*100))

# %% DT Model Accuracy
accuracy = accuracy_score(y_train,model.predict(X_train))
accuracy1 = accuracy_score(y_test,model.predict(X_test))
print('(Train) Accuracy: %.2f' % (accuracy*100))
print('(Test) Accuracy: %.2f' % (accuracy1*100))

# %% Garbage collecting memory
gc.collect()

# %% Classification Report
print(classification_report(y_train, model.predict(X_train)))
print()
print(classification_report(y_test, model.predict(X_test)))

# %% Dictionary for Classifying
classes={0:'Not an Attack', 
         1:'Exploits',
         2:'Reconnaissance', 
         3:'DoS',
         4:'Generic',
         5:'Shellcode',
         6:'Fuzzers',
         7:'Worms', 
         8:'Backdoors',
         9:'Analysis'}

# %% Delete stuff from memory
del [X_train,y_train,X_test,y_test]
gc.collect()

# %% ROC plots and results
#Calculate False Positive Rate and True Positive Rate for y_test
fpr1, tpr1, thresh1 = roc_curve(y_test.iloc[:,1, model.predict_proba(X_test)[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test.iloc[:,1], random_probs, pos_label=1)
    
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='red', label='Model 3')
plt.plot(p_fpr, p_tpr, linestyle='-.', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show();
    
    #Print ROC-AUC scores for both models
auc_score1 = roc_auc_score(y_test.iloc[:,1], model.predict_proba(X_test)[:,1])
    
print("ROC-AUC Score - ",auc_score1)

# %% Model Outputs
import matplotlib.pyplot as plt 
plt.style.use('seaborn') 
plt.figsize=(20,10)
xgb.plot_importance(model,height=0.2)
xgb.plot_tree(model, num_trees=2)
xgb.to_graphviz(model, num_trees=2)

'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00   1552939
           1       1.00      1.00      1.00    225091

    accuracy                           1.00   1778030
   macro avg       1.00      1.00      1.00   1778030
weighted avg       1.00      1.00      1.00   1778030


              precision    recall  f1-score   support

           0       1.00      1.00      1.00    665821
           1       0.99      0.99      0.99     96192

    accuracy                           1.00    762013
   macro avg       0.99      0.99      0.99    762013
weighted avg       1.00      1.00      1.00    762013
'''