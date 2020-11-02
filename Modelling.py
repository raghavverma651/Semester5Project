<<<<<<< HEAD
# %% Import Libraries
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from sklearn.metrics import roc_curve,roc_auc_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import os
import tensorboard
from datetime import datetime 
from xgboost import XGBClassifier
import xgboost as xgb
import graphviz
from sklearn import tree
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical 
import tensorflow as tf
import numpy as np
import pickle
print(tf.config.list_physical_devices('GPU'))

# %%

# %%
import winsound
frequency = 400  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
# %% Tensorboard
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# %% Results Pickle
everything=pd.read_pickle(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle')
print(everything)
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
# %% Wandb Setup
#!wandb login fecf0f8ecccbce95a5bd0d2ce6b8cfb12b90bd0b --relogin
import wandb
from wandb import util
from wandb.keras import WandbCallback
wandb.init(project="nids")
config = wandb.config
config.learning_rate = 0.0012
config.batch_size = 2048
config.optimizer = 'adam'
config.epochs=200

# %% Load Dataset
df=pd.read_csv("D:/Semester5Project/datasetnorm.csv")
df.drop(columns=["Unnamed: 0"],inplace=True)
df.drop(index=df[df.duplicated()].index,inplace=True)
cols=list(df.columns)

# %% Imputing
imp=SimpleImputer()
df=imp.fit_transform(df)
df=pd.DataFrame(df,columns=cols)

# %% Seperating Dataset into X and y
X=df.drop(columns=['Label'])
y=df['Label']
del df
gc.collect()
gc.collect()

# %% Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
del X
del y
gc.collect()

# %% Logistic Regression
model=LogisticRegression(solver='newton-cg',max_iter=300,verbose=6,n_jobs=3)
model.fit(X_train,y_train)
y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)
everything.loc['Logistic Regression']['Training Accuracy']=accuracy_score(y_train,y_pred_train)*100
everything.loc['Logistic Regression']['Testing Accuracy']=accuracy_score(y_test,y_pred_test)*100
train=confusion_matrix(y_train,y_pred_train)
test=confusion_matrix(y_test,y_pred_test)
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['Logistic Regression']['Training DR']=(tp2/(tp2+fn2))
everything.loc['Logistic Regression']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['Logistic Regression']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['Logistic Regression']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,y_pred_train,output_dict=True)
crtest=classification_report(y_test,y_pred_test,output_dict=True)
everything.loc['Logistic Regression']['Precision for No Attack (Train)']=crtrain['0.0']['precision']
everything.loc['Logistic Regression']['Precision for No Attack (Test)']=crtest['0.0']['precision']
everything.loc['Logistic Regression']['Recall for No Attack (Train)']=crtrain['0.0']['recall']
everything.loc['Logistic Regression']['Recall for No Attack (Test)']=crtest['0.0']['recall']
everything.loc['Logistic Regression']['F1 Score for No Attack (Train)']=crtrain['0.0']['f1-score']
everything.loc['Logistic Regression']['F1 Score for No Attack (Test)']=crtest['0.0']['f1-score']
everything.loc['Logistic Regression']['Precision for Attack (Train)']=crtrain['1.0']['precision']
everything.loc['Logistic Regression']['Precision for Attack (Test)']=crtest['1.0']['precision']
everything.loc['Logistic Regression']['Recall for Attack (Train)']=crtrain['1.0']['recall']
everything.loc['Logistic Regression']['Recall for Attack (Test)']=crtest['1.0']['recall']
everything.loc['Logistic Regression']['F1 Score for Attack (Train)']=crtrain['1.0']['f1-score']
everything.loc['Logistic Regression']['F1 Score for Attack (Test)']=crtest['1.0']['f1-score']
print(everything.loc['Logistic Regression'])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Decision Tree Model Fitting
model=DecisionTreeClassifier(max_depth=9,random_state=42)
model.fit(X_train,y_train)
everything.loc['Decision Tree']['Training Accuracy']=accuracy_score(y_train,model.predict(X_train))*100
everything.loc['Decision Tree']['Testing Accuracy']=accuracy_score(y_test,model.predict(X_test))*100
train=confusion_matrix(y_train,model.predict(X_train))
test=confusion_matrix(y_test,model.predict(X_test))
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['Decision Tree']['Training DR']=(tp2/(tp2+fn2))
everything.loc['Decision Tree']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['Decision Tree']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['Decision Tree']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,model.predict(X_train),output_dict=True)
crtest=classification_report(y_test,model.predict(X_test),output_dict=True)
everything.loc['Decision Tree']['Precision for No Attack (Train)']=crtrain['0.0']['precision']
everything.loc['Decision Tree']['Precision for No Attack (Test)']=crtest['0.0']['precision']
everything.loc['Decision Tree']['Recall for No Attack (Train)']=crtrain['0.0']['recall']
everything.loc['Decision Tree']['Recall for No Attack (Test)']=crtest['0.0']['recall']
everything.loc['Decision Tree']['F1 Score for No Attack (Train)']=crtrain['0.0']['f1-score']
everything.loc['Decision Tree']['F1 Score for No Attack (Test)']=crtest['0.0']['f1-score']
everything.loc['Decision Tree']['Precision for Attack (Train)']=crtrain['1.0']['precision']
everything.loc['Decision Tree']['Precision for Attack (Test)']=crtest['1.0']['precision']
everything.loc['Decision Tree']['Recall for Attack (Train)']=crtrain['1.0']['recall']
everything.loc['Decision Tree']['Recall for Attack (Test)']=crtest['1.0']['recall']
everything.loc['Decision Tree']['F1 Score for Attack (Train)']=crtrain['1.0']['f1-score']
everything.loc['Decision Tree']['F1 Score for Attack (Test)']=crtest['1.0']['f1-score']
with open('results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Random Forest Classfier
model=RandomForestClassifier(n_estimators=300,random_state=42,verbose=3,n_jobs=3)
model.fit(X_train,y_train)
everything.loc['Random Forest Classifier']['Training Accuracy']=accuracy_score(y_train,model.predict(X_train))*100
everything.loc['Random Forest Classifier']['Testing Accuracy']=accuracy_score(y_test,model.predict(X_test))*100
train=confusion_matrix(y_train,model.predict(X_train))
test=confusion_matrix(y_test,model.predict(X_test))
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['Random Forest Classifier']['Training DR']=(tp2/(tp2+fn2))
everything.loc['Random Forest Classifier']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['Random Forest Classifier']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['Random Forest Classifier']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,model.predict(X_train),output_dict=True)
crtest=classification_report(y_test,model.predict(X_test),output_dict=True)
everything.loc['Random Forest Classifier']['Precision for No Attack (Train)']=crtrain['0.0']['precision']
everything.loc['Random Forest Classifier']['Precision for No Attack (Test)']=crtest['0.0']['precision']
everything.loc['Random Forest Classifier']['Recall for No Attack (Train)']=crtrain['0.0']['recall']
everything.loc['Random Forest Classifier']['Recall for No Attack (Test)']=crtest['0.0']['recall']
everything.loc['Random Forest Classifier']['F1 Score for No Attack (Train)']=crtrain['0.0']['f1-score']
everything.loc['Random Forest Classifier']['F1 Score for No Attack (Test)']=crtest['0.0']['f1-score']
everything.loc['Random Forest Classifier']['Precision for Attack (Train)']=crtrain['1.0']['precision']
everything.loc['Random Forest Classifier']['Precision for Attack (Test)']=crtest['1.0']['precision']
everything.loc['Random Forest Classifier']['Recall for Attack (Train)']=crtrain['1.0']['recall']
everything.loc['Random Forest Classifier']['Recall for Attack (Test)']=crtest['1.0']['recall']
everything.loc['Random Forest Classifier']['F1 Score for Attack (Train)']=crtrain['1.0']['f1-score']
everything.loc['Random Forest Classifier']['F1 Score for Attack (Test)']=crtest['1.0']['f1-score']
print(everything.loc['Random Forest Classifier'])
with open('results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Naive Bayes Classifier
model=GaussianNB()
model.fit(X_train,y_train)
y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)
everything.loc['Naive Bayes Classifier']['Training Accuracy']=accuracy_score(y_train,y_pred_train)*100
everything.loc['Naive Bayes Classifier']['Testing Accuracy']=accuracy_score(y_test,y_pred_test)*100
train=confusion_matrix(y_train,y_pred_train)
test=confusion_matrix(y_test,y_pred_test)
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['Naive Bayes Classifier']['Training DR']=(tp2/(tp2+fn2))
everything.loc['Naive Bayes Classifier']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['Naive Bayes Classifier']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['Naive Bayes Classifier']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,y_pred_train,output_dict=True)
crtest=classification_report(y_test,y_pred_test,output_dict=True)
everything.loc['Naive Bayes Classifier']['Precision for No Attack (Train)']=crtrain['0.0']['precision']
everything.loc['Naive Bayes Classifier']['Precision for No Attack (Test)']=crtest['0.0']['precision']
everything.loc['Naive Bayes Classifier']['Recall for No Attack (Train)']=crtrain['0.0']['recall']
everything.loc['Naive Bayes Classifier']['Recall for No Attack (Test)']=crtest['0.0']['recall']
everything.loc['Naive Bayes Classifier']['F1 Score for No Attack (Train)']=crtrain['0.0']['f1-score']
everything.loc['Naive Bayes Classifier']['F1 Score for No Attack (Test)']=crtest['0.0']['f1-score']
everything.loc['Naive Bayes Classifier']['Precision for Attack (Train)']=crtrain['1.0']['precision']
everything.loc['Naive Bayes Classifier']['Precision for Attack (Test)']=crtest['1.0']['precision']
everything.loc['Naive Bayes Classifier']['Recall for Attack (Train)']=crtrain['1.0']['recall']
everything.loc['Naive Bayes Classifier']['Recall for Attack (Test)']=crtest['1.0']['recall']
everything.loc['Naive Bayes Classifier']['F1 Score for Attack (Train)']=crtrain['1.0']['f1-score']
everything.loc['Naive Bayes Classifier']['F1 Score for Attack (Test)']=crtest['1.0']['f1-score']
print(everything.loc['Naive Bayes Classifier'])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Adaboost
estimator=RandomForestClassifier(n_estimators=3,random_state=42,verbose=3,n_jobs=3)
model=AdaBoostClassifier(base_estimator=estimator,n_estimators=200,random_state=42)
model.fit(X_train,y_train)
y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)
everything.loc['AdaBoost Classifier']['Training Accuracy']=accuracy_score(y_train,y_pred_train)*100
everything.loc['AdaBoost Classifier']['Testing Accuracy']=accuracy_score(y_test,y_pred_test)*100
train=confusion_matrix(y_train,y_pred_train)
test=confusion_matrix(y_test,y_pred_test)
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['AdaBoost Classifier']['Training DR']=(tp2/(tp2+fn2))
everything.loc['AdaBoost Classifier']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['AdaBoost Classifier']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['AdaBoost Classifier']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,y_pred_train,output_dict=True)
crtest=classification_report(y_test,y_pred_test,output_dict=True)
everything.loc['AdaBoost Classifier']['Precision for No Attack (Train)']=crtrain['0.0']['precision']
everything.loc['AdaBoost Classifier']['Precision for No Attack (Test)']=crtest['0.0']['precision']
everything.loc['AdaBoost Classifier']['Recall for No Attack (Train)']=crtrain['0.0']['recall']
everything.loc['AdaBoost Classifier']['Recall for No Attack (Test)']=crtest['0.0']['recall']
everything.loc['AdaBoost Classifier']['F1 Score for No Attack (Train)']=crtrain['0.0']['f1-score']
everything.loc['AdaBoost Classifier']['F1 Score for No Attack (Test)']=crtest['0.0']['f1-score']
everything.loc['AdaBoost Classifier']['Precision for Attack (Train)']=crtrain['1.0']['precision']
everything.loc['AdaBoost Classifier']['Precision for Attack (Test)']=crtest['1.0']['precision']
everything.loc['AdaBoost Classifier']['Recall for Attack (Train)']=crtrain['1.0']['recall']
everything.loc['AdaBoost Classifier']['Recall for Attack (Test)']=crtest['1.0']['recall']
everything.loc['AdaBoost Classifier']['F1 Score for Attack (Train)']=crtrain['1.0']['f1-score']
everything.loc['AdaBoost Classifier']['F1 Score for Attack (Test)']=crtest['1.0']['f1-score']
print(everything.loc['AdaBoost Classifier'])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)



# %% XGBoost 
model=XGBClassifier(predictor='gpu_predictor',
                    objective='multi:softmax',
                    n_estimators=350,
                    scale_pos_weight=20,
                    max_depth=9,
                    num_class=2,
                    verbosity=3)

# %% XGBoost Fitting
model.fit(X_train,y_train,verbose=True)
everything.loc['XGBoost']['Training Accuracy']=accuracy_score(y_train,model.predict(X_train))*100
everything.loc['XGBoost']['Testing Accuracy']=accuracy_score(y_test,model.predict(X_test))*100
train=confusion_matrix(y_train,model.predict(X_train))
test=confusion_matrix(y_test,model.predict(X_test))
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['XGBoost']['Training DR']=(tp2/(tp2+fn2))
everything.loc['XGBoost']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['XGBoost']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['XGBoost']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,model.predict(X_train),output_dict=True)
crtest=classification_report(y_test,model.predict(X_test),output_dict=True)
everything.loc['XGBoost']['Precision for No Attack (Train)']=crtrain['0.0']['precision']
everything.loc['XGBoost']['Precision for No Attack (Test)']=crtest['0.0']['precision']
everything.loc['XGBoost']['Recall for No Attack (Train)']=crtrain['0.0']['recall']
everything.loc['XGBoost']['Recall for No Attack (Test)']=crtest['0.0']['recall']
everything.loc['XGBoost']['F1 Score for No Attack (Train)']=crtrain['0.0']['f1-score']
everything.loc['XGBoost']['F1 Score for No Attack (Test)']=crtest['0.0']['f1-score']
everything.loc['XGBoost']['Precision for Attack (Train)']=crtrain['1.0']['precision']
everything.loc['XGBoost']['Precision for Attack (Test)']=crtest['1.0']['precision']
everything.loc['XGBoost']['Recall for Attack (Train)']=crtrain['1.0']['recall']
everything.loc['XGBoost']['Recall for Attack (Test)']=crtest['1.0']['recall']
everything.loc['XGBoost']['F1 Score for Attack (Train)']=crtrain['1.0']['f1-score']
everything.loc['XGBoost']['F1 Score for Attack (Test)']=crtest['1.0']['f1-score']
print(everything.loc['XGBoost'])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% SVM Fitting
model=SGDClassifier(loss='squared_hinge',n_jobs=3,random_state=42)
model.fit(X_train,y_train)
everything.loc['SVM']['Training Accuracy']=accuracy_score(y_train,model.predict(X_train))*100
everything.loc['SVM']['Testing Accuracy']=accuracy_score(y_test,model.predict(X_test))*100
train=confusion_matrix(y_train,model.predict(X_train))
test=confusion_matrix(y_test,model.predict(X_test))
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['SVM']['Training DR']=(tp2/(tp2+fn2))
everything.loc['SVM']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['SVM']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['SVM']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,model.predict(X_train),output_dict=True)
crtest=classification_report(y_test,model.predict(X_test),output_dict=True)
everything.loc['SVM']['Precision for No Attack (Train)']=crtrain['0']['precision']
everything.loc['SVM']['Precision for No Attack (Test)']=crtest['0']['precision']
everything.loc['SVM']['Recall for No Attack (Train)']=crtrain['0']['recall']
everything.loc['SVM']['Recall for No Attack (Test)']=crtest['0']['recall']
everything.loc['SVM']['F1 Score for No Attack (Train)']=crtrain['0']['f1-score']
everything.loc['SVM']['F1 Score for No Attack (Test)']=crtest['0']['f1-score']
everything.loc['SVM']['Precision for Attack (Train)']=crtrain['1']['precision']
everything.loc['SVM']['Precision for Attack (Test)']=crtest['1']['precision']
everything.loc['SVM']['Recall for Attack (Train)']=crtrain['1']['recall']
everything.loc['SVM']['Recall for Attack (Test)']=crtest['1']['recall']
everything.loc['SVM']['F1 Score for Attack (Train)']=crtrain['1']['f1-score']
everything.loc['SVM']['F1 Score for Attack (Test)']=crtest['1']['f1-score']
print(everything.loc['SVM'])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)


# %% RFECV
dt=DecisionTreeClassifier(max_depth=10,random_state=42)
rfecv=RFECV(estimator=dt, step=1, cv=10, scoring='accuracy')
rfecv=rfecv.fit(X_train,y_train)
X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)
dt.fit(X_train_rfecv,y_train)
winsound.Beep(frequency, duration)
print(accuracy_score(y_test,dt.predict(X_test)))

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


# %% Keras N.N Model
def baseline_model():
    model=models.Sequential()
    model.add(layers.Dense(140,input_shape=X_train.shape,kernel_initializer='LecunNormal'))
    model.add(layers.Dropout(0.22, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.17, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.13, noise_shape=None, seed=7))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.095, noise_shape=None, seed=42))
    model.add(layers.Dense(140,activation='selu'))
    model.add(layers.Dropout(0.06, noise_shape=None, seed=42))
    model.add(layers.Dense(2,activation='softmax'))
    return model

y_train = to_categorical(y_train, dtype ="uint8") 
y_test = to_categorical(y_test, dtype ="uint8") 
model=baseline_model()
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12,restore_best_weights=True)
opt = Adamax(learning_rate=0.0012)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=250,batch_size=2048,
          callbacks=[earlystop])
#Add tensorboard_callback in callbacks for graph
y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)
y_train=np.argmax(y_train, axis=1)
y_test=np.argmax(y_test, axis=1)
y_pred_train=np.argmax(y_pred_train, axis=1)
y_pred_test=np.argmax(y_pred_test, axis=1)
trainacc=(model.evaluate(X_train,to_categorical(y_train))[1])
testacc=(model.evaluate(X_test,to_categorical(y_test))[1])
everything.loc['Neural Network']['Training Accuracy']=trainacc*100
everything.loc['Neural Network']['Testing Accuracy']=testacc*100
train=confusion_matrix(y_train,y_pred_train)
test=confusion_matrix(y_test,y_pred_test)
tn1=train[0][0]
fp1=train[0][1]
tp1=train[1][1]
fn1=train[1][0]
tn2=test[0][0]
fp2=test[0][1]
tp2=test[1][1]
fn2=test[1][0]
everything.loc['Neural Network']['Training DR']=(tp2/(tp2+fn2))
everything.loc['Neural Network']['Training FAR']=((fp2+fn2)/(fp2+tp2+fn2+tn2))
everything.loc['Neural Network']['Testing DR']=(tp1/(tp1+fn1))
everything.loc['Neural Network']['Testing FAR']=((fp1+fn1)/(fp1+tp1+fn1+tn1))
crtrain=classification_report(y_train,y_pred_train,output_dict=True)
crtest=classification_report(y_test,y_pred_test,output_dict=True)
everything.loc['Neural Network']['Precision for No Attack (Train)']=crtrain['0']['precision']
everything.loc['Neural Network']['Precision for No Attack (Test)']=crtest['0']['precision']
everything.loc['Neural Network']['Recall for No Attack (Train)']=crtrain['0']['recall']
verything.loc['Neural Network']['Recall for No Attack (Test)']=crtest['0']['recall']
everything.loc['Neural Network']['F1 Score for No Attack (Train)']=crtrain['0']['f1-score']
everything.loc['Neural Network']['F1 Score for No Attack (Test)']=crtest['0']['f1-score']
everything.loc['Neural Network']['Precision for Attack (Train)']=crtrain['1']['precision']
everything.loc['Neural Network']['Precision for Attack (Test)']=crtest['1']['precision']
everything.loc['Neural Network']['Recall for Attack (Train)']=crtrain['1']['recall']
everything.loc['Neural Network']['Recall for Attack (Test)']=crtest['1']['recall']
everything.loc['Neural Network']['F1 Score for Attack (Train)']=crtrain['1']['f1-score']
everything.loc['Neural Network']['F1 Score for Attack (Test)']=crtest['1']['f1-score']
print(everything.loc['Neural Network'])
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\results.pickle','wb') as f:
    pickle.dump(everything,f)
winsound.Beep(frequency, duration)

# %% Neural Model Accuracy
_, accuracy = model3.evaluate(X_train, y_train)
_, accuracy1 = model3.evaluate(X_test, y_test)
print(model.summary())
print('(Train) Accuracy: %.2f' % (accuracy*100))
print('(Test) Accuracy: %.2f' % (accuracy1*100))


# %% Classification Report
print(classification_report(y_train, model2.predict(X_train)))
print()
print(classification_report(y_test, model2.predict(X_test)))


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

# %% Save the XGB Model
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\model.pickle','wb') as f:
    pickle.dump(model,f)
    
# %% Load the XGB Model
with open(r'C:\Users\RAGHAV VERMA\Semester5Project\model.pickle','wb') as f:
    model=pickle.load(f)