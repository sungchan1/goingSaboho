import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV, ShuffleSplit, learning_curve, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.optimizers import Adam

train_df = pd.read_csv('creditcard.csv')
X = train_df.drop(columns={'Class'})
y = train_df['Class']


from sklearn.model_selection import train_test_split
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_test = y_test.ravel()
y_train = y_train.ravel()

X.info()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,14))
corr = X.corr()
sns.heatmap(corr)

y.value_counts()

y.describe()

fraud = train_df[train_df['Class'] == 1]
valid = train_df[train_df['Class'] == 0]

print("Fraud transaction statistics")
print(fraud["Amount"].describe())
print("\nNormal transaction statistics")
print(valid["Amount"].describe())

# describes info about train and test set
print("X_train dataset: ", X_train.shape)
print("y_train dataset: ", y_train.shape)
print("X_test dataset: ", X_test.shape)
print("y_test dataset: ", y_test.shape)

print("before applying smote:",format(sum(y_train == 1)))
print("before applying smote:",format(sum(y_train == 0)))

# import SMOTE module from imblearn library
# pip install imblearn if you don't have
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2, sampling_strategy= 0.7)
X_train, y_train = sm.fit_resample(X_train, y_train)

print('After applying smote X_train: {}\n'.format(X_train.shape))
print('After applying smote y_train: {}\n'.format(y_train.shape))

print("After applying smote label '1': {}\n".format(sum(y_train == 1)))
print("After applying smote label '0': {}\n".format(sum(y_train == 0)))

X_train = X_train.reshape(X_train.shape[0] , X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0] , X_test.shape[1],1)

X_train.shape , X_test.shape

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# Initialising the CNN
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Convolution1D(32 , 2 , activation='relu',input_shape=X_train[0].shape))
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Convolution1D(64 , 2 , activation='relu'))
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Convolution1D(128 , 2 , activation='relu'))
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(units=256, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer=Adam(lr = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()


metrics = [
    'acc',
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]

classifier.compile(optimizer= Adam(learning_rate=0.001,),
              loss='binary_crossentropy',
              metrics=metrics,
              )

ep = 100
bs = 512
history = classifier.fit(X_train, y_train, epochs=ep, batch_size=bs, validation_data=(X_test, y_test),)
score = classifier.evaluate(X_test, y_test, verbose=0)

x_range = range(1,101)

plt.figure(figsize=(10,10))
plt.plot(x_range, history.history['loss'], label='Training Loss')
plt.plot(x_range, history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Error')
# 그림에 선 표시
plt.grid(True)
# 범례 표시: best - 자동으로 최적의 위치에
plt.legend(loc="best")
plt.show()
plt.savefig('CNN.png')

# history = classifier.fit(X_train, y_train, batch_size = 100, epochs = 10 , validation_data=(X_test,y_test),verbose=1)


# Predicting the Test set results
y_pred = classifier.predict(X_test).flatten().round()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True)
#find accuracy
from sklearn.metrics import accuracy_score
print('CNN:',accuracy_score(y_test,y_pred))
# find classification report
from sklearn.metrics import f1_score , precision_score , recall_score , classification_report


print('classification_report:',classification_report(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))
print('precision_score:',precision_score(y_test,y_pred))
print('recall_score:',recall_score(y_test,y_pred))