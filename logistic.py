import pandas as pd 
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/creditcard.csv')
# df.Class.value_counts(normalize=True).plot(kind='bar')
# print(df.Class.value_counts(normalize=True)*100)


X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=10)

# print("----------- No Over Sampling ----------------")
# print("--------- LogisticRegression --------------")
# lr = LogisticRegression()
# modeling(lr,X_train,X_test,y_train,y_test)
# print("--------- LGBMClassifier --------------")
# lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
# modeling(lgb,X_train,X_test,y_train,y_test)


# print("--------- LGBMClassifier - Flag : unbalance = true, boost_from_average=False --------------")
# lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,
#                      is_unbalance = True,boost_from_average=False)
# modeling(lgb,X_train,X_test,y_train,y_test)


print("------------- Over Sampling ----------------")
smote = SMOTE(random_state=0)
X_train_over,y_train_over = smote.fit_sample(X_train,y_train)

overSampling = DataFrame(X_train_over,y_train_over)
overSampling.plot(kind='bar')
# print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
# print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
# print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
# print("--------- LogisticRegression --------------")
# lr = LogisticRegression()
# modeling(lr,X_train_over,X_test,y_train_over,y_test)
# print("--------- LGBMClassifier --------------")
# lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
# modeling(lgb,X_train_over,X_test,y_train_over,y_test)


#모델링
def modeling(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    metrics(y_test,pred)
  
#평가 지표
def metrics(y_test,pred):
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred,average='macro')
    print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy,precision,recall))
    print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1,roc_score,recall))
