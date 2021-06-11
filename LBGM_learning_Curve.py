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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit, learning_curve
import matplotlib.pyplot as plt

import numpy as np

#모델링
def modeling(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    metrics(y_test,pred)
    # print("1 " , sum(pred == 1))
    # print("0 " , sum(pred == 0))
#평가 지표
def metrics(y_test,pred):
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    print('accuracy : ', accuracy , ' precision : ', precision , ' recall : ', recall , 'f1 score ',f1)

def plot_learning_curve(estimator, x, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), s=None):
    if ylim is not None:
        plt.ylim(*ylim)
    # train_sizes : (392 + 392)의 80% 를 0.1, 0.325, 0.55, 0.775, 1의 비율로 학습시긴다.
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=s)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # 평균에 표준 편차를 +-해준 영역을 색칠한다.
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
    plt.xlabel('Training size')
    plt.ylabel('F1 Score')
    # 그림에 선 표시
    plt.grid(True)
    # 범례 표시: best - 자동으로 최적의 위치에
    plt.legend(loc="best")
    plt.show()
    

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/creditcard.csv')
df = df.sample(frac=1)
# df.Class.value_counts(normalize=True).plot(kind='bar')
# print(df.Class.value_counts(normalize=True)*100)




X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=10)

# df = df.sample(frac=1)
# fraud의 수가 492개 이므로 492개의 non_fraud를 가져온다.
# fraud_df = df.loc[df['Class'] == 1]
# non_fraud_df = df.loc[df['Class'] == 0][:492]
# pd.concat: data frame 합치기
# undersampling_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1)
# ux = undersampling_df.iloc[:,:-1]
# uy = undersampling_df.iloc[:,-1]

# ux_train, ux_test, uy_train, uy_test = train_test_split(ux,uy,test_size=0.25,random_state=0)



# print("----------- No Over Sampling ----------------")
# print("--------- LogisticRegression --------------")
# lr = LogisticRegression()
# modeling(lr,X_train,X_test,y_train,y_test)



# print("------------- Over Sampling ----------------")


# print("Test - 0", sum(y_train == 0))
# print("Test - 1", sum(y_train == 1))



# print("Test - 0", sum(y_train_over == 0))
# print("Test - 1", sum(y_train_over == 1))

# # print("--------- LGBMClassifier --------------")

# modeling(lgb,X_train_over,X_test,y_train_over,y_test)


# print("--------- LGBMClassifier - Flag : unbalance = true, boost_from_average=False --------------")
# lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,
#                      is_unbalance = True,boost_from_average=False)
# modeling(lgb,X_train_over,X_test,y_train_over,y_test)


# print("--------- LogisticRegression --------------")

# for c in [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001,]  :
#   lru = LogisticRegression(C=c, max_iter = 100000)
#   # modeling(lru,X_train_over,X_test,y_train_over,y_test)


ratio = list()
# ratio.append(0.1)
# ratio.append(0.4)
ratio.append(0.7)
ratio.append(1.0)

for r in ratio : 
  smote = SMOTE( sampling_strategy= r , random_state=0 )
  X_train_over,y_train_over = smote.fit_sample(X_train,y_train)


  lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
  cv = ShuffleSplit(n_splits=5, test_size=0.2)
  plt.figure(figsize=(10, 7))
  plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"AUC" , fontsize=14)
  plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='roc_auc')

  plt.figure(figsize=(10, 7))
  plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"F1" , fontsize=14)
  plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='f1')





