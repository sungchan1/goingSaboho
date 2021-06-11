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

pricision_list = list()
recall_list = list ()
f1_score_list = list()

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
    pricision_list.append(precision) 
    recall_list.append(recall)
    f1_score_list.append(f1)

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
    plt.ylabel('neg_mean_squared_error')
    # 그림에 선 표시
    plt.grid(True)
    # 범례 표시: best - 자동으로 최적의 위치에
    plt.legend(loc="best")
    plt.show()



df = pd.read_csv('creditcard.csv')
df = df.sample(frac=1)
# df.Class.value_counts(normalize=True).plot(kind='bar')
# print(df.Class.value_counts(normalize=True)*100)




X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=10)


smote = SMOTE( sampling_strategy= 0.7 , random_state=0 )
X_train_over,y_train_over = smote.fit_resample(X_train,y_train)

depth = [20, 30, 40, 50, -1]
n_estimators = [100,200,300,400,500,600,700,800,900,1000]
leaves = range(2,64)
for leaf in leaves :
    lgb = LGBMClassifier(n_estimators=400,num_leaves=leaf,n_jobs=-1,max_depth  = 30 , boost_from_average=False, device = 'gpu')
    # print("Depth ", dep , "Estimators :" , est,end="")
    print("leaf number : ", leaf , end=" ")
    modeling(lgb,X_train_over,X_test,y_train_over,y_test)

contest = ["precision", "recall", "f1"]

plt.figure()
plt.plot(leaves, pricision_list, 'o-', color="#ff7473", label= "precision")
plt.plot(leaves, recall_list, 'o-', color="#ffc952", label= "recall")
plt.plot(leaves, f1_score_list, 'o-', color="#47b8e0", label= "f1")
plt.xlabel('number of leaves')
plt.ylabel("Score")
# 그림에 선 표시
plt.grid(True)
# 범례 표시: best - 자동으로 최적의 위치에
plt.legend(loc="best")
plt.show()

# plt.figure()
# plt.plot(leaves, recall_list, 'o-', color="#ff9124", label= "recall")
# plt.xlabel('number of leaves')
# plt.ylabel("recall")
# # 그림에 선 표시
# plt.grid(True)
# # 범례 표시: best - 자동으로 최적의 위치에
# plt.legend(loc="best")
# plt.show()

# plt.figure()
# plt.plot(leaves, f1_score_list, 'o-', color="#ff9124", label= "f1")
# plt.xlabel('number of leaves')
# plt.ylabel("f1")
# # 그림에 선 표시
# plt.grid(True)
# # 범례 표시: best - 자동으로 최적의 위치에
# plt.legend(loc="best")
# plt.show()

