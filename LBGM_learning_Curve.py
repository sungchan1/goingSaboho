import pandas as pd 
from pandas import Series, DataFrame
from scipy.sparse.construct import random
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
import time


global bestRecall 
global bestF1  


#모델링
def modeling(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    metrics(y_test,pred)
    # print("1 " , sum(pred == 1))
    # print("0 " , sum(pred == 0))

#평가 지표
def metrics(y_test,pred):
    global bestRecall 
    global bestF1  
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    print('accuracy : ', accuracy , ' precision : ', precision , ' recall : ', recall , 'f1 score ',f1)
    if  recall > bestRecall :
        bestRecall = recall
        print("best Recall", bestRecall)
    if  f1 > bestF1 :
        bestF1 = f1
        print("best f1", bestF1)


def plot_learning_curve(estimator, x, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 20), s=None):
    if ylim is not None:
        plt.ylim(*ylim)
    # train_sizes : (392 + 392)의 80% 를 0.1, 0.325, 0.55, 0.775, 1의 비율로 학습시긴다.
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=s)
    train_scores = -train_scores
    test_scores = -test_scores
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


 
def ratio_learnin_Curve():
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(ratio)
    n = 1

    for r in ratio : 
        print("n", n)
        smote = SMOTE( sampling_strategy= r , random_state=0 )
        X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
    #   lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
        # lgb = LGBMClassifier(min_data_in_leaf = 320 ,n_estimators=400,num_leaves=20,n_jobs=-1,max_depth  = 30 , boost_from_average=False, device = 'gpu')

        lgb = LGBMClassifier( device = 'gpu', max_depth= 5, min_child_samples = 14, num_leaves=  15)
        modeling(lgb,X_train_over,X_test,y_train_over,y_test)
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"AUC" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='roc_auc')

        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"F1" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='f1')

        plt.subplot(2,5,n)
        plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"neg_mean_squared_error" , fontsize=14)
        plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='neg_mean_squared_error')
        
        n = n+1
    plt.show()

def learnin_Curve():
    smote = SMOTE( sampling_strategy= 0.1 , random_state=0 )
    X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
#   lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
    # lgb = LGBMClassifier(min_data_in_leaf = 320 ,n_estimators=400,num_leaves=20,n_jobs=-1,max_depth  = 30 , boost_from_average=False, device = 'gpu')

    lgb = LGBMClassifier( device = 'gpu', max_depth= 4, min_child_samples = 14, num_leaves=  14)
    modeling(lgb,X_train_over,X_test,y_train_over,y_test)
    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    # plt.figure(figsize=(10, 7))
    # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"AUC" , fontsize=14)
    # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='roc_auc')

    # plt.figure(figsize=(10, 7))
    # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"F1" , fontsize=14)
    # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='f1')

    plt.figure()
    plt.title("LGBM Learning Curve" , fontsize=14)
    plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='neg_mean_squared_error')
    plt.show()


def tuning_depth():

    depth = range(1, 8)

    
    smote = SMOTE( sampling_strategy= 0.1 , random_state=0 )
    X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
#   lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
    # lgb = LGBMClassifier(min_data_in_leaf = 320 ,n_estimators=400,num_leaves=20,n_jobs=-1,max_depth  = 30 , boost_from_average=False, device = 'gpu')
    for dep in depth :
        print("depth", dep)
        lgb = LGBMClassifier( device = 'gpu', max_depth= dep )
        modeling(lgb,X_train_over,X_test,y_train_over,y_test)
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"AUC" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='roc_auc')

        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"F1" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='f1')

        plt.subplot(3,5,int(dep))
        plt.title
        plt.title("LGBM - depth :" + str(dep) ,fontsize=14)
        plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='neg_mean_squared_error')

    plt.show()

def tuning_leaf():
    
    leaves = range(10, 30)
    smote = SMOTE( sampling_strategy= 0.1 , random_state=0 )
    X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
#   lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
    # lgb = LGBMClassifier(min_data_in_leaf = 320 ,n_estimators=400,num_leaves=20,n_jobs=-1,max_depth  = 30 , boost_from_average=False, device = 'gpu')
    for leaf in leaves :
        print("leaf", leaf)
        lgb = LGBMClassifier( device = 'gpu', max_depth= 4, min_child_samples = leaf )
        modeling(lgb,X_train_over,X_test,y_train_over,y_test)
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"AUC" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='roc_auc')

        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"F1" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='f1')

        plt.subplot(2,10,int(leaf) - 9)
        plt.title("LGBM - leaf :" + str(leaf) ,fontsize=14)
        plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='neg_mean_squared_error')

    plt.show()

def tuning_leavs():
    # 14
    leaves = range(2, 17)
    smote = SMOTE( sampling_strategy= 0.1 , random_state=0 )
    X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
#   lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)
    # lgb = LGBMClassifier(min_data_in_leaf = 320 ,n_estimators=400,num_leaves=20,n_jobs=-1,max_depth  = 30 , boost_from_average=False, device = 'gpu')
    for leaf in leaves :
        print("leaf", leaf)
        lgb = LGBMClassifier( device = 'gpu', max_depth= 4, min_child_samples = 14, num_leaves=  leaf)
        modeling(lgb,X_train_over,X_test,y_train_over,y_test)
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"AUC" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='roc_auc')

        # plt.figure(figsize=(10, 7))
        # plt.title("LGBM Learning Curve - Over Sampling ratio "+(str(r))+"F1" , fontsize=14)
        # plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='f1')

        plt.subplot(2,8,leaf-1)
        plt.title("LGBM - leaf :" + str(leaf))
        plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='neg_mean_squared_error')

    plt.show()
    

def greedy_search():
    depth = [3,4,5]
    childs = [13,14,15]
    leaves = [13,14,15]
    n = 1 
    f=1
    final_list = [1,4,10,11,24]

    for dep in depth :
        for child in childs :
            for leaf in leaves :
                print ("Count " ,n)
                if n in final_list :
                    smote = SMOTE( sampling_strategy= 0.1 , random_state=0 )
                    X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
                    lgb = LGBMClassifier( device = 'gpu', max_depth= dep, min_child_samples = child, num_leaves=  leaf)
                    modeling(lgb,X_train_over,X_test,y_train_over,y_test)
                    cv = ShuffleSplit(n_splits=5, test_size=0.2)
                    print("depth :", dep, " child : ", child , " leaf : ",leaf)
                    plt.subplot(1,5,f)
                    plt.title("LGBM Learning Curve "+str(n))
                    plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='neg_mean_squared_error')
                    f=f+1
                n = n+1

    plt.show()

def measure_mean():

    n = 1 
    f=1
    final_list = [1,4,10,11,24]
    random_list  = [0,10,20,30,40,50,60,70,80,90]

    recall_list = list()
    f1_list = list()
    precision_list =list()
    accuracy_list = list()
     
    time_list = list()
    
    for random in random_list :
        smote = SMOTE( sampling_strategy= 0.1 , random_state= random )
        X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
        lgb = LGBMClassifier( device = 'gpu', max_depth= 5, min_child_samples = 14, num_leaves=  15)
       

        

        lgb.fit(X_train_over,y_train_over)
        time_before = time.time()
        pred = lgb.predict(X_test)
        time_after = time.time()
        # metrics(y_test,pred)
        

        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred)
        recall = recall_score(y_test,pred)
        f1 = f1_score(y_test,pred)

        print('accuracy : ', accuracy , ' precision : ', precision , ' recall : ', recall , 'f1 score ',f1)


        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        time_list.append(time_after-time_before)


    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)
    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)

    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)

    time_list_mean = np.mean(time_list)
    time_list_std = np.std(time_list)



    print ("accuracy_mean : ",accuracy_mean,  " accuracy_std : ",accuracy_std,  "precision_mean: ",precision_mean, " precision_std : ",precision_std) 
    print ("recall_mean : " ,recall_mean, " recall_std : ",recall_std, " f1_mean : " ,f1_mean, " f1_std : ",f1_std) 
    print("time mean : ", time_list_mean, " time_std :", time_list_std)



    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    plt.figure()
    plt.title("LGBM Learning Curve "+str(n))
    plot_learning_curve(lgb, X_train_over, y_train_over, cv=cv, s='neg_mean_squared_error')
    

    plt.show()



df = pd.read_csv('creditcard.csv')
df = df.sample(frac=1)
# df.Class.value_counts(normalize=True).plot(kind='bar')
# print(df.Class.value_counts(normalize=True)*100)






X = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(df.describe())
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=10)
bestRecall =0
bestF1  =0
ratio_learnin_Curve()

