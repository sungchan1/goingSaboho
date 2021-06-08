import pandas as pd 
import matplotlib.pyplot as plt
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
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


k_list = range(1,101)
accuracies_before_OverSampling = []
accuracies_after_OverSampling = []


df = pd.read_csv('creditcard.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
smote = SMOTE(random_state=0)


# devide train, test -> over Sampling
print("------------- 오버 샘플링 나중 ----------------")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=10)
X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
for k in k_list:
    print(k)
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train_over, y_train_over)
    pred = classifier.predict(X_test)
    precision = precision_score(y_test,pred)
    accuracies_before_OverSampling.append(precision)
    # print(classifier.score(X_test, y_test))
    # accuracies_before_OverSampling.append(classifier.score(X_test, y_test))

plt.plot(k_list, accuracies_before_OverSampling)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()


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


# # over Sampling  -> devide train, test
# print("------------- 오버 샘플링 먼저 ----------------")
# smote2 = SMOTE(random_state=0)
# X_over,y_over = smote.fit_sample(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X_over,y_over,test_size=0.25,random_state=10)
# for k in k_list:
#     classifier = KNeighborsClassifier(n_neighbors = k)
#     classifier.fit(X_train, y_train)
#     # print(classifier.score(X_test, y_test))
#     accuracies_after_OverSampling.append(classifier.score(X_test, y_test))
# plt.plot(k_list, accuracies_after_OverSampling)
# plt.xlabel("k")
# plt.ylabel("Validation Accuracy")
# plt.title("Breast Cancer Classifier Accuracy")
# plt.show()

