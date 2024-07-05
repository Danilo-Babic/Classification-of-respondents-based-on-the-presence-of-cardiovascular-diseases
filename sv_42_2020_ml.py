import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import datasets
import seaborn as sns
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# Ucitavanje dataset-a
df = pd.read_csv('cardio_train.csv',delimiter=';')
print(df.shape)
df.head()

print("Info o obelezjima")
print(df.describe())
#df.info()


# mozda da izbacis kao outlier skroz
negativne_vrednosti_sistolni = df.loc[df['ap_hi']<0,'ap_hi'].unique()
for i in negativne_vrednosti_sistolni:
  df['ap_hi'].replace(negativne_vrednosti_sistolni,abs(negativne_vrednosti_sistolni), inplace=True)

# izbaci??
negativne_vrednosti_dijastolni = df.loc[df['ap_lo']<0,'ap_lo'].unique()
for i in negativne_vrednosti_dijastolni:
  df['ap_lo'].replace(negativne_vrednosti_dijastolni,abs(negativne_vrednosti_dijastolni), inplace=True)
  
  
inx = df.loc[df['height']<120].index
df.drop(inx, inplace= True, axis = 0)

inx1 = df.loc[df['ap_hi']>370].index
df.drop(inx1, inplace= True, axis = 0)

inx2 = df.loc[df['ap_lo']>360].index
df.drop(inx2, inplace= True, axis = 0)

inx3 = df.loc[df['ap_hi']<50].index
df.drop(inx3, inplace= True, axis = 0)

inx4 = df.loc[df['ap_lo']<20].index
df.drop(inx4, inplace= True, axis = 0)

inx5 = df.loc[df['weight']<50].index
df.drop(inx5, inplace= True, axis = 0)

y = df['cardio']
df.drop(['id', 'cardio'], inplace= True, axis = 1)

df['weight'] = df['weight'].astype(int)
numerical_feats = df.dtypes[df.dtypes == "int64"].index

corr = df[numerical_feats].corr()
f = plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df,y,test_size=0.1,random_state=10,stratify=y)

s = StandardScaler()
s.fit(x_train)
x_train_std = s.transform(x_train)
x_test_std = s.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)

x_train_std.columns = list(df.columns)
x_test_std.columns = list(df.columns)
x_train_std.head()

# PCA
pca = PCA(n_components=None) 
pca.fit(x_train_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of PCA components')
plt.ylabel('Variance ratio')
plt.grid(True)

#PCA FINALNO?
pca_finalno = PCA(n_components=7)
pca_finalno.fit(x_train_std)
x_train_pca = pca_finalno.transform(x_train_std)
x_test_pca = pca_finalno.transform(x_test_std)

# Logisticka regresija PCA
#parameters_LR_PCA = {'solver':('lbfgs', 'sag', 'saga')}
#classifier_LR_PCA = LogisticRegression(fit_intercept = True)
#clf_LR_PCA = GridSearchCV(classifier_LR_PCA, parameters_LR_PCA, scoring='recall', cv=10, verbose=3)
#clf_LR_PCA.fit(x_train_pca, y_train)

##print("LOG REG PCA:")
#print("best score: ", clf_LR_PCA.best_score_)
#print("best hyperparameters: ", clf_LR_PCA.best_params_)
#print("*****************************")

# # Random Forrest Tree PCA
# parameters_RFT_PCA = {'n_estimators': [100, 325, 550, 775, 1000],'max_depth': [10, 50, 100]}
# classifier_RFT_PCA = RandomForestClassifier()
# clf_RFT_PCA = GridSearchCV(classifier_RFT_PCA, parameters_RFT_PCA, scoring='recall', cv=10, verbose=3)
# clf_RFT_PCA.fit(x_train_pca, y_train)

# print("RANDOM FORREST TREE REZ:")
# print("najbolji skor: ", clf_RFT_PCA.best_score_)
# print("najbolji hiperparametri: ", clf_RFT_PCA.best_params_)
# print("*****************************")


parameters_SVC_pca = {'C':[0.1, 1, 10], 'gamma':[0.1, 1, 10]}
classifier_SVC_pca = SVC(kernel = 'rbf')
clf_SVC_pca = GridSearchCV(classifier_SVC_pca, parameters_SVC_pca, scoring='recall', cv=10, verbose=3,n_jobs=-1)
clf_SVC_pca.fit(x_train_pca,y_train)

print("best score: ", clf_SVC_pca.best_score_)
print("best hyperparameters: ", clf_SVC_pca.best_params_)