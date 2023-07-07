import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

def Metrics_calc(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('F1score: ', f1score)
    print('Recall: ', recall)

def KNN_analysis(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    Metrics_calc(y_test, y_pred)

def SVM_analysis(X_train, X_test, y_train, y_test, k):
    clf = svm.SVC(kernel=k)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    Metrics_calc(y_test, y_pred)


path = 'Data/features_3_sec.csv'
df = pd.read_csv(path)

df = df.sample(frac=1).reset_index(drop=True)

    # Removendo todas as linhas duplicadas
df.drop_duplicates()
    # df.notnull().all() verifica se todos os valores de todas colunas não são nulos
#print(df.notnull().all())

#sns.heatmap(df.corr())
#plt.show()

X = df.drop(['label', 'filename', 'length'], axis=1)
y = df['label']

    #   Até k=40 os classificadores apresentam bons resultados
#X_new = SelectKBest(f_classif, k=30).fit_transform(X, y)
    #   Aplicar o PCA apresentou resultados piores do que simplesmente selecionar os k melhores
#pca = PCA(n_components=40, random_state=2020)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=237)


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('KNN: ')
KNN_analysis(X_train, X_test, y_train, y_test, 3)

print('SVM: ')
SVM_analysis(X_train, X_test, y_train, y_test, 'linear')