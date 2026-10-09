import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
df=pd.read_csv("dataset/heart.csv"); X=df.drop(columns="target"); y=df["target"]; tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
m=DecisionTreeClassifier(max_depth=5,random_state=42); m.fit(tr,ytr); p=m.predict(te); print("Accuracy:",accuracy_score(yte,p)); print(classification_report(yte,p)); print(confusion_matrix(yte,p))