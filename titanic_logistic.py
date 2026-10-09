import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix

df=pd.read_csv("dataset/titanic.csv"); f=["pclass","sex","age","sibsp","parch","fare","embarked"]; X,y=df[f],df["survived"]
num=["pclass","age","sibsp","parch","fare"]; cat=["sex","embarked"]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
m=Pipeline([("pre",pre),("model",LogisticRegression(max_iter=1000))]); tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
m.fit(tr,ytr); p=m.predict(te); pr=m.predict_proba(te)[:,1]
print("Accuracy",accuracy_score(yte,p)); print("Precision",precision_score(yte,p)); print("Recall",recall_score(yte,p)); print("F1",f1_score(yte,p)); print("ROC-AUC",roc_auc_score(yte,pr)); print(confusion_matrix(yte,p))