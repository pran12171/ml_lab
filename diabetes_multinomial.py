import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

df=pd.read_csv("dataset/diabetes.csv"); df["severity"]=pd.cut(df["Glucose"],bins=[-1,99,125,float("inf")],labels=["Low","Moderate","High"])
X=df.drop(columns=["Outcome","severity"]); y=df["severity"]
m=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(multi_class="multinomial",max_iter=2000))])
tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); m.fit(tr,ytr); p=m.predict(te)
print("Accuracy:",accuracy_score(yte,p)); print(classification_report(yte,p))