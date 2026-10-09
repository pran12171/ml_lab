import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
df=pd.read_csv("dataset/winequality-red.csv")
df["quality_class"]=pd.cut(df["quality"],bins=[-1,4,6,10],labels=["Low","Medium","High"])
X=df.drop(columns=["quality","quality_class"]); y=df["quality_class"]; tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
for name,m in {"Decision Tree":DecisionTreeClassifier(max_depth=6,random_state=42),"Random Forest":RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42,n_jobs=-1)}.items():
 m.fit(tr,ytr); p=m.predict(te); print("\n",name,"Accuracy:",accuracy_score(yte,p)); print(classification_report(yte,p))