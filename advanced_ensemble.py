import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("dataset/heart.csv"); X=df.drop(columns="target"); y=df["target"]; tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
try:
 from xgboost import XGBClassifier
 m=XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=42); m.fit(tr,ytr); p=m.predict(te); print("XGBoost Accuracy:",accuracy_score(yte,p))
 try:
  import shap
  shap.summary_plot(shap.TreeExplainer(m).shap_values(te),te,show=False); plt.tight_layout(); plt.show()
 except ImportError: print("Install SHAP: pip install shap")
except ImportError: print("Install XGBoost: pip install xgboost")
try:
 from lightgbm import LGBMClassifier
 m2=LGBMClassifier(n_estimators=200,max_depth=5,learning_rate=.05,random_state=42,verbosity=-1); m2.fit(tr,ytr); print("LightGBM Accuracy:",accuracy_score(yte,m2.predict(te)))
except ImportError: print("Install LightGBM: pip install lightgbm")