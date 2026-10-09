import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df=pd.read_csv("dataset/california_housing.csv"); X=df.drop(columns="MedHouseVal"); y=df["MedHouseVal"]
num=X.select_dtypes("number").columns.tolist(); cat=[c for c in X.columns if c not in num]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); results={}
for name,reg in {"Linear Regression":LinearRegression(),"Ridge":Ridge(alpha=1),"Lasso":Lasso(alpha=.01)}.items():
    pipe=Pipeline([("pre",pre),("model",reg)]); pipe.fit(tr,ytr); p=pipe.predict(te); results[name]=r2_score(yte,p)
    print(name,"MAE",mean_absolute_error(yte,p),"RMSE",mean_squared_error(yte,p)**.5,"R2",results[name])
plt.bar(results.keys(),results.values()); plt.ylabel("R²"); plt.xticks(rotation=15); plt.tight_layout(); plt.show()