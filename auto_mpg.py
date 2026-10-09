import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("dataset/auto_mpg.csv"); X=df.drop(columns=["mpg","car_name"]); y=df["mpg"]
num=["cylinders","displacement","horsepower","weight","acceleration","model_year"]; cat=["origin"]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
m=Pipeline([("pre",pre),("model",LinearRegression())]); tr,te,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); m.fit(tr,ytr); p=m.predict(te)
print("RMSE:",mean_squared_error(yte,p)**.5,"R2:",r2_score(yte,p)); plt.scatter(yte,p); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Auto MPG"); plt.tight_layout(); plt.show()