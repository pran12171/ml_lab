import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

df=pd.read_csv("dataset/titanic.csv"); features=["pclass","sex","age","sibsp","parch","fare","embarked"]; X=df[features]; y=df["survived"]
num=["pclass","age","sibsp","parch","fare"]; cat=["sex","embarked"]
pre=ColumnTransformer([("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])
clean=pd.DataFrame(pre.fit_transform(X),columns=pre.get_feature_names_out()); clean["survived"]=y.reset_index(drop=True)
clean.to_csv("titanic_cleaned.csv",index=False); print("Saved titanic_cleaned.csv"); print(clean.head())