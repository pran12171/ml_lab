import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv("dataset/titanic.csv"); print("Shape:",df.shape); print(df.head())
print("\nMissing values:\n",df.isnull().sum())
sns.countplot(data=df,x="survived"); plt.title("Titanic Survival Distribution"); plt.tight_layout(); plt.show()
features=["pclass","sex","age","sibsp","parch","fare","embarked"]; X,y=df[features],df["survived"]
num=["pclass","age","sibsp","parch","fare"]; cat=["sex","embarked"]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("preprocessor",pre),("classifier",LogisticRegression(max_iter=1000))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); model.fit(Xtr,ytr); p=model.predict(Xte)
print("Accuracy:",accuracy_score(yte,p)); print(classification_report(yte,p)); print(confusion_matrix(yte,p))