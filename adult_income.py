import pandas as pd, matplotlib.pyplot as plt
df=pd.read_csv("dataset/adult.csv"); print(df.head()); print("Shape:",df.shape)
df["capital_net"]=df["capitalgain"]-df["capitalloss"]; df["hours_per_age"]=df["hoursperweek"]/(df["age"]+1); df["education_num_age"]=df["education-num"]*df["age"]
df["class"].value_counts().plot(kind="bar"); plt.title("Adult Income Class Distribution"); plt.tight_layout(); plt.show()
print(df.describe(include="all"))