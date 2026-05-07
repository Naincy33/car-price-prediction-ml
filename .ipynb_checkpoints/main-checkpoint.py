import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("ford.csv")
df.head()
df.info()
df.describe()
df.shape
df.isnull().sum()
#sns
sns.histplot(df["price"],bins=50, kde=True)
plt.show()
df.corr(numeric_only=True)

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

sns.boxplot(data = df, x = 'year', y = 'price')
plt.xticks(rotation = 90)
plt.show()

sns.scatterplot(data = df, x = 'mileage',y = 'price')
plt.show()

sns.boxplot(data = df, x = 'engineSize', y = 'price')
plt.show()

df.columns
sns.boxplot(data = df, x = 'transmission',y = 'price')
plt.show()
sns.boxplot(data = df,x = 'fuelType', y = 'price' )
plt.show()
sns.boxplot(x = df['model'],y = df['price'])
plt.xticks(rotation = 90)
plt.show()

X = df.drop(columns = ['price'],axis = 1)
y = df['price']
X.head()
df.columns

X_one_encode = pd.get_dummies(X,columns = ['model','transmission','fuelType'],drop_first = True)
X_one_encode = X_one_encode.astype(int)
X_one_encode


from sklearn.preprocessing import LabelEncoder

columns = ['model', 'transmission', 'fuelType']

Xlable = X.copy()  # make a safe copy
label_encoders = {}

for col in columns:
    le = LabelEncoder()
    Xlable[col] = le.fit_transform(Xlable[col].astype(str))  # Convert to string in case of nulls
    label_encoders[col] = le
Xlable
