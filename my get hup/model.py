import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
Cancer1= pd.read_csv('cancer patient data sets.csv')
Cancer2 = Cancer1.drop(['index'],axis=1)
Cancer3 = Cancer2.drop(['Patient Id'],axis=1)
df= Cancer3.drop(['Passive Smoker'],axis=1)
df
# Select independent and dependent variable
X=df.drop(['Level'],axis=1)
y=df['Level']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = LogisticRegression()

# Fit the model
classifier.fit(X_train, y_train)

# Save the model and the scaler to pickle files
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))
