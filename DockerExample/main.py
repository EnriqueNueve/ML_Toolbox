import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import pandas as pd 
from sklearn.linear_model import LinearRegression

TestData = sys.argv[1]
# Read the data that will serve for training and test data
X_full = pd.read_csv("train_data.csv")
# Read the data that will serve for testing
X_test = pd.read_csv(TestData)

# Seperate target from predictors in both training and test data
y = X_full.y
y_test = X_test.y 

X_full.drop(['y'],axis=1,inplace=True)
X_test.drop(['y'],axis=1,inplace=True)

# Apply model 
model = LinearRegression()
model.fit(X_full,y)
preds = model.predict(X_test)

# Save output 
output = pd.DataFrame({'Id':X_test.index,'Y Original': y_test, 'Y predicted':preds})
output.to_csv('/data/outputTest.txt',index=False)

