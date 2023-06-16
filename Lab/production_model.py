import pandas as pd
from sklearn.metrics import classification_report
import joblib

#loading sample
test_sample = pd.read_csv("Data/test_sample.csv")

#Dividing  up x and values
X,y = test_sample.drop('cardio', axis  = 1), test_sample['cardio']
#the model loaded
model = joblib.load("Lab/my_model.pkl")

#Using our model to make predictions
predict = model.predict(X)

#not working as i have used 'hard' voting in my classifier, something I had to do as 'soft' voting did not work with my
#linearSVC model. I will however write the rest of the code I would have used in order to complete the last steps
#in the lab
probabilities = model.predict_proba(X)

probability_1 = probabilities.loc[:,1]
probability_0 = probabilities.loc[:,0]

#dict with prediction, and probabilities
prob = {"Predictions": predict, "Probability class 1": probability_1, "Probability class 0": probability_0}

predictions = pd.DataFrame(prob)

predictions.to_csv("Data/predictions.csv")