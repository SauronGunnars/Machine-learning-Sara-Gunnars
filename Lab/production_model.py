import pandas as pd
from sklearn.metrics import classification_report
import joblib

#loading sample
test_sample = pd.read_csv("Data/test_sample.csv")

#Dividing  up x and values
X,y = test_sample.drop('cardio', axis  = 1), test_sample['cardio']
