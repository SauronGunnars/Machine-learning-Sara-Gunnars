import pandas as pd
from sklearn.metrics import classification_report
import joblib

test_sample = pd.read_csv("Data/test_sample.csv")

print(test_sample.head())
