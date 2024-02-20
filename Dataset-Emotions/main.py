import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('text.csv')
print(df.describe())

df['text']
