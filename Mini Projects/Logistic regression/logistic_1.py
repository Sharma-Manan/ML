# -*- coding: utf-8 -*-
"""LR_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ygu4rJD0xQgEezHr0LPh-DoDGAwYm2YA
"""

import pandas as pd
import numpy as np

df = pd.read_csv('p1.csv')
df.head()

df.info()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Gender'] = LE.fit_transform(df['Gender'])

X = df[['Gender','Age','EstimatedSalary']]
Y = df['Purchased']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)

y_hat = lr.predict(X_test)

from sklearn.metrics import accuracy_score as AS

accuracy = AS(Y_test, y_hat)
accuracy*100

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_hat)
print("Confusion Matrix:\n", cm)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix as heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()