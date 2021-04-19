#IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

#Change working dir 
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\Natural Language Processing')

#get data
yelp = pd.read_csv('C:\\Users\\User\\Documents\\yelp.csv')

#EDA
print('\n',yelp.head(),'\n')

print('\n',yelp.info(),'\n')

print('\n',yelp.describe(),'\n')

yelp['text length'] = yelp['text'].apply(len)

print('\n',yelp.head(),'\n')

f = sns.FacetGrid(data=yelp,col='stars')
f.map(plt.hist,'text length',bins=40)
f.savefig('Text Length Facet Grid.jpg')
plt.show()

#BOXPLOT
sns.boxplot(data=yelp,x='stars',y='text length',palette='rainbow')
plt.savefig('Rating Boxplot.jpg')
plt.show()

#COUNTPLOT
sns.countplot(data=yelp,x='stars',palette='rainbow')
plt.savefig('Rating Count.jpg')
plt.show()

#4 and 5 star rating are the most common

print('\n',yelp.groupby('stars').mean(),'\n')

print('\n',yelp.groupby('stars').mean().corr(),'\n')

#HEATMAP
sns.heatmap(data=yelp.groupby('stars').mean().corr(),cmap='coolwarm',annot=True)
plt.savefig('heatmap.jpg')
plt.show()

X = yelp['text']
y= yelp['stars']

#Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

preds = nb.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print('Confusion Matrix','\n')
print('\n',confusion_matrix(y_test,preds),'\n')
print('Classification Report','\n')
print('\n',classification_report(y_test,preds),'\n')

# NOT good performance

#LETS try Tfidf Transformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('cv',CountVectorizer()),('tfidf',TfidfTransformer()),('nb',MultinomialNB())])

X = yelp['text']
y= yelp['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)

print('Confusion Matrix:','\n')
print(confusion_matrix(y_test,predictions),'\n')
print('Classification Report:','\n')
print(classification_report(y_test,predictions),'\n')

#To simplify model, lets only include rating with 1 or 5 stars
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

X = yelp_class['text']
y= yelp_class['stars']

cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

preds = nb.predict(X_test)

print('Confusion Matrix','\n')
print('\n',confusion_matrix(y_test,preds),'\n')
print('Classification Report','\n')
print('\n',classification_report(y_test,preds),'\n')

from sklearn.metrics import plot_confusion_matrix

sns.set_style('white')
plot_confusion_matrix(nb,X_test,y_test)
plt.savefig('1 or 5 Confusion Matrix.jpg')
plt.show()

#DECENT PERFORMANCE

#LETS TRY Tfidf
pipeline = Pipeline([('cv',CountVectorizer()),('tfidf',TfidfTransformer()),('nb',MultinomialNB())])

X = yelp_class['text']
y= yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)

print('Confusion Matrix:','\n')
print(confusion_matrix(y_test,predictions),'\n')
print('Classification Report:','\n')
print(classification_report(y_test,predictions),'\n')

sns.set_style('white')
plot_confusion_matrix(pipeline,X_test,y_test)
plt.savefig('Tfidf Confusion Matrix.jpg')
plt.show()

#WE get better performance by not using Tfidf transformer

#END







