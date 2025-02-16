import pandas as pd

messages = pd.read_csv('EmailCollection', sep='\t' , names=['LABEL', 'MESSAGES'])#Seperating the each by the tab space

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='LABEL', data=messages)
plt.show()

import nltk
import re #Regular expression is used 
nltk.download('punkt') #Punkt function is used to tokenize the words by the space
nltk.download('stopwords') #Stopwords are the words that are not required for the analysis("is ,was ,it etc")
nltk.download('wordnet') #Wordnet is a dictionary of words

from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['MESSAGES'][i])#Removing the special characters(Non Alphabets)
    review = review.lower()#Converting to lowercase
    review = review.split()#Splitting the words by the space
    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]# Removing the stopwords
    review = ' '.join(review)#Joining the words by the space
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer is used to count the words
cv = CountVectorizer(max_features=3500) #Maximum features are 3500
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(messages['LABEL'])
print('xYxYxY',X)
y=y.iloc[:,1].values #1st column is the spam column The main use of iloc is select the specific
print("XXXXX",X)
print("YYYY",y)
import pickle
pickle.dump(cv,open('cv-transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
mnb = MultinomialNB(alpha=0.8)
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100,"%")

pickle.dump(mnb,open('mnb-model.pkl','wb'))
message = "Hello, how are you doing today?"
data = [message]
vect = cv.transform(data).toarray()
my_prediction = mnb.predict(vect)
if my_prediction[0] == 1:
    print("Spam")
else:
    print("Not Spam")
