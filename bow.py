import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Read the data
data = pd.read_csv("test.csv")
data = data[['text', 'sentiment']]

# Tokenizing the text
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

# Vectorizing text
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=tokenizer.tokenize)
text_counts = vectorizer.fit_transform(data['text'])

# Spliting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(text_counts, data['sentiment'], test_size=0.25, random_state=5)

# Training the model
model = MultinomialNB()
model.fit(X_train, Y_train)

# Calculating the accuracy score of the model
predicted = model.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, Y_test)
print("Accuracy Score: ", accuracy_score)

# Reading the reviews data
reviews = pd.read_csv('user_review.csv')

# Transforming reviews text
review_counts = vectorizer.transform(reviews['review'])

# Predict sentiments of reviews
new_predicted_sentiment = model.predict(review_counts)
reviews['sentiment'] = new_predicted_sentiment

sentiment_distribution = reviews['sentiment'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#Output in csv
reviews.to_csv('analyzed_user_reviews.csv', index = False)

#print("Predicted Sentiment: ", new_predicted_sentiment)

