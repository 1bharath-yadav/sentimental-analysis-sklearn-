import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

app = Flask(__name__)

# Load the training data
data = pd.read_excel("train.xlsx")

# Preprocess the data and extract features
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['News Headline'])
y = data['Sentiment']

# Define the model
model = SVC()

# Define the hyperparameters grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Get the best model and hyperparameters
best_model = grid_search.best_estimator_


from sklearn.model_selection import train_test_split

# Load the training data
data = pd.read_excel("train.xlsx")

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the test data to a new file
test_data.to_excel('test.xlsx', index=False)


# Load the testing data
test_data = pd.read_excel("test.xlsx")

# Preprocess the testing data
X_test = tfidf_vectorizer.transform(test_data['News Headline'])
y_test = test_data['Sentiment']

# Make predictions on the testing data
y_pred = best_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save the best model and vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the form
    text = request.form['text']

    # Preprocess the input text and transform it using the TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])

    # Make predictions using the trained model
    prediction = best_model.predict(text_vectorized)

    # Return the predicted sentiment
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
