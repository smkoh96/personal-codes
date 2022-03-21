
# Sentimental Analysis 

This README file will explain how to deploy all the model artificals for my data science project 'Product reviews - Sentimental Analysis'. 
I will provide step by step instructions on how to reproduce the result on any dataset with reviews.





## Procedures on Sentimental Analysis

- Import the reviews as Pandas Dataframe/Series, make sure to combine review title and body as one.
- Preprocess the reviews using the function 'clean_text' shown below.
```python
import re
import string


def clean_text(text):
    ''' 
    1) Remove html tag
    
    2) Remove punctuation
    
    3) Remove Stopwords
    
    4) Lemmatize 
    
    5) Remove numbers
    
    6) Remove 1 lettered words
    '''
    
    #remove html tag
    text =BeautifulSoup(text,'html.parser').text
    
    text = text.lower()
    #remove punctuation
    
    text = re.sub('[!"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',' ',text)
  
    # remove double quotes
    text = re.sub(r'"', '', text)
    
    #remove stopwords
    sentence = nlp(text)
    sentence   =' '.join([notStopWords.text for notStopWords in sentence if not notStopWords.is_stop])
    
    doc = nlp(sentence)
    
    #remove lemmatize

    sentence =" ".join([token.lemma_ for token in doc])
    
    #remove numbers
    
    sentence = re.sub(r'[0-9]', r'', sentence)
    
    # reduce multiple spaces and newlines to only one
    
    #remove 1 lettered words
    sentence =  ' '.join([w for w in sentence.split() if len(w)>1])  

    return sentence
 ```
- Vectorize the clean text using TfidfVectorize, load the vectorizer from the pickle file ''.
    ```python
    # must use this library
    import joblib

    # Load the model from the file
    vectorizer = joblib.load('vectorizer_sentimental.pkl')
    X_train_tfidf = vectorizer.transform(list_of_clean_text)
    ```

- Predict Sentiment of each review using logreg. Positive if prediction is 1, else Negative.
    ```python
    # Load the model from the file
    logreg = joblib.load(logreg.pkl)
    prediction = logreg.predict(X_train_tfidf)
    ```




