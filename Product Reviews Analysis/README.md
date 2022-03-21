
# Product Segmentation Demo

This README file will explain how to deploy all the model artificals for my data science project 'Product Segmentation'. 
I will provide step by step instructions on how to reproduce the result on any dataset with reviews.





## Procedures on Product Segmentation

- Import the reviews as Pandas Dataframe/Series, make sure to combine review title and body as one.
- Preprocess the reviews using the function 'clean_text' shown below.
```python
def clean_text(text):
    ''' 
    1) Remove html tag
    
    2) Remove punctuation
    
    3) Remove Stopwords
    
    4) Lemmatize 
    
    5) Remove numbers
    
    6) Remove 1-3 lettered words
    
    7) Only keep Noun and Adjective
    '''
    import re  

    #remove html tag
    BeautifulSoup(text,'html.parser').text
    
    text = text.lower()
    #remove punctuation
    
    text = re.sub('[!"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',' ',text)
    
    #remove stopwords
    sentence = nlp(text)
    sentence   =' '.join([notStopWords.text for notStopWords in sentence if not notStopWords.is_stop])
    
    doc = nlp(sentence)
    
    #lemmatize

    sentence =" ".join([token.lemma_ for token in doc])
      
    #remove numbers
    
    sentence = re.sub(r'[0-9]', r'', sentence)
    
    # reduce multiple spaces and newlines to only one
    
    
    #remove 1-3 words
    sentence =  ' '.join([w for w in sentence.split() if len(w)>3])
    
    doc = nlp(sentence)
    
    #adjective and nouns only
    sentence = ' '.join([w.text for w in doc if (w.tag_ =='NN' or w.tag_ == 'NNS' or w.tag_ =='JJ')])
    
    return sentence
 ```

- Vectorize the clean text using count_vector, load the count_vector from the pickle file ''.
    ```python
    # must use this library
    import joblib

    # Load the model from the file
    count_vect = joblib.load('count_vector.pkl')
    X_count = count_vect.transform(list_of_clean_text)
    ```

- Create topic probability distribution for each review
    ```python
    # Load the model from the file
    lda_model = joblib.load('lda_model_topics.pkl')
    X_topics = lda_model.transform(X_count)
    ```
- Scale X_topic
    ```python
    # Load the model from the file
    scaler = joblib.load(scaling_topic_matrix.pkl)
    X_scaled = lda_model.transform(X_topics)
    ```
- Assign Clusters for each review
    ```python
    # Load the model from the file
    kmeans = joblib.load(kmean_model.pkl)
    clusters = kmeans.predict(X_scaled)
    ```




