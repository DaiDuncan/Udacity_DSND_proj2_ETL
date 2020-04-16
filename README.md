# Disaster Response Pipeline Project

### Dataset:
- There are a lot of messages as the response to disasters.
- These messages can be divided into three different genres(**'direct', 'news', 'social'**)
- Each message was described to 36 different classes.
- Class with value '1': represent that the message is classified in this class.

### Data Wrangling:
##### 1. `message`
- **Tokenize** the text of the message into words
- Use `nltk` to **normalize, lemmatize** the words, and also **delete the stopwords**

##### 2. `categories`
- Extract the categories information from the Dataset(DataFrame)
- Find two special categories
    - `related`: some values are '2', change them to '1'
    - `child_alone`: there is just one unique value, useless to classfy the message, delete this categorie


### Building Model
##### Feature1: `message` (INPUT)
Raw Data: already tokenized words of text
- Use `CountVectorizer()` with tokenizer in Data Wrangling to make the words to vectors
- Use `TfidfTransformer()` to normalize the value of the vectors

##### Feature2: `genre` (INPUT)
*!!!GUESS: maybe genre is related to the classification.*
- Use `OneHotEncoder()`, change 'genre' objects to numbers  

##### Labels: `categories` (OUTPUT)

##### Model
- Use `ColumnTransformer()`, transform the two features from different sources in one transformer called `'preprocessor'`
- Use `Pipeline()`, build a model with transformer `'preprocessor'` and classifier `RandomForestClassifier()`
    - **Notice**: since there are many classes, use `MultiOutputClassifier()`


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl df_overview.csv df_accuracy.csv`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001

### Result:
1. Model:
![predict accuracy](./images/classification_accuracy.png)
2. Web APP:
![Overview](./images/Overview.png)
![Response:query](./images/response1.png)
![response_classification](./images/response_classification.png)


# Warning:
1. Develop enev  
The basic frame use old python modules, e.g. `joblib`  
update:
```py
import joblib #instead: from sklearn.externals import joblib
```
