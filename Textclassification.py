import pandas as pd
import numpy as np
import time
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# read the files form the folder
main_folder = "C:/Users/visha/Desktop/bbc/"
file_data = pd.DataFrame(columns=['Category', 'File_Name', 'Data'])

category_list = []
files_list = []
data_list = []

for category in os.listdir(main_folder):
    print("\n", category)
    subfolder_path = os.path.join(main_folder, category)
    for files in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, files)

        # print(file_path)
        category_list.append(category)
        files_list.append(files)
        file_ptr = open(file_path)
        data = file_ptr.read().split('\n')
        data = list(filter(None, data))
        # data = data.split(' ')
        data_list.append(data)

file_data['Category'] = category_list
file_data['File_Name'] = files_list
file_data['Data'] = data_list

file_data.head()

file_data.describe()

file_data['Category'].value_counts()

file_data.Data[0][:1000]

# 0 - business, 1 -entertainment, 2 - politics, 3 - sport, 4 - tech
label_encode = LabelEncoder()
file_data['Label'] = label_encode.fit_transform(file_data['Category'])
file_data.sample(5)

#convert data into an array
data_array = np.array(file_data['Data'])

stop_words = stopwords.words('english')
ps = PorterStemmer()

# convert into tokens, remove stop words and stem the tokens
tokenizer = RegexpTokenizer('[A-Za-z]\w+')
for idx in range(len(data_array)):
    data_array[idx] = tokenizer.tokenize(str(data_array[idx]))

data_array = [[ps.stem(token) for token in doc if token not in stop_words] for doc in data_array]

file_data['Token_Data']=data_array
file_data.sample(10)

#replace , with space in token list
file_data['Token_Data2'] = [ ' '.join(map(str,tok)) for tok in file_data['Token_Data']]

file_data.head(2)

x_train,x_test,y_train,y_test = train_test_split(file_data['Token_Data2'].values,file_data['Label'].values)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

x_train[0]

x_test.shape

# Form tf-idf vector
vectorizer = TfidfVectorizer()

test_input = ["This is sports column"]
test_input = np.array(test_input)
x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector =vectorizer.transform(x_test)
test_vector = vectorizer.transform(test_input)
pickle.dump(x_train, open("Vishanthini_data.npy", 'wb'))

x_train_vector.shape, x_test_vector.shape,test_vector.shape

print(list(label_encode.classes_))


# Use multiple classifiers and grid search for prediction
def ML_modeling(models, params, X_train, X_test, y_train, y_test):
    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError('Some estimators are missing parameters')

    for key in models.keys():
        model = models[key]
        param = params[key]
        gs = GridSearchCV(model, param, cv=10, error_score=0, refit=True)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)

        # Print scores for the classifier
        print(key, ':', gs.best_params_)
        print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (
        accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='macro'),
        recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')))

    return gs

# reference : https://www.kaggle.com/rockystats/bbc-text-classification-word2vec-vs-tf-idf
# Preparing to make a pipeline
models = {
    'Naive Bayes': MultinomialNB(),
}


#params
params = {
    'Naive Bayes': { 'alpha': [0.5, 1], 'fit_prior': [True, False] },
}

trained_model_NB = ML_modeling(models, params, x_train_vector, x_test_vector, y_train, y_test)
## ML_modeling method also prints performance scores for each classifier

trained_model_NB.get_params


# Test the subject classification model:
def vectorize(test_input):
    # stem and  stop words
    global vectorizer
    stop_words = stopwords.words('english')
    ps = PorterStemmer()

    # convert into tokens, remove stop words and stem the tokens
    tokenizer = RegexpTokenizer('[A-Za-z]\w+')
    test_input = tokenizer.tokenize(str(test_input))

    test_input = [ps.stem(token) for token in test_input if token not in stop_words]
    #     print(test_input)

    test_input = [' '.join(map(str, test_input))]
    print(test_input)
    test_input = np.array(test_input)
    #     print(test_input)
    test_vector = vectorizer.transform(test_input)
    #     print(test_vector)

    return test_vector

test_input = "hello World sensex"
# test_input= test_input.replace('"', ' ')
print(type(test_input))
print(test_input)

# test_input = np.array(test_input)
# print(test_input)
# test_vector = vectorizer.transform(test_input)
# print(test_vector)

#predict the subject classifiction
test_vector= vectorize(test_input)
y_pred = trained_model_NB.predict(test_vector)
y_prob = trained_model_NB.predict_proba(test_vector)
y_pred,y_prob,y_prob[0][np.argmax(y_prob)]

# save the model to disk
filename = 'subject_Classification_NB.sav'
pickle.dump(trained_model_NB, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(test_vector)
print(label_encode.inverse_transform(result))

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(24,20))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()