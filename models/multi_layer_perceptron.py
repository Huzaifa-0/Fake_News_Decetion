
import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""> ## Import and examine data"""

real_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
real_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([real_df, fake_df], axis=0, ignore_index=True)
print(real_df.shape)
print(fake_df.shape)

df.info()

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df["label"])
plt.show()

from wordcloud import WordCloud, STOPWORDS
plt.figure()
wc= WordCloud(max_words=100, stopwords= STOPWORDS).generate(" ".join(df[df['label']==1]['text']))
plt.title('Real News Title WordCloud')
plt.imshow(wc, interpolation='bilinear')

plt.figure()
wc= WordCloud(max_words=100, stopwords= STOPWORDS).generate(" ".join(df[df['label']==0]['text']))
plt.title('Fake News Title WordCloud')
plt.imshow(wc, interpolation='bilinear')

"""## Preprocess data"""

import warnings   
warnings.filterwarnings(action = 'ignore') 

import re
import pickle
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data = df
data.drop(['title'], inplace=True, axis=1)
data.drop(['subject'], inplace=True, axis=1)
data.drop(['date'], inplace=True, axis=1)
data.head()

def preprocess(df):
    lemmatizer = WordNetLemmatizer()

    text_processed = []
    for text in df.text:
        # remove punctuation and lowercase
        text = re.sub(r'[^a-zA-Z]', ' ', text) 
        text = text.lower()
        
        # tokenize and lemmatize tokens
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(x) for x in tokens]
        text_processed.append(' '.join(tokens))
        
    # token count vectorization
    text_vectorizer = CountVectorizer(stop_words='english', max_features=4000)
    text_matrix = text_vectorizer.fit_transform(text_processed).toarray()
    
    # save vectorizers
    pickle.dump(text_vectorizer, open('text_vectorizer.pkl','wb'))
    
    # store label then drop old text columns and label
    y = np.array(df.label)
    df.drop(['text','label'], inplace=True, axis=1)
    
    # return X, y as np matrices
    return text_matrix, y

X, y = preprocess(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""## Feed-forward Multi-Layer Perceptron"""

import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # input layer
        self.l1 = nn.Linear(4000, 2000) 
        self.relu1 = nn.ReLU()
        
        # hidden layer 1
        self.l2 = nn.Linear(2000, 500)  
        self.relu2 = nn.ReLU()
        
        # hidden layer 2
        self.l3 = nn.Linear(500, 100)    
        self.relu3 = nn.ReLU()
        
        # hidden layer 3
        self.l4 = nn.Linear(100, 20)   
        self.relu4 = nn.ReLU()
        
        # output layer
        self.l5 = nn.Linear(20, 2)    
        
    def forward(self, X):
        out = self.l1(X)
        out = self.relu1(out)
        
        out = self.l2(out)
        out = self.relu2(out)
        
        out = self.l3(out)
        out = self.relu3(out)
        
        out = self.l4(out)
        out = self.relu4(out)
        
        out = self.l5(out)
        return out

"""## Optimizer and loss function"""

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
error = nn.CrossEntropyLoss()
print(model)

"""## Training"""

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train).type(torch.LongTensor)

X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test).type(torch.LongTensor)

epochs = 20

for epoch in range(epochs):
    
    # clear gradients
    optimizer.zero_grad()
    
    # forward pass
    out = model(X_train)
    
    # compute loss
    loss = error(out, y_train)
    
    # backprop
    loss.backward()
    
    # update parameters
    optimizer.step()
    
    # print train loss
    print(f'Epoch {epoch} Loss: {loss}')

"""## Evaluate model"""

from sklearn.metrics import accuracy_score
y_pred = model(X_test)
y_pred_max = torch.max(y_pred,1)[1]
test_accuracy = accuracy_score(y_pred_max, y_test)
print(f'Test accuracy: {test_accuracy}')

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_pred=y_pred_max, y_true=y_test)

fig,ax = plt.subplots(figsize=(6,6))
sns.heatmap(confusion_matrix,annot=True,fmt="0.1f",linewidths=1.5)
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
f1 = f1_score(y_true=y_test, y_pred=y_pred_max)
precision = precision_score(y_true=y_test, y_pred=y_pred_max)
recall = recall_score(y_true=y_test, y_pred=y_pred_max)
print(f'f1_score: {f1}')
print(f'precision: {precision}')
print(f'recall: {recall}')

import pickle
pickle.dump(model, open('multi-layer-perceptron-parameters.pkl', 'wb'))