#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle 

channels = ['ndtv', 'indiatoday', 'republic']
data = {}
for c in (channels):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c] = [file.read().decode("utf-8") ]


# In[2]:


data.keys()


# In[3]:


import pandas as pd
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df


# In[4]:


data_df.transcript.loc['ndtv']


# In[5]:


import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)


# In[6]:


data_clean = pd.DataFrame(data_df.transcript.apply(round1))


# In[7]:


data_clean


# In[8]:


def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)


# In[9]:


data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
data_clean


# In[10]:


data_df
data_df.to_pickle("corpus.pkl")


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm


# In[12]:


data_dtm.to_pickle("dtm.pkl")


# In[13]:


data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))


# In[ ]:




