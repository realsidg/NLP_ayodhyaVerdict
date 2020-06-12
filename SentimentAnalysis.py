#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_pickle('corpus.pkl')
data


# In[2]:


from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)
data


# In[9]:


[i for i in data.index]


# In[11]:


# Let's plot the results
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]

for index, channel in enumerate(data.index):
    x = data.polarity.loc[channel]
    y = data.subjectivity.loc[channel]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, channel, fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


# In[12]:


import numpy as np
import math

def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list


# In[13]:


data


# In[14]:


list_pieces = []
for t in data.transcript:
    split = split_text(t)
    list_pieces.append(split)
    
list_pieces


# In[15]:


len(list_pieces)


# In[16]:


len(list_pieces[0])


# In[18]:


polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)
    
polarity_transcript


# In[22]:


plt.plot(polarity_transcript[0])
plt.title(data.index[0])
plt.show()


# In[29]:


# Show the plot for all comedians
plt.rcParams['figure.figsize'] = [16, 12]

for index, channel in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(channel)
    plt.ylim(ymin=-.2, ymax=.3)
    
plt.show()


# Unusual to see every channel has been pretty posiitive regarding the ayodhya verdict

# In[ ]:




