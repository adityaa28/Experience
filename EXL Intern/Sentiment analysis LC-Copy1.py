#!/usr/bin/env python
# coding: utf-8

# In[34]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
data=pd.read_csv('Desktop/Data/EXL intern/data2.csv')
df=data.copy()


# # EDA

# In[35]:


print(df.shape)
#df.head()


# In[36]:


df['Content'] = df['Content'].astype(str).str.lower()
df.head(3)


# In[37]:


df.groupby('Source')['id'].count()


# In[38]:


df.groupby('Tags')['id'].count()


# In[39]:


df.isna().sum()


# In[40]:


df['Time2'] = pd.to_datetime(df['Time2'])
df


# # Data Processing

# ## NLP

# ### Cleaning and removing numerical characters

# In[41]:


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
df['Content'] = df['Content'].apply(lambda x: cleaning_numbers(x))
df['Content'].head()


# ### Spelling correction

# In[42]:


def correct_spelling(text):
    corrected_text = []
    word_list = text.split()
    for word in word_list:
        corrected_word = TextBlob(word)
        corrected_text.append(str(corrected_word.correct()))
    correct_text = ' '.join(corrected_text)
    return correct_text

df['Content'] = df['Content'].apply(lambda x: cleaning_numbers(x))
df['Content'].head()


# ### Tokenization

# In[43]:


from nltk.tokenize import RegexpTokenizer


# In[44]:


regexp = RegexpTokenizer('\w+')

df['1_token']=df['Content'].apply(regexp.tokenize)
df.head(3)


# ### Remove Stopwords

# In[45]:


import nltk
nltk.download('stopwords')


# In[46]:


from nltk.corpus import stopwords


# In[47]:


# Make a list of english stopwords
stopwords = nltk.corpus.stopwords.words("english")


# In[48]:


# Extend the list with your own custom stopwords
my_stopwords = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an', 'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do', 'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some',
                'such', 't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre", "youve", 'your', 'yours', 'yourself', 'yourselves', 'lending', 'club', 'lc', 'loan', 'also',
                'trustpilot', 'lendingclub', 'loans', 'money', 'would']
stopwords.extend(my_stopwords)


# In[49]:


df['1_token'] = df['1_token'].apply(lambda x: [item for item in x if item not in stopwords])


# In[50]:


df


# ### Tokenizing all unique words

# In[51]:


df['1_string'] = df['1_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
df


# In[52]:


all_words = ' '.join([word for word in df['1_string']])


# In[53]:


all_words


# In[54]:


#nltk.download('punkt')
tokenized_words = nltk.tokenize.word_tokenize(all_words)


# In[55]:


tokenized_words


# In[56]:


from nltk import FreqDist
fdist = FreqDist(tokenized_words)
fdist


# In[57]:


#nltk.download('wordnet')
#nltk.download('omw-1.4')


# In[58]:


df['1_string_fdist'] = df['1_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))
df


# ### Lemmatizing

# In[59]:


from nltk.stem import WordNetLemmatizer

wordnet_lem = WordNetLemmatizer()

df['1_string_lem'] = df['1_string_fdist'].apply(wordnet_lem.lemmatize)


# In[60]:


# check if the columns are equal
df['is_equal']= (df['1_string_fdist']==df['1_string_lem'])


# In[61]:


df


# In[62]:


df.is_equal.value_counts()


# In[63]:


conda install -c conda-forge wordcloud


# In[64]:


all_words_lem = ' '.join([word for word in df['1_string_lem']])


# ### Wordcloud function

# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def wordcloud(df):
    wordcloud = WordCloud(width=600, 
                         height=400, 
                         random_state=2, 
                         #max_font_size=100).generate(' '.join([word for word in df_process[df_process['sentiment']=='negative']['1_string_lem']]))
                         max_font_size=100).generate(" ".join(df))
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off');


# In[66]:


wordcloud(df['1_string_lem'])


# In[67]:


'''
from wordcloud import WordCloud
def wordcloud(df, feature):
    wordcloud = WordCloud(width = 800,
                         height = 600,
                         colormap = 'Set3',
                         margin = 0,
                         max_words = 200,
                         min_word_length = 4,
                         max_font_size = 130, min_font_size = 15,
                         background_color ='black').generate(" ".join(df[feature]))
    plt.figure(figsize = (20,15))
    plt.imshow(wordcloud)
    plt.axis('off')
'''


# In[68]:


#wordcloud(df,'1_string_lem')


# In[69]:


import numpy as np

x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wc = WordCloud(background_color="white", repeat=True, mask=mask)
wc.generate(all_words_lem)

plt.axis("off")
plt.imshow(wc, interpolation="bilinear");


# In[70]:


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

words = nltk.word_tokenize(all_words_lem)
fd = FreqDist(words)


# In[71]:


fd.most_common()


# In[ ]:





# In[72]:


fd.tabulate(10)


# In[73]:


# Obtain top 10 words
top_10 = fd.most_common(10)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))


# In[74]:


import seaborn as sns
sns.set_theme(style="ticks")
sns.barplot(y=fdist.index, x=fdist.values, color='blue');


# In[75]:


#pip install plotly


# In[76]:


import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()


# In[77]:


#nltk.download('vader_lexicon')


# In[78]:


from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


# In[79]:


df_process = df[(df['1_string_lem'].str.contains('process')) | (df['1_string_lem'].str.contains('application'))]
df_service= df[df['1_string_lem'].str.contains('service')]
df_rates= df[df['1_string_lem'].str.contains('rates') | df['1_string_lem'].str.contains('interest') | df['1_string_lem'].str.contains('rate') | df['1_string_lem'].str.contains('apr') ]
df_exp= df[(df['1_string_lem'].str.contains('experience')) | (df['1_string_lem'].str.contains('exp'))]


# In[80]:


df_process['1_string_lem']


# In[81]:


df['polarity'] = df['1_string_lem'].apply(lambda x: analyzer.polarity_scores(x))
df.tail(3)


# In[82]:


# Change data structure
df = pd.concat(
    [df.drop(['polarity'], axis=1), 
     df['polarity'].apply(pd.Series)], axis=1)
df.head()


# In[83]:


df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
df.head()


# In[84]:


df.loc[df['compound'].idxmax()].values


# In[85]:


df.loc[df['compound'].idxmin()].values


# In[86]:


# Number of tweets 
def Cplot(df):
    sns.countplot(y='sentiment', 
                 data=df, 
                 palette=['#b2d8d8',"#008080", '#db3d13']
                 );


# In[87]:


Cplot(df)


# In[88]:


print(Cplot(df[df['Source']=='TrustPilot']))


# In[89]:


print(Cplot(df[df['Source']=='LC']))


# In[90]:


# Lineplot function
def timeplot(df):
    g = sns.lineplot(x='Time2', y='compound', data=df)

    g.set(xticklabels=[]) 
    g.set(title='Sentiment of Tweets')
    g.set(xlabel="Time")
    g.set(ylabel="Sentiment")
    g.tick_params(bottom=False)

    g.axhline(0, ls='--', c = 'grey');


# In[91]:


#Boxplot function
def Bplot(df):
    sns.boxplot(y='compound', 
                x='sentiment',
                palette=['#b2d8d8',"#008080", '#db3d13'], 
                data=df);


# In[92]:


timeplot(df)


# In[93]:


print(timeplot(df[df['Source']=='TrustPilot']))


# In[94]:


print(timeplot(df[df['Source']=='LC']))


# In[95]:


#df = df.sort_values(by="Time2")
fig = px.line(df,x='Time2', y='compound')

# sort values
#fig.update_layout(barmode='line')

# show plot

fig.show()


# In[96]:


Bplot(df)


# In[97]:


print(Bplot(df[df['Source']=='TrustPilot']))


# In[98]:


print(Bplot(df[df['Source']=='LC']))


# In[99]:


nltk.download('state_union')


# ### 1. Application Process sentiment analysis

# In[100]:


df_process.reset_index(inplace = True)


# In[101]:


analyzer2 = SentimentIntensityAnalyzer()
df_process['polarity'] = df_process['1_string_lem'].apply(lambda x: analyzer2.polarity_scores(x))
# Change data structure
df_process = pd.concat(
    [df_process.drop(['polarity'], axis=1), 
     df_process['polarity'].apply(pd.Series)], axis=1)
df_process.head()


# In[102]:


df_process['sentiment'] = df_process['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(df_process.loc[df_process['compound'].idxmax()].values)
print(df_process.loc[df_process['compound'].idxmin()].values)
# Number of tweets 


# In[103]:


timeplot(df_process)


# In[104]:


print(timeplot(df_process[df_process['Source']=='TrustPilot']))


# In[105]:


print(timeplot(df_process[df_process['Source']=='LC']))


# In[106]:


df_process = df_process.sort_values(by="Time2")
fig = px.line(df_process,x='Time2', y='compound')

# sort values
#fig.update_layout(barmode='line')

# show plot

fig.show()


# In[107]:


Bplot(df_process)


# In[108]:


Bplot(df_process[df_process['Source']=='TrustPilot'])


# In[109]:


Bplot(df_process[df_process['Source']=='LC'])


# In[110]:


Cplot(df_process[df_process['Source']=='TrustPilot'])


# In[111]:


Cplot(df_process[df_process['Source']=='LC'])


# In[112]:


print(wordcloud(df_process[df_process['sentiment']=='positive']['1_string_lem']))
print(wordcloud(df_process[df_process['sentiment']=='negative']['1_string_lem']))


# ### 2. Service sentiment analysis

# In[113]:


df_service.reset_index(inplace = True)


# In[114]:


analyzer3 = SentimentIntensityAnalyzer()

df_service['polarity'] = df_service['1_string_lem'].apply(lambda x: analyzer3.polarity_scores(x))
# Change data structure
df_service = pd.concat(
    [df_service.drop(['polarity'], axis=1), 
     df_service['polarity'].apply(pd.Series)], axis=1)
df_service['sentiment'] = df_service['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(df_service.loc[df_service['compound'].idxmax()].values)
print(df_service.loc[df_service['compound'].idxmin()].values)
# Number of tweets 


# In[115]:


timeplot(df_service)


# In[116]:


print(timeplot(df_service[df_service['Source']=='TrustPilot']))


# In[117]:


print(timeplot(df_service[df_service['Source']=='LC']))


# In[118]:


df_service = df_service.sort_values(by="Time2")
fig = px.line(df_service,x='Time2', y='compound')

# sort values
#fig.update_layout(barmode='line')

# show plot

fig.show()


# In[119]:


Bplot(df_service)


# In[120]:


Bplot(df_service[df_service['Source']=='TrustPilot'])


# In[121]:


Bplot(df_service[df_service['Source']=='LC'])


# In[122]:


Cplot(df_service)


# In[123]:


Cplot(df_service[df_service['Source']=='TrustPilot'])


# In[124]:


Cplot(df_service[df_service['Source']=='LC'])


# In[125]:


print(wordcloud(df_service[df_service['sentiment']=='positive']['1_string_lem']))
print(wordcloud(df_service[df_service['sentiment']=='negative']['1_string_lem']))


# ### 3. Interest rates sentiment analysis

# In[126]:


df_rates.reset_index(inplace = True)


# In[127]:


analyzer4 = SentimentIntensityAnalyzer()

df_rates['polarity'] = df_rates['1_string_lem'].apply(lambda x: analyzer4.polarity_scores(x))
# Change data structure
df_rates = pd.concat(
    [df_rates.drop(['polarity'], axis=1), 
     df_rates['polarity'].apply(pd.Series)], axis=1)
df_rates['sentiment'] = df_rates['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(df_rates.loc[df_rates['compound'].idxmax()].values)
print(df_rates.loc[df_rates['compound'].idxmin()].values)
# Number of tweets 


# In[128]:


timeplot(df_rates)


# In[129]:


print(timeplot(df_rates[df_rates['Source']=='TrustPilot']))


# In[130]:


print(timeplot(df_rates[df_rates['Source']=='LC']))


# In[131]:


df_rates = df_rates.sort_values(by="Time2")
fig = px.line(df_rates,x='Time2', y='compound')

# sort values
#fig.update_layout(barmode='line')

# show plot

fig.show()


# In[132]:


Bplot(df_rates)


# In[133]:


Bplot(df_rates[df_rates['Source']=='TrustPilot'])


# In[134]:


Bplot(df_rates[df_rates['Source']=='LC'])


# In[135]:


Cplot(df_rates)


# In[136]:


Cplot(df_rates[df_rates['Source']=='TrustPilot'])


# In[137]:


Cplot(df_rates[df_rates['Source']=='LC'])


# In[138]:


print(wordcloud(df_rates[df_rates['sentiment']=='positive']['1_string_lem']))
print(wordcloud(df_rates[df_rates['sentiment']=='negative']['1_string_lem']))


# In[139]:


df_rates[df_rates['sentiment']=='positive'].to_csv('Downloads/rpos.csv')
df_rates[df_rates['sentiment']=='negative'].to_csv('Downloads/rneg.csv')


# ### 4. Customer experience sentiment analysis

# In[140]:


df_exp.reset_index(inplace = True)


# In[141]:


analyzer5 = SentimentIntensityAnalyzer()

df_exp['polarity'] = df_exp['1_string_lem'].apply(lambda x: analyzer5.polarity_scores(x))
# Change data structure
df_exp = pd.concat(
    [df_exp.drop(['polarity'], axis=1), 
     df_exp['polarity'].apply(pd.Series)], axis=1)
df_exp['sentiment'] = df_exp['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(df_exp.loc[df_exp['compound'].idxmax()].values)
print(df_exp.loc[df_exp['compound'].idxmin()].values)
# Number of tweets 


# In[142]:


timeplot(df_exp)


# In[143]:


print(timeplot(df_exp[df_exp['Source']=='TrustPilot']))


# In[144]:


print(timeplot(df_exp[df_exp['Source']=='LC']))


# In[145]:


Bplot(df_exp)


# In[146]:


Bplot(df_exp[df_exp['Source']=='TrustPilot'])


# In[147]:


Bplot(df_exp[df_exp['Source']=='LC'])


# In[148]:


Cplot(df_exp)


# In[149]:


Cplot(df_exp[df_exp['Source']=='TrustPilot'])


# In[150]:


Cplot(df_exp[df_exp['Source']=='LC'])


# In[151]:


print(wordcloud(df_exp[df_exp['sentiment']=='positive']['1_string_lem']))
print(wordcloud(df_exp[df_exp['sentiment']=='negative']['1_string_lem']))


# ### Collocations

# In[152]:


words1= nltk.word_tokenize(all_words_lem)


# In[153]:


# use to find bigrams, which are pairs of words
from nltk.metrics import BigramAssocMeasures
import nltk
from nltk.collocations import *
from nltk import bigrams
from nltk import trigrams


# ### Bigrams
# 

# In[154]:


#from nltk import quadgrams
#fgm = nltk.collocations.QuadgramAssocMeasures()
#list1=pd.Series(all_words_lem).str.split(expand=True)
filter1 = lambda *w: 'service' not in w
bgm    = nltk.collocations.BigramAssocMeasures()
finder1 = nltk.collocations.BigramCollocationFinder.from_words(words1)
finder1.apply_ngram_filter(filter1)
print (finder1.nbest(bgm.likelihood_ratio, 10))
#scored = finder.score_ngrams( bgm.likelihood_ratio  )


# ### Trigrams
# 

# In[155]:


filter2 = lambda *w: 'process' not in w
filter22= lambda *w: 'credit' in w
tgm = nltk.collocations.TrigramAssocMeasures()
finder2 = nltk.collocations.TrigramCollocationFinder.from_words(words1)
finder2.apply_ngram_filter(filter2)
finder2.apply_ngram_filter(filter22)
print (finder2.nbest(tgm.likelihood_ratio, 10))


# In[156]:


filter3 = lambda *w: 'rates' not in w
filter32= lambda *w: 'credit' in w
tgm = nltk.collocations.TrigramAssocMeasures()
finder3 = nltk.collocations.TrigramCollocationFinder.from_words(words1)
finder3.apply_ngram_filter(filter3)
finder3.apply_ngram_filter(filter32)
print (finder3.nbest(tgm.likelihood_ratio, 10))


# In[157]:


#print (finder2.nbest(bgm.likelihood_ratio, 10))


# In[158]:


#finder.nbest(bigram_measures.likelihood_ratio, 10)


# ###  Application Process

# In[159]:


wd=[]
for s in df_process['1_string_lem']:
    #s='The world is a small place, we should try to take care of this place.'
    m1 = re.search(r'((?:\w+\W+){,4})(process)\W+((?:\w+\W+){,4})', s)
    m2 = re.search(r'((?:\w+\W+){,4})(processing)\W+((?:\w+\W+){,4})', s)
    m3 = re.search(r'((?:\w+\W+){,4})(application)\W+((?:\w+\W+){,4})', s)
    #m3 = re.search(r'((?:\w+\W+){,4})(application)', s)
    #print(m3)
    if m1:
        l = [ x.strip().split() for x in m1.groups()]
    elif m2:
        l = [ x.strip().split() for x in m2.groups()]
    elif m3:
        l = [ x.strip().split() for x in m3.groups()]
    left, right = l[0], l[2]
    if(len(left)==0 & len(right)==0):
        words = s.split()
        if(words[-1]=='application' or words[-1]=='process' or words[-1]=='processing'):
            #print(words[-5:-1])
            left=words[-4:-1]
        else:
            #print(words[1:4])
            right=words[1:3]
    #else:
     #   print (left, right)
    wd.append(left+right)
    left.clear()
    right.clear()
    #print(left+right)
dfpro2=pd.DataFrame()
dfpro2['wd']=wd


# In[160]:


dfpro2['Source']=df_process['Source']
dfpro2['Time2']=df_process['Time2']
dfpro2['string'] = dfpro2['wd'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
dfpro2


# In[161]:


analyzer6 = SentimentIntensityAnalyzer()
dfpro2['polarity'] = dfpro2['string'].apply(lambda x: analyzer6.polarity_scores(x))
# Change data structure
dfpro2 = pd.concat(
    [dfpro2.drop(['polarity'], axis=1), 
     dfpro2['polarity'].apply(pd.Series)], axis=1)
dfpro2.head()


# In[162]:


dfpro2['sentiment'] = dfpro2['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(dfpro2.loc[dfpro2['compound'].idxmax()].values)
print(dfpro2.loc[dfpro2['compound'].idxmin()].values)


# In[163]:


print(timeplot(dfpro2))


# In[164]:



print(Bplot(dfpro2))


# In[165]:


print(timeplot(dfpro2[dfpro2['Source']=='TrustPilot']))


# In[166]:



print(Bplot(dfpro2[dfpro2['Source']=='TrustPilot']))


# In[167]:


print(timeplot(dfpro2[dfpro2['Source']=='LC']))


# In[168]:



print(Bplot(dfpro2[dfpro2['Source']=='LC']))


# In[169]:


Cplot(dfpro2)


# In[170]:


print(wordcloud(dfpro2[dfpro2['sentiment']=='positive']['string']))
print(wordcloud(dfpro2[dfpro2['sentiment']=='negative']['string']))


# ### Service

# In[171]:


wd.clear()
for s in df_service['1_string_lem']:
    #s='The world is a small place, we should try to take care of this place.'
    m1 = re.search(r'((?:\w+\W+){,4})(service)\W+((?:\w+\W+){,4})', s)
    if m1:
        l = [ x.strip().split() for x in m1.groups()]
    left, right = l[0], l[2]
    if(len(left)==0 & len(right)==0):
        words = s.split()
        if(words[-1]=='service'):
            #print(words[-5:-1])
            left=words[-4:-1]
        else:
            #print(words[1:4])
            right=words[1:3]
    #else:
     #   print (left, right)
    wd.append(left+right)
    left.clear()
    right.clear()
dfservice2=pd.DataFrame()
dfservice2['wd']=wd


# In[172]:


dfservice2['Source']=df_service['Source']
dfservice2['Time2']=df_service['Time2']
dfservice2['string'] = dfservice2['wd'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
analyzer7 = SentimentIntensityAnalyzer()
dfservice2['polarity'] = dfservice2['string'].apply(lambda x: analyzer7.polarity_scores(x))
# Change data structure
dfservice2 = pd.concat(
    [dfservice2.drop(['polarity'], axis=1), 
     dfservice2['polarity'].apply(pd.Series)], axis=1)
dfservice2.head()


# In[173]:


dfservice2['sentiment'] = dfservice2['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(dfservice2.loc[dfservice2['compound'].idxmax()].values)
print(dfservice2.loc[dfservice2['compound'].idxmin()].values)


# In[174]:


print(timeplot(dfservice2))


# In[175]:



print(Bplot(dfservice2))


# In[176]:


print(timeplot(dfservice2[dfservice2['Source']=='TrustPilot']))


# In[177]:



print(Bplot(dfservice2[dfservice2['Source']=='TrustPilot']))


# In[178]:


print(timeplot(dfservice2[dfservice2['Source']=='LC']))


# In[179]:



print(Bplot(dfservice2[dfservice2['Source']=='LC']))


# In[180]:


Cplot(dfservice2)


# In[181]:


print(wordcloud(dfservice2[dfservice2['sentiment']=='positive']['string']))
print(wordcloud(dfservice2[dfservice2['sentiment']=='negative']['string']))


# ### Interest rates

# In[182]:


wd.clear()
for s in df_rates['1_string_lem']:
    #s='The world is a small place, we should try to take care of this place.'
    m1 = re.search(r'((?:\w+\W+){,4})(interest)\W+((?:\w+\W+){,4})', s)
    m2 = re.search(r'((?:\w+\W+){,4})(rates)\W+((?:\w+\W+){,4})', s)
    m3 = re.search(r'((?:\w+\W+){,4})(apr)\W+((?:\w+\W+){,4})', s)
    #m3 = re.search(r'((?:\w+\W+){,4})(application)', s)
    #print(m3)
    if m1:
        l = [ x.strip().split() for x in m1.groups()]
    elif m2:
        l = [ x.strip().split() for x in m2.groups()]
    elif m3:
        l = [ x.strip().split() for x in m3.groups()]
    left, right = l[0], l[2]
    if(len(left)==0 & len(right)==0):
        words = s.split()
        if(words[-1]=='interest' or words[-1]=='rates' or words[-1]=='apr'):
            #print(words[-5:-1])
            left=words[-4:-1]
        else:
            #print(words[1:4])
            right=words[1:3]
    #else:
     #   print (left, right)
    wd.append(left+right)
    left.clear()
    right.clear()
    #print(left+right)
dfrates2=pd.DataFrame()
dfrates2['wd']=wd


# In[183]:


dfrates2['Source']=df_rates['Source']
dfrates2['Time2']=df_rates['Time2']
dfrates2['string'] = dfrates2['wd'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
analyzer8 = SentimentIntensityAnalyzer()
dfrates2['polarity'] = dfrates2['string'].apply(lambda x: analyzer8.polarity_scores(x))
# Change data structure
dfrates2 = pd.concat(
    [dfrates2.drop(['polarity'], axis=1), 
     dfrates2['polarity'].apply(pd.Series)], axis=1)
dfrates2.head()


# In[184]:


dfrates2['sentiment'] = dfrates2['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(dfrates2.loc[dfrates2['compound'].idxmax()].values)
print(dfrates2.loc[dfrates2['compound'].idxmin()].values)


# In[185]:


print(timeplot(dfrates2))


# In[186]:



print(Bplot(dfrates2))


# In[187]:


print(timeplot(dfrates2[dfrates2['Source']=='TrustPilot']))


# In[188]:



print(Bplot(dfrates2[dfrates2['Source']=='TrustPilot']))


# In[189]:


print(timeplot(dfrates2[dfrates2['Source']=='LC']))


# In[190]:



print(Bplot(dfrates2[dfrates2['Source']=='LC']))


# In[191]:


Cplot(dfrates2)


# In[192]:


print(wordcloud(dfrates2[dfrates2['sentiment']=='positive']['string']))
print(wordcloud(dfrates2[dfrates2['sentiment']=='negative']['string']))


# ### Experience

# In[193]:


wd.clear()
for s in df_exp['1_string_lem']:
    #s='The world is a small place, we should try to take care of this place.'
    m1 = re.search(r'((?:\w+\W+){,4})(experience)\W+((?:\w+\W+){,4})', s)
    m2 = re.search(r'((?:\w+\W+){,4})(exp)\W+((?:\w+\W+){,4})', s)
    #m3 = re.search(r'((?:\w+\W+){,4})(application)', s)
    #print(m3)
    if m1:
        l = [ x.strip().split() for x in m1.groups()]
    elif m2:
        l = [ x.strip().split() for x in m2.groups()]
    left, right = l[0], l[2]
    if(len(left)==0 & len(right)==0):
        words = s.split()
        if(words[-1]=='experience' or words[-1]=='exp'):
            #print(words[-5:-1])
            left=words[-4:-1]
        else:
            #print(words[1:4])
            right=words[1:3]
    #else:
     #   print (left, right)
    wd.append(left+right)
    left.clear()
    right.clear()
    #print(left+right)
dfexp2=pd.DataFrame()
dfexp2['wd']=wd


# In[194]:


dfexp2['Source']=df_exp['Source']
dfexp2['Time2']=df_exp['Time2']
dfexp2['string'] = dfexp2['wd'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
analyzer9 = SentimentIntensityAnalyzer()
dfexp2['polarity'] = dfexp2['string'].apply(lambda x: analyzer9.polarity_scores(x))
# Change data structure
dfexp2 = pd.concat(
    [dfexp2.drop(['polarity'], axis=1), 
     dfexp2['polarity'].apply(pd.Series)], axis=1)
dfexp2.head()


# In[195]:


dfexp2['sentiment'] = dfexp2['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
print(dfexp2.loc[dfexp2['compound'].idxmax()].values)
print(dfexp2.loc[dfexp2['compound'].idxmin()].values)


# In[196]:


print(timeplot(dfexp2))


# In[197]:



print(Bplot(dfexp2))


# In[198]:


print(timeplot(dfexp2[dfexp2['Source']=='TrustPilot']))


# In[199]:



print(Bplot(dfexp2[dfexp2['Source']=='TrustPilot']))


# In[200]:


print(timeplot(dfexp2[dfexp2['Source']=='LC']))


# In[201]:



print(Bplot(dfexp2[dfexp2['Source']=='LC']))


# In[202]:


Cplot(dfexp2)


# In[203]:


print(wordcloud(dfexp2[dfexp2['sentiment']=='positive']['string']))
print(wordcloud(dfexp2[dfexp2['sentiment']=='negative']['string']))


# In[ ]:




