#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:11:19 2019

@author: sanjanamendu
"""

from tqdm import tqdm
import pandas as pd
import string
import sklearn
import sklearn.decomposition
import contractions
import itertools
import nltk
import re
import os

tqdm.pandas()
home = os.path.expanduser("~")
mallet_path = home+"/Downloads/Mallet/"

# =============================================================================
#                                  TF-IDF
# =============================================================================

fb_agg = pd.read_csv(mallet_path+"fbmsg_agg.csv").drop('Unnamed: 0',1)
fb_agg.Clean_Content = fb_agg.Clean_Content.fillna('')
fb_agg.Clean_Content = fb_agg.groupby('PID').Clean_Content.transform(lambda x: ' '.join(x))
fb_agg = fb_agg.groupby('PID').first().reset_index()
fb_agg.Clean_Content = fb_agg.Clean_Content.str.replace('\s\s+','',regex=True)



word_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,3), analyzer='word')
X = word_vectorizer.fit_transform(fb_agg.Clean_Content)
tfidf = pd.DataFrame(X.toarray(), columns=word_vectorizer.get_feature_names(), index=fb_agg.PID)

word_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,3), analyzer='word')
X = word_vectorizer.fit_transform(fb_agg.Clean_Content)
frequencies = pd.DataFrame(X.toarray(), columns=word_vectorizer.get_feature_names(), index=fb_agg.PID)

binary = (frequencies > 0).astype(int)

min_participants = 10
vocab = [x for x in binary.columns.values[binary.sum(axis=0) > min_participants] \
                            if not x.replace(' ','').isdigit()]

# Pointwise Mutual Information
text =  " ".join(fb_agg.Clean_Content)

trigram_measures = nltk.collocations.TrigramAssocMeasures()
trigram_coloc = nltk.collocations.TrigramCollocationFinder.from_words(nltk.tokenize.word_tokenize(text))
trigram_pmi = trigram_coloc.score_ngrams(trigram_measures.pmi)

trigram_keep = [' '.join(term) for term, pmi in trigram_pmi if pmi > 9]

bigram_measures = nltk.collocations.BigramAssocMeasures()
bigram_coloc = nltk.collocations.BigramCollocationFinder.from_words(nltk.tokenize.word_tokenize(text))
bigram_pmi = bigram_coloc.score_ngrams(bigram_measures.pmi)

bigram_keep = [' '.join(term) for term, pmi in bigram_pmi if pmi > 6]

final_vocab = [v for v in tqdm(vocab) if len(v.split()) == 1 or v in bigram_keep or v in trigram_keep]

tfidf[final_vocab].to_csv("fbmsg_term_tfidf.csv")

#binary[final_vocab].to_csv("fbmsg_term_binary.csv")
#
#final_freq = frequencies[final_vocab]
#final_freq.to_csv("fbmsg_term_freq.csv")
#
#final_relativfreq = final_freq.div(final_freq.sum(axis=1), axis=0) # normalize
#final_relativfreq.to_csv("fbmsg_term_relativfreq.csv")
#
#final_anscombe = final_freq.apply(lambda x: 2 * np.sqrt(x + 0.375)) # apply Anscombe transformation
#final_anscombe.to_csv("fbmsg_term_anscombe.csv")

# =============================================================================
#                              TOPIC MODELING
# =============================================================================

fb_agg = pd.read_csv("fbmsg_agg (participant).csv")
fb_agg.Content = fb_agg.Content.fillna('')

# Lists of Words to Omit
stop = nltk.corpus.stopwords.words('english')

extra_stop = ['get','got','tho','one',"i'll","i'm","that's","can't",'yall','hey',"i've",'also',\
              'dont','thats','yes','ani','tah','thats','yea','dont','hey','also',"i'll",'that!',\
              'ur',"It's","i've","i'm","'cause","they'll","i'll","that's","how'd",\
              "who's","let's",'ill','want','also','might',"he's",'cant',"y'all",'hey','much',\
              'good','also','doesnt','didnt','wont',"we're",'whats',"there's","what's","ain't",\
              'imma','kekek','idek','maybe','itll','isnt','that ','hows','would',"there's",\
              'havent','zach','kkkk',"what's","we'll",'isnt','whats','maybe','u','im']

friends_names = ['katie','erik','chris','johnny','alan','brandon','andrew','daniel','tony',\
                 'richie','adrien','john','andrew','stephen','crystal','david','juhan','erik',\
                 'edmund','claire','sadia','jenny','chelsea','tony','kirtana','kadariya','bishal',\
                 'tony','nihar','adrien','tina','erica','pooja','hajur','steph','rohan','kuo','rashid']

names = [n.lower() for n in nltk.corpus.names.words()]

punctuation = list(string.punctuation)

# Concatenate all lists of words to omit (except punctuation)
no_punc = stop + extra_stop + friends_names + names

# Regex + Non-Regex List of Words to Omit
rm_words = no_punc + punctuation
rm_regex = no_punc + [re.escape(x) for x in punctuation]

# Vocabulary
tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

pat = r'\b(?:{})\b'.format('|'.join(rm_regex))

tokenized = fb_agg.Content.progress_apply(lambda x: ' '.join([t for t in tknzr.tokenize(x) if t not in rm_words]))
fb_agg['Tokenized_Content'] = tokenized

def clean_text(x):
    x = ' '.join([t for t in tknzr.tokenize(x)])
    x = contractions.fix(x) # Expand Contractions
    x = ''.join(''.join(s)[:2] for _, s in itertools.groupby(x)) # Standardize
    x = re.sub('http\S+','',x) # Remove URLs
    x = re.sub('www\S+','',x) # Remove URLs
    x = re.compile(pat, re.I).sub("",x) # Remove Stopwords
    x = re.sub(r'\b\w{1,2}\b','',x) # Remove words with < 2 letters
    x = re.sub('\s\s+',' ',x) # Remove Extra Spaces
    return x

fb_agg['Clean_Content'] = fb_agg.Tokenized_Content.progress_apply(lambda x: clean_text(x))

fb_agg = fb_agg.replace(r'\\n',' ', regex=True) 

fb_agg[["PID","Clean_Content"]].to_csv("fbmsg_agg.csv")

# --- Mallet Commands
os.system(mallet_path + "bin/mallet import-file " + \
          "--input "+ mallet_path +"fbmsg_agg.csv --output fbmsg_agg.mallet " + \
          "--keep-sequence")

os.system(mallet_path+"bin/mallet train-topics --config "+mallet_path+"config.txt")

doc_topic = pd.read_csv("fbmsg_agg_doc-topics.csv", header=None)
doc_topic.columns = ['Document'] + ['Topic '+str(i) for i in range(100)]
doc_topic["PID"] = doc_topic.pop('Document').str.split(',',expand=True).iloc[:,1]
doc_topic = doc_topic[doc_topic.PID.str.len() == 4]
avg_doc_topic = doc_topic.groupby('PID').mean().reset_index()
avg_doc_topic.to_csv("fbmsg_doc_topic.csv",index=False)

topic_usage = pd.read_csv("fbmsg_doc_topic.csv")
tfidf = pd.read_csv("fbmsg_term_tfidf.csv")


semantic_feat = pd.merge(tfidf, avg_doc_topic, 'left', on='PID')