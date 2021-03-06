import sys
import pickle
from googletrans import Translator
from gensim.models import Word2Vec
import numpy as np
import re
import json
import splitter
from nltk.tokenize import TweetTokenizer
from nltk.stem import LancasterStemmer


def create_data():
    tt = TweetTokenizer()
    ls = LancasterStemmer()
    tr = Translator()

    f1 = open('../combined_useful.json')
    f2 = open('../combined_useless.json')

    parsed_json = []
    for line in f1:
        parsed_json.append(json.loads(line))

    for line in f2:
        parsed_json.append(json.loads(line))

    print len(parsed_json[0]['data'])
    print len(parsed_json[1]['data'])

    stop_words_file = open('twitterStopWords.txt')
    punctuation_file = open('punctuations.txt')
    stop_words = []
    punctuations = []

    for word in stop_words_file:
        stop_words.append(word.rstrip('\n'))

    for punc in punctuation_file:
        punctuations.append(punc.rstrip('\n'))

    abbrs = {
        'u': 'you',
        'n': 'and',
        'l8': 'late',
        'ur': 'your',
        'k': 'ok',
        'wer': 'where',
        'wen': 'when',
        'b': 'be',
        'y': 'why',
        'dis': 'this',
        'v': 'we',
        'plz': 'please',
        'pls': 'please',
        'thr': 'there',
        'shd': 'should',
        'iam': 'i am',
        'masikitos': 'mosquito',
        'sec': 'security',
        'reqd': 'required',
        'mgmt': 'management',
        'hyd': 'hyderabad',
        'brb': 'be right back',
        'fr': 'for',
        'prob': 'problem',
        'don': 'do not',
        'pic': 'picture',
        'ny': 'any',
        'nyc': 'nice',
        'nic': 'nice',
        'tmrw': 'tomorrow'
    }

    len1 = len(parsed_json[0]['data'])
    len2 = len1

    data = []
    for d in parsed_json[0]['data'][:400]:
        data.append(d)
    for d in parsed_json[1]['data'][:400]:
        data.append(d)

    tweets = []
    names = []
    handles = []
    hashtags = []

    labels = []
    for i, d in enumerate(data):
        temp = ""
        cnt1 = 0
        try:
            x = re.sub(r"http\S+", "", d['text'])
            x = re.sub(r"www\S+", "", x)
        except Exception:
            continue

        for ch in x:
            if ord(ch) > 127:
                cnt1 += 1
                continue
            if str(ch.encode('ascii', 'ignore')) in punctuations:
                temp += ' '
            else:
                temp += ch.lower()

        x = [k for k in tt.tokenize(temp)]
        tweets.append(x)

    temp_tweets = []
    for t in tweets:
        temp_t = []
        temp_hash = []
        temp_handles = []

        for w in t:
            if any(ch.isdigit() for ch in w):
                continue
            if w.startswith('@'):
                temp_handles.append(w.split('@')[1])
            elif w.startswith('#'):
                temp_hash.append(w.split('#')[1])
                for sw in splitter.split(w.split('#')[1]):
                    temp_t.append(sw)
            elif w in abbrs:
                for ab in abbrs[w].split(' '):
                    temp_t.append(ab)
            else:
                temp_t.append(w)
        temp_tweets.append(temp_t)
        hashtags.append(temp_hash)
        handles.append(temp_handles)

    tweets = temp_tweets

    new_tweets = []

    for tw in tweets:
        new_tw = []
        for word in tw:
            if word in stop_words:
                continue
            else:
                new_tw.append(word.encode('ascii', 'ignore'))
        new_tweets.append(new_tw)

    max_len = 0
    for tw in new_tweets:
        if len(tw) > max_len:
            max_len = len(tw)

    print "Training "
    w2v = Word2Vec(new_tweets, size=128, min_count=2, iter=50, negative=20, window=10)
    w2v.save('relevantNetW2V_5')
    # w2v = Word2Vec.load('relevantNetW2V')

    vocab = w2v.wv.vocab

    embedded_data = []
    for i, tw in enumerate(new_tweets):
        # print labels[i]
        # print tw
        d = []
        for w in tw:
            try:
                d.append(w2v[w])
            except Exception:
                d.append(np.zeros(128))
        embedded_data.append(d)
    pickle.dump([(new_tweets, embedded_data, labels), handles, hashtags, vocab, max_len], open('dataset.p', 'wb'))
    return (new_tweets, embedded_data, labels), handles, hashtags, vocab, max_len


(new_tweets, embedded_tweets, labels), handles, hashtags, vocab, max_len = create_data()

for i, e in enumerate(embedded_tweets):
    print(labels[i])
    print(new_tweets[i])

#
# print max_len
# print len(vocab)

# for i, et in enumerate(embedded_tweets):
#     print labels[i]
#     print et
