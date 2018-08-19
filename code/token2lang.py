import pickle
from nltk.tokenize import TweetTokenizer
from googletrans import Translator

translator = Translator()
tweetTokenizer = TweetTokenizer()

(tw, l) = pickle.load(open('tokenizedTweets.p', 'rb'))
arr = []

for item in tw:
    for token in item:
        temp = translator.detect(token)
        try:
            lan = temp.lang
            if temp.lang != 'en':
                if temp.lang == 'hi':
                    op = translator.translate(token, src='hi', dest='en').text
                elif temp.lang == 'te':
                    op = translator.translate(token, src='te', dest='en').text
                else:
                    n = translator.translate(token, src='te', dest='en')
                    lan = translator.detect(n.text).lang

                    if lan == 'en':
                        op = n.text
                    else:
                        n = translator.translate(token, dest='en')
                        lan = translator.detect(n.text).lang

                        if lan == 'en':
                            op = n.text
                        else:
                            op = token
            else:
                op = token

            print(token, op, lan)

            lis = tweetTokenizer.tokenize(op)
            for temp in lis:
                arr.append(temp)

        except Exception as e:
            print(e)

f = open('output.p', 'wb')
pickle.dump(arr, f)