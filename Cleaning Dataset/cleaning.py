from nltk import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


token = TweetTokenizer()


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatize_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatize_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatize_sentence


# Data cleaning, getting rid of words not needed for analysis.
stop_words = stopwords.words('english')


def cleaned(token):
    if token == 'u':
        return 'you'
    if token == 'r':
        return 'are'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == 'secs':
        return 'seconds'
    if token == 'pls' or token == 'plz':
        return 'please'
    if token == '2morow':
        return 'tomorrow'
    if token == '2day':
        return 'today'
    if token == '4got' or token == '4gotten':
        return 'forget'
    if token == 'amp' or token == 'quot' or token == 'lt' or token == 'gt':
        return ''
    return token


# Noise removal from data, removing links, mentions and words with less than 3 length.
def remove_noise(tokens):
    cleaned_tokens = []
    for token, tag in pos_tag(tokens):
        # using non capturing groups ?:)// and eleminating the token if its a link.
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F]))+', '', token)
        token = re.sub('[^a-zA-Z]', ' ', token)
        # eliminating token if its a mention
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith("VB"):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        cleaned_token = cleaned(token.lower())
        # Eliminating if the length of the token is less than 3, if its a punctuation or if it is a stopword.
        if cleaned_token not in string.punctuation and len(cleaned_token) > 2 and cleaned_token not in stop_words:
            cleaned_tokens.append(cleaned_token)
    return cleaned_tokens

