import os
import spacy
import pickle
import swifter
import gc
import argparse

import numpy as np
import pandas as pd

import multiprocessing as mp

from multiprocessing import Pool
from datetime import datetime
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc, Span, Token

from pywsd.lesk import simple_lesk

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# only worked inside the virtual environment, not in container
# def load_data(file_name):
#     """"Loads data from the json files in pandas dataframe """
#
#     max_records = 1e5
#
#     reader = pd.read_json(file_name, lines=True, chunksize=max_records)
#
#     df = pd.DataFrame()  # Initialize the dataframe
#
#     try:
#         for df_chunk in reader:
#             df = pd.concat([df, df_chunk])
#     except ValueError:
#         print('\nSome messages in the file cannot be parsed')
#
#     return df


# simplification: classify if review is useful or not, u
# useful if certain percentage (threshold) of reactions is useful

def generate_labels(review_data, threshold): # 0.60
    """Label review as useful (1) if useful compliments account for the majority (according to threshold) of compliments"""

    review_data['useful_perc'] = review_data['useful'] / (review_data['cool'] + review_data['funny'] + review_data['useful'])
    review_data['useful_perc'] = review_data['useful_perc'].replace(np.nan, 0)

    review_data['useful_01'] = review_data['useful_perc']
    review_data.loc[review_data['useful_01'] >= threshold, 'useful_01'] = 1
    review_data.loc[review_data['useful_01'] < threshold, 'useful_01'] = 0

    print('0: ', len(review_data['useful_01'][review_data['useful_01'] == 0]) / len(review_data), ' %')  # 0.7172087527483211
    print('1: ', len(review_data['useful_01'][review_data['useful_01'] == 1]) / len(review_data), ' %')  # 0.2827912472516789

    return review_data

def merge_review_and_user_data(review_data, user_data):
    """merges review and user data"""

    return review_data.merge(user_data, how='left', on='user_id', suffixes=('_review', '_user'))

def downsample_data(merged_data, percentage_to_keep):
    """ Downsample the data to speed up computation time"""

    return merged_data.sample(frac = percentage_to_keep, random_state = 101).reset_index()


# feature generation

# features
# 1) Review features: word count, perc unique words without stop words, sentiment score (Senti Wordnet) (?), avg of word vecs, no of adjectives, stars
# 2) User features: no of good writer compliments, no of write more compliments, no of thank you compliments, no hot stuff compliments
# no great pic compliments, days yelping, no of reviews, no of fans


def generate_features(df):
    """Generates all the above mentioned features"""

    feature_data = df[['useful_01', 'text', 'stars', 'average_stars', 'compliment_more', 'compliment_hot', 'compliment_photos',
                        'compliment_writer', 'compliment_plain',
                        'fans', 'review_count', 'yelping_since', 'user_id', 'useful_review']]

    feature_data = feature_data.merge(pd.DataFrame(feature_data.groupby(feature_data['user_id']).useful_review.sum()), \
                                      how='left', on='user_id',
                                      suffixes=('', '_compliments_received_for_other_reviews'))

    feature_data['useful_review_compliments_received_for_other_reviews'] = \
        feature_data['useful_review_compliments_received_for_other_reviews'] - feature_data['useful_review']



    nlp = spacy.load('en_core_web_lg', disable=['ner', "parser"])

    # define for each token the attribute _.ignore:
    ignore_getter = lambda token: (token.is_stop or  # remove stop words
                                   token.lower_ in STOP_WORDS or  # ignore stop words independent of their case
                                   # token.like_num or # ignore stuff that looks like a number
                                   # token.is_digit or #ignore tokens consisting of digits
                                   token.is_punct or  # ignore punctuation tokens
                                   token.is_left_punct or  # ignore left punctuation
                                   token.is_right_punct or  # ignore right punctuation
                                   token.is_space or  # ignore tokens consisting of spaces
                                   token.is_bracket or  # ignore brackets
                                   token.is_quote or  # ignore quotation marks
                                   not token.is_alpha)  # ignore everything that is not only letters
    # (this might be too strict, but without it special characters
    # like +$% etc will stay). With it, however, most of the previous
    # stuff is not needed..
    Token.set_extension('ignore', getter=ignore_getter, force=True)  # add the _.ignore attribute

    def get_days_yelping(input_timestamp):
        days_yelp = datetime.now().date() - datetime.strptime(input_timestamp.split()[0], '%Y-%m-%d').date()
        return days_yelp.days

    def get_tokens(nlp_text):
        return [token for token in nlp_text if not (token._.ignore or not (len(token) > 1))]

    def get_unique__perc(token_list):
        if not len(token_list) == 0:
            return len(set([str(tok) for tok in token_list])) / len(token_list)
        else:
            return 0

    def get_word_vectors(token_list):
        if not len(token_list) == 0:
            word_vectors = [token.vector for token in token_list]
            return np.linalg.norm(sum(word_vectors) / len(token_list))
        else:
            return 0

    def get_sentiment(token_list, nlp_review):
        """Get sentiment using SentiWordNet"""

        # WordNet (and thus SentiWordNet) use different POS tags than spacy output
        # here we transform from spacy to WordNet
        wordNet_pos_dict = {"NOUN": "n",
                            "VERB": "v",
                            "ADJ": "a",
                            "ADV": "r"}

        def posTag_to_wordNetTag(tag):
            if tag in wordNet_pos_dict:
                return wordNet_pos_dict[tag]
            else:
                return None

        # to get the semantic score of a word from SentiWordNet we first need to find this word in WordNet
        # This process is called Word Sense Disambiguation and we use the simple lesk algorithm for that
        def get_semantic_score_with_context(token, nlp_review):

            word = token.lower_  # get lowercased token text
            position = token.idx  # get position of word in document
            pos = posTag_to_wordNetTag(token.pos_)  # get POS of token, for better word sense disambiguation

            # define how many tokens around the token of interest we look at
            num_surrounding_words = 10
            # careful if there are less then num_surrounding_words before our token or after our token
            leftmost_word_idx = max(0, position - num_surrounding_words)
            rightmostword_idx = min(len(nlp_review), position + num_surrounding_words)
            surrounding_text = nlp_review[leftmost_word_idx: rightmostword_idx].text

            # determine word with the closest sense in WordNet
            #     print(word,"....",surrounding_text,pos)
            try:
                word_with_closest_sense = simple_lesk(surrounding_text, word, pos=pos)
            except:
                word_with_closest_sense = simple_lesk(surrounding_text, word)
            #     print(word,pos,word_with_closest_sense)
            # find the sentiment score to the word we found in wordnet
            if word_with_closest_sense:
                sentiword = swn.senti_synset(word_with_closest_sense.name())

                sent_scores = {"objective": sentiword.obj_score(), "positive": sentiword.pos_score(),
                               "negative": sentiword.neg_score()}

                sentiment = max(sent_scores, key=sent_scores.get)

                return sentiment
            else:
                return 'no_sentiment_assigned'

        if not len(token_list) == 0:
            sentiments = []
            for token in token_list:
                sentiments.append(get_semantic_score_with_context(token, nlp_review))

            counts = Counter(sentiments)

            return max(counts, key=counts.get)
        else:
            return 'no_sentiment_assigned'


    feature_data['days_yelping'] = feature_data.yelping_since.swifter.apply(get_days_yelping)
    del feature_data['yelping_since']

    feature_data['text_processed'] = feature_data.text.swifter.apply(lambda x: nlp(x))
    feature_data['tokens'] = feature_data.text_processed.swifter.apply(lambda x: get_tokens(x))

    feature_data['word_count'] = feature_data.tokens.swifter.apply(lambda x: len(x))
    feature_data['unique_words_perc'] = feature_data.tokens.swifter.apply(get_unique__perc)
    feature_data['adjectives'] = feature_data.tokens.swifter.apply(lambda x: [token for token in x if token.pos_ == 'ADJ'])
    feature_data['no_adjectives'] = feature_data.adjectives.swifter.apply(lambda x: len(x))
    feature_data['verbs'] = feature_data.tokens.swifter.apply(lambda x: [token for token in x if token.pos_ == 'VERB'])
    feature_data['perc_unique_verbs'] = feature_data.verbs.swifter.apply(get_unique__perc)
    feature_data['perc_unique_adjectives'] = feature_data.adjectives.swifter.apply(get_unique__perc)
    feature_data['norm_of_wordvecs'] = feature_data.tokens.swifter.apply(get_word_vectors)
    feature_data['adj_wordvecs'] = feature_data.adjectives.swifter.apply(get_word_vectors)
    feature_data['verb_wordvecs'] = feature_data.verbs.swifter.apply(get_word_vectors)

    feature_data['adj_maj_sent'] = feature_data.swifter.apply(lambda x: get_sentiment(x.adjectives, x.text_processed), axis=1)
    feature_data['verb_maj_sent'] = feature_data.swifter.apply(lambda x: get_sentiment(x.verbs, x.text_processed), axis=1)

    feature_data = feature_data.merge(pd.get_dummies(feature_data[['adj_maj_sent', 'verb_maj_sent']], prefix = ['adj_maj_sent_', 'verb_maj_sent']), \
                       left_index = True, right_index = True)

    # feather.write_dataframe(feature_data, os.getcwd() + 'feature_data.feather')

    return feature_data


# df = feather.read_dataframe(path)

def train_test_split(feature_data, feature_list):
    #x_data = feature_data.drop('useful_01', axis=1)
    #x_data = x_data[feature_list]
    #y_labels = feature_data['useful_01']
    # for whatever reason this error is thrown
    # TypeError: train_test_split() got an unexpected keyword argument 'stratify'
    # I am using sckit learn version '0.20.2', feature was introduced in 0.17
    # couldn't find any quick solution online
    #return train_test_split(x_data, y_labels, test_size=0.3, random_state=101 )#,stratify=y_labels)
    # workaround
    feature_data = feature_data[feature_list]
    test_data = feature_data.sample(frac = 0.3, random_state=101)
    idx_to_keep = [idx for idx in feature_data.index if not idx in test_data.index]
    training_data = feature_data.loc[idx_to_keep]
    print('#### Training Data ####')
    share_zeros_training = round(len(training_data['useful_01'][training_data['useful_01'] == 0]) / len(training_data) ,ndigits=2)
    share_ones_training = round( len(training_data['useful_01'][training_data['useful_01'] == 1]) / len(training_data), ndigits=2)
    print('0: ',share_zeros_training,' %')
    print('1: ', share_ones_training, ' %')
    print('#### Test Data ####')
    share_zeros_test = round(len(test_data['useful_01'][test_data['useful_01'] == 0]) / len(test_data), ndigits=2)
    share_ones_test = round(len(test_data['useful_01'][test_data['useful_01'] == 1]) / len(test_data), ndigits=2)
    print('0: ', share_zeros_test,' %')
    print('1: ', share_ones_test, ' %')
    #if not share_zeros_training==share_zeros_test or not share_ones_training == share_ones_test:
    #    raise ValueError('Distribution of classes differ in test and training set. Please run function again!')
    X_train = training_data.drop('useful_01', axis=1)
    X_test = test_data.drop('useful_01', axis=1)
    y_train = training_data['useful_01']
    y_test = test_data['useful_01']

    return X_train, X_test, y_train, y_test



def train_and_evaluate_model(training_x, training_y, test_x, test_y):
    cores = mp.cpu_count()
    model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state = 101, max_features=10, class_weight='balanced',n_jobs = cores)
    model.fit(training_x, training_y)
    predictions = model.predict(test_x)
    auc = roc_auc_score(test_y, predictions)
    pred_train = model.predict(training_x)
    auc_train = roc_auc_score(training_y, pred_train)
    print(classification_report(test_y, predictions))
    print(confusion_matrix(test_y, predictions, labels=[0,1]))
    print('AUC on TEST set is ', auc)
    print('AUC on TRAINING set is', auc_train)

    with open(os.getcwd() + '/' + 'Prediction_Results.txt', 'w') as text_file:
        text_file.write('AUC on test set is %s \n' % auc)

    # save the model to disk
    filename = 'finalized_model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    return

def main():
#def main(percentage_to_keep):
    #users = pd.read_csv('user.json.csv')
    #reviews = pd.read_csv('review.json.csv')
    #generate_labels(reviews, 0.6)
    #merged = merge_review_and_user_data(reviews, users)
    #del users
    #del reviews
    #gc.collect()
    #print('Share of data being used is %s percent of original dataset' % percentage_to_keep)
    #if percentage_to_keep is not None:
        #merged = downsample_data(merged, percentage_to_keep)
    merged = pd.read_pickle('merged_small.pkl')
    features = generate_features(merged)
    features_to_use = ['useful_01',
                       'stars',
                       'average_stars',
                       'compliment_more',
                       'compliment_hot',
                       'compliment_photos',
                       'compliment_writer',
                       'compliment_plain',
                       'fans',
                       'review_count',
                       'days_yelping',
                       'word_count',
                       'unique_words_perc',
                       'no_adjectives',
                       'perc_unique_verbs',
                       'perc_unique_adjectives',
                       'norm_of_wordvecs',
                       'norm_of_wordvecs',
                       'verb_wordvecs',
                       'adj_maj_sent__negative',
                       'adj_maj_sent__objective',
                       'adj_maj_sent__positive',
                       'adj_maj_sent__no_sentiment_assigned',
                       'verb_maj_sent_negative',
                       'verb_maj_sent_objective',
                       'verb_maj_sent_positive',
                       'verb_maj_sent_no_sentiment_assigned',
                       'useful_review_compliments_received_for_other_reviews'] # the one user who wrote the review
    X_train, X_test, y_train, y_test = train_test_split(features, features_to_use)
    train_and_evaluate_model(X_train, y_train, X_test, y_test)
    return



if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Train model to classify Yelp reviews as useful or not useful')
    #parser.add_argument('share_to_keep', metavar='s', type=float, nargs=1,
    #                    help='Determine how much data you want to use for model training and testing. The entire data set consists of 6 685 900 reviews.')
    #args = parser.parse_args()
    #main(share_to_keep=args.share_to_keep[0])
    main()



