import numpy as np 
import pandas as pd 
from tqdm import tqdm
tqdm.pandas()
import re
import matplotlib.pyplot as plt
np.random.seed(32)

import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer 
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Literal

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers    import Input
from tensorflow.keras.layers    import Conv1D, MaxPooling1D, LSTM, Bidirectional, Flatten, concatenate, Dropout, Input, Embedding, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models    import Model, Sequential
from tensorflow.keras.utils     import plot_model
from tensorflow.keras.utils import to_categorical

# 1) embed_size: the length of each word vector
embed_size = 300
# 2) features: unique words to use
max_features = 750000
# 3) maxlen: max number of words to use
maxlen = 100
# the number of samples to use for one update
batch = 64
# the max number of epochs to use
num_epochs = 3
# how many folds to use for cross validation
folds = 10
train = 'DA_train_labeled.tsv'
dev = 'DA_dev_labeled.tsv'
test = 'DA_test_unlabeled.tsv'
w2v_data = 'cbow_100.bin'

# read the data
def read_files(path):
    file = pd.read_csv(path, sep='\t')
    print ('shape', file.shape)
    return file

train_df = read_files(train)
dev_df = read_files(dev)
test_df = read_files(test)

# clean data
def normalize(text):
    normalized = str(text)
    normalized = re.sub('URL','',normalized) # remove links
    normalized = re.sub('USER','',normalized) # remove USER
    normalized = re.sub('#','',normalized) # remove #
    #normalized = re.sub('(@[A-Za-z0-9]+)_[A-Za-z0-9]+','',normalized) # remove @names with underscore
    #normalized = re.sub('(@[A-Za-z0-9]+)','',normalized) # remove @names
    #normalized = re.sub('pic\S+','',normalized) # remove pic.twitter.com links
    normalized = re.sub('\d+','',normalized) # remove numbers
    normalized = re.sub('-','',normalized) # remove symbols - . /
    normalized = re.sub('[a-zA-Z0-9]+','',normalized) # remove English words 
    normalized = re.sub('!','',normalized) # remove English words
    normalized = re.sub(':','',normalized) # remove English words
    normalized = re.sub('[()]','',normalized) # remove English words
    normalized = re.sub('☻','',normalized) # remove English words
    normalized = re.sub('[""]','',normalized) # remove English words
    normalized = re.sub('é','',normalized) # remove English words
    normalized = re.sub('\/','',normalized) # remove English words
    normalized = re.sub('؟','',normalized) # remove English words
    return normalized

train_df['#2_tweet'] = train_df['#2_tweet'].progress_apply(lambda text: normalize(text))
dev_df['#2_tweet'] = dev_df['#2_tweet'].progress_apply(lambda text: normalize(text))
test_df['#2_tweet'] = test_df['#2_tweet'].progress_apply(lambda text: normalize(text))

# data type alias where value must be 0 or 1
Binary = Literal[0, 1]

class LinguisticFeatureEncoder(DictVectorizer):
    """
    Encodes linguistic features defined in self
    """
    def __init__(self, **kwargs):        
        super().__init__(sparse=kwargs.get("sparse", False))
        self.use_negative_features = kwargs.get("use_negative_features", False)
        # all positive features
        self.pos_features: Dict[str, Callable[str, Binary]] = {
            # AFRICA
            #"egy_dem": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sدي\s', u'\sده\s', u'\sدى\s')) else 0,
            #"egypt_neg": lambda text: 1 if text.find(u'\sمش\s') >= 0 else 0,
            "tunis_iterog": lambda text: 1 if text.find(u'\sعلاش\s') >= 0 else 0,
            "tunis_degree": lambda text: 1 if text.find(u'\sبرشا\s') >= 0 else 0,
            "tunis_contextualword": lambda text: 1 if text.find(u'\sباهي\s') >= 0 else 0,
            "algeria": lambda text: 1 if text.find(u'\sكاش\s') >= 0 else 0,
            "mor_dem": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sديال\s', u'\sديالي\s', u'\sديالى\s')) else 0,
            "mauritania": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sكاغ\s', u'\sايكد\s')) else 0,
            "sudan": lambda text: 1 if text.find(u'\sياخ\s') >= 0 else 0,
            "somalia": lambda text: 1 if text.find(u'\sتناطل\s') >= 0 else 0,
            "dijubuti": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sهاد\s', u'\sهلق\s')) else 0,
            
            # ASIA
            "iraq_degree": lambda text: 1 if any(text.find(i) >=0 for i in (u' خوش ', u' كاعد ')) else 0, 
            #"iraq_dem": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sهاي\s', u'\sدا\s')) else 0, 
            #"iraq_degree": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sخوش\s', u'\sكاعد\s')) else 0, 
            #"iraq_adj": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sفدوه\s', u'\sفدوة\s')) else 0, 
            #"iraq_interrog": lambda text: 1 if text.find(u'\sشديحس\s') >= 0 else 0,
            #"iraq_tensemarker": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sهسه\s', u'\sهسع\s', u'\sلهسه\s')) else 0, 
            "saudi_dem": lambda text: 1 if text.find(u'\sكذا\s') >= 0 else 0,
            "qatar": lambda text: 1 if text.find(u'\sوكني\s') >= 0 else 0,
            "bahrain": lambda text: 1 if text.find(u'\sشفيها\s') >= 0 else 0,
            "emirates": lambda text: 1 if text.find(u'\sعساه\s') >= 0 else 0,
            "kuwait": lambda text: 1 if text.find(u'\sعندج\s') >= 0 else 0,
            "oman": lambda text: 1 if text.find(u'\sعيل\s') >= 0 else 0,
            "yemen": lambda text: 1 if text.find(u'\sكدي\s') >= 0 else 0,
            #"syria": lambda text: 1 if text.find(u'\sشنو\s') >= 0 else 0,
            #"palestine": lambda text: 1 if text.find(u'\sليش\s') >= 0 else 0,
            "jordan": lambda text: 1 if text.find(u'\sهاظ\s') >= 0 else 0,
            #"lebanon": lambda text: 1 if text.find(u'\sهيدي\s') >= 0 else 0,   
    
        }
        
    @property
    def size(self) -> int:
        return len(self.get_feature_names())
    
    def create_feature_dict(self, datum) -> Dict[str, Binary]:
        """
        Creates a feature dictionary of str -> 1 or 0.
        Optionally include negated forms of each feature (i.e., NOT_*)
        """
        # 1 if value == 0 else value)
        pos_features = dict((feat, fn(datum)) for (feat, fn) in self.pos_features.items())
        neg_features = dict()
        if not self.use_negative_features:
            return pos_features
        # assumes we're using positive features
        neg_features = dict((f"NOT_{feat}", not value) for (feat, value) in pos_features.items())
        return {**pos_features, **neg_features}
            
    def fit(self, X, y = None):
        dicts = [self.create_feature_dict(datum = datum) for datum in X]
        super().fit(dicts)
        
    def transform(self, X, y = None):
        return super().transform([self.create_feature_dict(datum) for datum in X])

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)


class DA(object):
    def __init__(self):
        self.tokenizer = self.tokenize()
        self.w2v, self.embedding_matrix = self.create_embeddings_matrix()
        self.encoder = LabelEncoder()
        self.ling_encoder, self.size = self.fit_ling()
        self.net = self.make_model()
        
    def tokenize(self):
        tk = Tokenizer(num_words=max_features)
        train_X = train_df["#2_tweet"]
        train_X = train_X.astype(str)
        tk.fit_on_texts(train_X)
        #print ('\nTokenizer is working: ', tk)
        return tk
    
    def prepare_text(self, x):
        tk = self.tokenizer
        x = tk.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=maxlen)
        return x
    
    def fit_ling(self):
        ling_encoder = LinguisticFeatureEncoder(use_negative_features=False)
        ling_encoder.fit(list(train_df['#2_tweet'].astype(str)))
        size = ling_encoder.size
        #print ('\nFitting the linguistic feature encoder: ', ling_encoder)
        #print ('\nThe linguistic feature size: ', size)
        return ling_encoder, size 

    def prepare_linguistic_text(self, l):
        lfe = self.ling_encoder
        l = lfe.transform(l)
        return l
            
    def prepare_labels (self, y):
        self.encoder.fit(y)
        y = self.encoder.transform(y)
        N_CLASSES = np.max(y) + 1
        y = to_categorical(y, N_CLASSES)
        #print('Shape of label tensor:', y.shape)
        return y
  
    def create_embeddings_matrix(self):
        tk = self.tokenizer
        print ('please wait ... loading the word embeddings')
        w2v = KeyedVectors.load_word2vec_format(w2v_data, binary=True, unicode_errors='ignore')
        print (w2v)
        my_dict = {}
        for index, key in enumerate(w2v.wv.vocab):
            my_dict[key] = w2v.wv[key]
        embedding_matrix = np.zeros((max_features, embed_size))
        for word, index in tk.word_index.items():
            if index > max_features - 1:
                break
            else:
                embedding_vector = my_dict.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        #print (embedding_matrix.shape)
        return w2v, embedding_matrix
    
    
    def make_model(self):
        ######## channel 1 ##########
        inputs_c1 = Input(maxlen,)
        embeddings_c1 = Embedding(max_features, embed_size, weights=[self.embedding_matrix],
                                input_length=maxlen,
                                trainable=True)(inputs_c1)
        #bi_c1 = Bidirectional(LSTM(300, return_sequences=True))(embeddings_c1)
        #drop_c1 = Dropout(0.5)(bi_c1)
        bidirectional_c1 = Bidirectional(LSTM(300))(embeddings_c1)
        flat_c1 = Flatten()(bidirectional_c1)
        ######## channel 2 ########## 
        inputs_c2      = Input(shape=(self.size,))
        embeddings_c2 = Embedding(self.size, embed_size, embeddings_initializer="uniform",
                                  embeddings_regularizer=None, activity_regularizer=None,
                                  embeddings_constraint=None, mask_zero=False, input_length=self.size,
                                  trainable=True)(inputs_c2)
        dense_c2   = Dense(100, activation="relu")(embeddings_c2)
        flat_c2       = Flatten()(dense_c2)
        
        # merge
        merged = concatenate([flat_c1, flat_c2])
        # interpretation
        hidden_c2_1   = Dense(100, activation="relu")(merged)
        hidden_c2_2   = Dense(100, activation="relu")(hidden_c2_1)
        outputs       = Dense(21, activation="softmax")(hidden_c2_2)#"softmax")(hidden_c2_2)
        
        model = Model(inputs=[inputs_c1, inputs_c2], outputs=outputs)
        # compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        #model.summary()
        return model

     
    def train(self):
        model = self.net
        # preparing for embeddings
        train_text = self.prepare_text(train_df['#2_tweet'])
        train_labels = self.prepare_labels(train_df['#3_country_label'])
        
        # prepare training for ling
        train_ling = self.prepare_linguistic_text(train_df['#2_tweet'].astype(str))
        #dev_data = lfe.transform(list(dev_df['#2_tweet'].astype(str)))
        #test_data = lfe.transform(list(test_df['#2_tweet'].astype(str)))
        
        # fit the model
        model.fit([train_text, train_ling], train_labels, epochs=num_epochs, verbose=1, batch_size=batch)

        
    def predict_dev(self):
        model = self.net
        dev_X = dev_df["#2_tweet"]
        dev_X = dev_X.astype(str)
        dev_text = self.prepare_text(dev_X)
        dev_ling = self.prepare_linguistic_text(dev_X)
        pred_dev_y = model.predict([dev_text, dev_ling], batch_size=50, verbose=1)

        # labels for the predicted dev data
        labels = np.argmax(pred_dev_y, axis=-1)  
        print('Labels are: ',labels)

        # getting the labels(inverse_transform)
        dev_y_predicted = self.encoder.inverse_transform(labels)
        print ('The length of predicted labels is: ', len(dev_y_predicted))

        # save labels to txt file
        with open("oop_3.txt", "w") as f:
            for s in dev_y_predicted:
                f.write(str(s) +"\n")
        
     
    
if __name__ == "__main__":
    da = DA()
    da.train()
    da.predict_dev()