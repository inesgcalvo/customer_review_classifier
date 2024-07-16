# Import Libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Options
pd.options.display.float_format = '{:.2f}'.format

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

class Preprocessing:

    def __init__(self):
        self.next_step = None
    
    def set_next(self, step):
        self.next_step = step
    
    def process(self, data, VERBOSE):
        if self.next_step:
            return self.next_step.process(data, VERBOSE)
        return data
    
    @staticmethod
    def detect_language(text):
        from langdetect import detect

        try:
            first_two_words = ' '.join(text.split()[:10])
            return detect(first_two_words)
        except:
            return 'unknown'
        
    @staticmethod
    def translate_text(text, translator):
        try:
            translation = translator.translate(text, src='es', dest='en')
            return translation.text
        except:
            return text
        
    @staticmethod
    def text_preprocessing(data, NUM_WORDS, MAX_SEQ_LEN):
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(data['text'])
        sequences = tokenizer.texts_to_sequences(data['text'])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
        padded_sequences = np.array(padded_sequences, dtype=np.float32)
        return padded_sequences

    @staticmethod
    def encode_labels(data, VERBOSE):
        labels = pd.get_dummies(data['labels'], dtype=int)
        labels = np.array(labels['__label__1'], dtype=np.float32)
        if VERBOSE: print(f'\nLABELS:\n{labels[0]}\n') 
        return labels

class DataInspection(Preprocessing):

    def process(self, data, VERBOSE):
        if VERBOSE: 
            print('\n**View Data Structure**')
            print('\nHEAD')
            try:
                display(data.head())
            except:
                print(data.head())
            print('\nSAMPLE')
            try:
                display(data.sample(5))
            except:
                print(data.sample(5))
            print('\nTAIL')
            try:
                display(data.tail())
            except:
                print(data.tail())
            print('\n**Info Summary**\n')
            data.info(memory_usage='deep')
            print('\n**Summary Statistics**')
            try:
                display(data.describe())
            except:
                print(data.describe())

        return super().process(data, VERBOSE)

class HandleMissingValues(Preprocessing):

    def process(self, data, VERBOSE):
        if VERBOSE: 
            print('\n**Handle Missing Values**\n')
            print(data.isna().sum().sort_values(ascending = False))

        for column in data.columns:
            data[column].fillna('no title', inplace=True)

        if VERBOSE: 
            print('\nActual Missing Values\n')
            print(data.isna().sum().sort_values(ascending = False))

        return super().process(data, VERBOSE)

class RemoveDuplicates(Preprocessing):

    def process(self, data, VERBOSE):
        if VERBOSE: 
            print('\n**Remove Duplicates**')
        
        if data.duplicated().any():
            if VERBOSE:
                print('\nYes, there are duplicates.')
                print(f'SHAPE BEFORE: {data.shape}')
            data.drop_duplicates(inplace=True)
        else:
            if VERBOSE:
                print('\nThere are no duplicated values.')

        if VERBOSE:
            print(f'ACTUAL SHAPE: {data.shape}')

        return super().process(data, VERBOSE)

class TranslateText(Preprocessing):

    def process(self, data, VERBOSE):
        if VERBOSE:
            print('\n**Translate Text**')

        from googletrans import Translator
        translator = Translator()

        spanish_texts = []
        english_translations = []
        english_texts = []
        original_indices = []

        for idx, text in tqdm(data['text'].items(), total=len(data), desc='Translating Text'):

            if Preprocessing.detect_language(text) == 'es':
                spanish_texts.append(text)
                translation = Preprocessing.translate_text(text, translator)
                english_translations.append(translation)
                original_indices.append(idx)
                english_texts.append(translation)
            else:
                english_texts.append(text)

        translations_df = pd.DataFrame({
            'spanish_texts': spanish_texts,
            'english_translations': english_translations,
            'original_index': original_indices
        })

        data['text'] = english_texts

        translations_df.to_csv('../data/translations.csv', index=False)
        data.to_csv('../data/data_preprocessed.csv', index=False)

        return super().process(data, VERBOSE)

class HandleOutliers(Preprocessing):

    def process(self, data, VERBOSE):
        if VERBOSE:
            print('\n**Handle Outliers**')

        data['text_words'] = data['text'].apply(lambda x: len(x.split()))

        Q1 = np.percentile(data['text_words'], 25)
        Q3 = np.percentile(data['text_words'], 75)
        IQR = Q3 - Q1

        IQR_min_text_words = Q1 - 1.5 * IQR
        IQR_max_text_words = Q3 + 1.5 * IQR

        detected_outliers = data['text_words'][(data['text_words'] < IQR_min_text_words) | (data['text_words'] > IQR_max_text_words)]

        if VERBOSE:
            print(f'\nNumber of detected outliers: {len(detected_outliers)}')
            print(f'SHAPE BEFORE: {data.shape}')

        data = data[data['text_words'] >= IQR_min_text_words]
        data = data[data['text_words'] <= IQR_max_text_words]

        if VERBOSE:
            print(f'ACTUAL SHAPE: {data.shape}\n')

        data.reset_index(inplace=True)
        data.to_csv('../data/data_preprocessed.csv', index=False)

        return super().process(data, VERBOSE)

class SplitDataset(Preprocessing):

    def process(self, data, VERBOSE):
        from sklearn.model_selection import train_test_split

        padded_sequences = Preprocessing.text_preprocessing(data, NUM_WORDS=5000, MAX_SEQ_LEN=50)
        labels = Preprocessing.encode_labels(data, VERBOSE)

        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, 
            labels, 
            test_size=0.2, 
            random_state=42)
        
        if VERBOSE:
            print('X_train shape:', X_train.shape)
            print('X_test shape:', X_test.shape)
            print('y_train shape:', y_train.shape)
            print('y_test shape:', y_test.shape)

        self.save_to_csv(X_train, 'X_train.csv')
        self.save_to_csv(X_test, 'X_test.csv')
        self.save_to_csv(y_train, 'y_train.csv')
        self.save_to_csv(y_test, 'y_test.csv')

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def save_to_csv(data, filename):
        data_dir = '../data'
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)

        if data.ndim == 1:  
            data = data.reshape(-1, 1)

        try:
            np.savetxt(filepath, data, delimiter=',', fmt='%s')
        except IOError as e:
            print(f'Error saving data to {filename}: {e}')