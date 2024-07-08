# Constants & Hyperparameters to define
RANDOM_SEED = 42
NUM_WORDS = 5000
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 50
NUM_FILTERS = 64
KERNEL_SIZE = 5
NUM_CLASSES = 2

######################################################################
######################################################################
######################################################################

# Import Libraries
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

# Options
pd.options.display.float_format = '{:.2f}'.format

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

######################################################################
######################################################################
######################################################################

class Preprocessing:

    def __init__(self):
        self.next_step = None
    
    def set_next(self, step):
        self.next_step = step
    
    def process(self, data):
        if self.next_step:
            return self.next_step.process(data)
        return data
    
    @staticmethod
    def detect_language(text):
        '''
        '''
        from langdetect import detect

        try:
            first_two_words = ' '.join(text.split()[:10])
            return detect(first_two_words)
        except:
            return 'unknown'
        
    def translate_text(text, translator):
        '''
        '''
        try:
            translation = translator.translate(text, src='es', dest='en')
            return translation.text
        except:
            return text
        
    def text_preprocessing(data):
        '''
        '''
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences


        text_data = data['text']

        # Text preprocessing
        tokenizer = Tokenizer(num_words=NUM_WORDS)
        tokenizer.fit_on_texts(text_data)
        sequences = tokenizer.texts_to_sequences(text_data)

        # Padding sequences
        sequences = pad_sequences(
            sequences, 
            maxlen=MAX_SEQ_LEN, 
            padding='post')

        return sequences

    def encode_labels(data):
        '''
        '''
        from tensorflow.keras.utils import to_categorical

        labels = pd.get_dummies(data['labels'], dtype=int)

        print(f'\nLABELS:\n{labels[:5]}\n')

        # One-hot encode labels
        labels_encoded = to_categorical(
            labels, 
            num_classes=NUM_CLASSES)

        return labels_encoded

######################################################################
######################################################################
######################################################################

class DataInspection(Preprocessing):

    def process(self, data):
        '''
        View Data Structure
        Info Summary
        Summary Statistics
        '''
        print('\n**View Data Structure**')
        print('\nHEAD')
        display(data.head())
        print('\nSAMPLE')
        display(data.sample(5))
        print('\nTAIL')
        display(data.tail())
        print('\n**Info Summary**\n')
        data.info(memory_usage='deep')
        print('\n**Summary Statistics**')
        display(data.describe())

        return super().process(data)

######################################################################
   
class HandleMissingValues(Preprocessing):

    def process(self, data):
        '''
        '''
        print('\n**Handle Missing Values**\n')
        print(data.isna().sum().sort_values(ascending = False))

        for column in data.columns:
            data[column].fillna('no title', inplace=True)

        print('\nActual Missing Values\n')
        print(data.isna().sum().sort_values(ascending = False))

        return super().process(data)

######################################################################

class RemoveDuplicates(Preprocessing):

    def process(self, data):
        '''
        '''
        print('\n**Remove Duplicates**')
        
        if data.duplicated().any():
            print('\nYes, there are duplicates.')
            print(f'SHAPE BEFORE: {data.shape}')
            data.drop_duplicates(inplace=True)
        else:
            print('\nThere are no duplicated values.')

        print(f'ACTUAL SHAPE: {data.shape}')

        return super().process(data)

######################################################################

class CorrectDataTypes(Preprocessing):

    def process(self, data):
        '''
        '''
        return super().process(data)

######################################################################

class TranslateText(Preprocessing):

    def process(self, data):
        '''
        '''
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

        # Create the DataFrame with the desired columns
        translations_df = pd.DataFrame({
            'spanish_texts': spanish_texts,
            'english_translations': english_translations,
            'original_index': original_indices
        })

        data['text'] = english_texts

        # Save the processed DataFrame
        translations_df.to_csv('../data/translations.csv', index=False)
        data.to_csv('../data/data_preprocessed.csv', index=False)

        return super().process(data)
    

######################################################################

class HandleOutliers(Preprocessing):

    def process(self, data):
        '''
        '''
        print('\n**Handle Outliers**')

        data['text_words'] = data['text'].apply(lambda x: len(x.split()))

        # Calculate IQR
        Q1 = np.percentile(data['text_words'], 25)
        Q3 = np.percentile(data['text_words'], 75)
        IQR = Q3 - Q1


        IQR_min_text_words = Q1 - 1.5 * IQR
        IQR_max_text_words = Q3 + 1.5 * IQR

        # Detect outliers using IQR method
        detected_outliers = data['text_words'][(data['text_words'] < IQR_min_text_words) | (data['text_words'] > IQR_max_text_words)]

        print(f'\nNumber of detected outliers: {len(detected_outliers)}')

        print(f'SHAPE BEFORE: {data.shape}')
        data = data[data['text_words'] >= IQR_min_text_words]
        data = data[data['text_words'] <= IQR_max_text_words]
        print(f'ACTUAL SHAPE: {data.shape}\n')

        data.reset_index(inplace=True)
        data.to_csv('../data/data_preprocessed.csv', index=False)

        return super().process(data)

######################################################################

class SplitDataset(Preprocessing):

    def process(self, data):
        '''
        '''
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            Preprocessing.text_preprocessing(data), 
            Preprocessing.encode_labels(data), 
            test_size=0.2, 
            random_state=RANDOM_SEED)

        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)

        # Save data as CSV files
        data_dir = os.path.join('..', 'data')
        os.makedirs(data_dir, exist_ok=True)

        data_list = [(X_train, 'X_train.csv'), (X_test, 'X_test.csv'), (y_train, 'y_train.csv'), (y_test, 'y_test.csv')]
        for data, filename in data_list:
            try:
                with open(os.path.join(data_dir, filename), 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerows(data.tolist() if isinstance(data[0], list) else data)
            except IOError as e:
                print(f'Error saving data to {filename}: {e}')

        print('\nData saved successfully to ../data directory in CSV format.')

        return X_train, X_test, y_train, y_test