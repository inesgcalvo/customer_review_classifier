import re
import bz2
from langdetect import detect
from googletrans import Translator



translator = Translator()



def preprocess_line(line):
    '''
    Function to preprocess each line of the dataset
    '''
    # Extract labels and text
    labels_text = line.split(' ', 1)
    labels = labels_text[0].strip()
    text = labels_text[1].strip()
    
    # Extract review title from text
    match = re.match(r'([^:]+):\s+(.*)', text)
    if match:
        review_title = match.group(1)
        text = match.group(2)
    else:
        review_title = ''
    
    # Return labels, review title, and text
    return labels, review_title, text



def read_dataset(file_path):
    '''
    Function to read and process dataset
    '''
    with bz2.open(file_path, 'rt') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        labels, review_title, text = preprocess_line(line)
        data.append((labels, review_title, text))
    
    return data



def detect_language(text):
    try:
        first_two_words = ' '.join(text.split()[:10])
        return detect(first_two_words)
    except:
        return 'unknown'
    

def translate_text(text, translator):
    try:
        translation = translator.translate(text, src='es', dest='en')
        return translation.text
    except:
        return text