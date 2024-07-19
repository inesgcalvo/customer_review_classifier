# Constants & Hyperparameters to define
VERBOSE = True
RANDOM_SEED = 42
EPOCHS = 10
BATCH_SIZE = 32

NUM_WORDS = 15000
EMBEDDING_DIM = 50
MAX_SEQ_LEN = 50
NUM_FILTERS = 64
KERNEL_SIZE = 5
NUM_CLASSES = 1

SAMPLE_FRAC = 0.5

# EXPERIMENT_NAME = 'exp.1'
# MLFLOW_TRACKING_URI = 'http://localhost:5000/'



# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Import Libraries
import pandas as pd
from tqdm.notebook import tqdm

import mlflow
import mlflow.keras
# from mlflow import MlflowClient
from mlflow.models import infer_signature


from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

# Disable XLA JIT Compilation
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import Functions
import sys
sys.path.append('../src')
from preprocessing import DataInspection, HandleMissingValues, RemoveDuplicates, TranslateText, HandleOutliers, SplitDataset

# Initialize Preprocessing Steps
data_inspection = DataInspection()
handle_missing_values = HandleMissingValues()
remove_duplicates = RemoveDuplicates()
translate_text = TranslateText()
handle_outliers = HandleOutliers()
split_dataset = SplitDataset()

# Chain Preprocessing Steps
data_inspection.set_next(handle_missing_values)
handle_missing_values.set_next(remove_duplicates)
remove_duplicates.set_next(translate_text)
translate_text.set_next(handle_outliers)
handle_outliers.set_next(split_dataset)

# Load Data
raw_data = pd.read_csv('../data/raw_data.csv')

# Sample the data
raw_data = raw_data.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED, ignore_index=True)

# Execute the pipeline
X_train, X_test, y_train, y_test = data_inspection.process(raw_data, VERBOSE)

padded_sequences = tf.convert_to_tensor(X_train, dtype=tf.float32)
labels = tf.convert_to_tensor(y_train, dtype=tf.float32)

# MLflow Logging and Model Training
with mlflow.start_run():

    # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.log_param('RANDOM_SEED', RANDOM_SEED)
    mlflow.log_param('EPOCHS', EPOCHS)
    mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
    mlflow.log_param('NUM_WORDS', NUM_WORDS)
    mlflow.log_param('MAX_SEQ_LEN', MAX_SEQ_LEN)
    mlflow.log_param('EMBEDDING_DIM', EMBEDDING_DIM)
    mlflow.log_param('NUM_FILTERS', NUM_FILTERS)
    mlflow.log_param('KERNEL_SIZE', KERNEL_SIZE)
    mlflow.log_param('NUM_CLASSES', NUM_CLASSES)
    mlflow.log_param('SAMPLE_FRAC', SAMPLE_FRAC)

    # Define the CNN model
    cnn_model = Sequential([
        Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
        Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        MaxPooling1D(pool_size=4, padding='same'),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(NUM_CLASSES, activation='sigmoid')
    ])

    # Compile the model
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])

    # Fit the model    
    history = cnn_model.fit(x=padded_sequences, y=labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # Metrics
    y_pred = cnn_model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    cr = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric('accuracy', cr.pop('accuracy'))
    for class_or_avg, metrics_dict in cr.items():
        for metric, value in metrics_dict.items():
            mlflow.log_metric(class_or_avg + '_' + metric, value)

    # Model Signature
    signature = infer_signature(X_train, cnn_model.predict(X_train))

    mlflow.keras.log_model(cnn_model, 'customer review classifier', signature=signature, registered_model_name='cnn_classifier')