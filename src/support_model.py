from tensorflow import keras
import tensorflow as tf




def f1_score(y_true, y_pred):
  '''
  Implement F1-score calculation here using y_true and y_pred tensors
  '''
  return tf.reduce_mean(f1(y_true, y_pred))