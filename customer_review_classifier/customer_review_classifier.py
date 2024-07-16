import mlflow
import mlflow.keras

with mlflow.start_run():
    mlflow.log_param('epochs', 10)
    mlflow.log_param('batch_size', 32)

    history = best_model.fit(X_padded, y, epochs=10, validation_split=0.2)
    mlflow.log_metric('accuracy', max(history.history['val_accuracy']))
    
    mlflow.keras.log_model(best_model, "model")