import mlflow
import mlflow.spark
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from src.data_utils import load_data
from src.data_utils import (
    mount_container, 
    load_data, 
    inspect_dataframe,
    convert_timestamps,
    extract_time_features,
    add_count_window_features,
    save_dataframe
)
from src.logger import get_logger


def evaluate_best_model():
    model_uri = "models:/click-ad-fraud-lightgbm/1"  # This si the best model - based on my validation data , I have chosen the model with highest auc which is 0.92 and saved it in model registry
    model = mlflow.spark.load_model(model_uri)

    test_path = "/mnt/data/test_processed.parquet"
    test_df = load_data(test_path)

    
    drop_cols = ["is_attributed", "click_time", "attributed_time", "ip"]
    feature_cols = [col for col in test_df.columns if col not in drop_cols]

    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    test_vector = assembler.transform(test_df)

    predictions = model.transform(test_vector)

    y_true = [int(row["is_attributed"]) for row in predictions.select("is_attributed").collect()]
    y_pred = [int(row["prediction"]) for row in predictions.select("prediction").collect()]
    y_prob = [float(row["probability"][1]) for row in predictions.select("probability").collect()]

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

if __name__ == "__main__":
    evaluate_best_model()
