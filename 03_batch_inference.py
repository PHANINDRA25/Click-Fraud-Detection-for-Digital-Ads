from src.data_utils import (
    mount_container,
    load_data,
    convert_timestamps,
    extract_time_features,
    add_count_window_features,
    save_dataframe,
    inspect_dataframe
)

from src.logger import get_logger
from pyspark.ml.feature import VectorAssembler
import mlflow.spark


logger = get_logger(__name__)

container_name = "data"
storage_account_name = "frauddata160"
mount_point = f"/mnt/{container_name}"
test_input_path = f"{mount_point}/batch.parquet/"
storage_account_key = "..."  

def run_preprocessing_pipeline():

    try:
        logger.info("Starting test data preprocessing pipeline")

       
        mount_container(container_name, storage_account_name, mount_point, storage_account_key)

       
        test_df = load_data(test_input_path)

        # Apply same preprocessing as train
        test_df = convert_timestamps(test_df)
        test_df = extract_time_features(test_df)
        test_df = add_count_window_features(test_df)

        
        inspect_dataframe(test_df)


        logger.info("✅ Batch data preprocessing completed successfully.")

    def run_batch_prediction():
        model_uri = "models:/click-ad-fraud-lightgbm/1"
        model = mlflow.spark.load_model(model_uri)

        
        drop_cols = ["is_attributed", "click_time", "attributed_time", "ip"]
        feature_cols = [col for col in test_df.columns if col not in drop_cols]

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        test_vector = assembler.transform(test_df)

        predictions = model.transform(test_vector)
        predictions.select("prediction", "probability", "features").show(5)
        
        
        save_dataframe(predictions, "/mnt/data/batch_predictions.parquet")

    

    except Exception as e:
        logger.error(f"Test preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    run_preprocessing_pipeline()

