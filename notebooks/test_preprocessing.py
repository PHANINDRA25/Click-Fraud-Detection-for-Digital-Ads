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


logger = get_logger(__name__)

container_name = "data"
storage_account_name = "frauddata160"
mount_point = f"/mnt/{container_name}"
test_input_path = f"{mount_point}/test.parquet/"
test_output_path = f"{mount_point}/test_processed.parquet"
storage_account_key = "..."  v

def run_preprocessing_pipeline():

    try:
        logger.info("🚀 Starting test data preprocessing pipeline")

        
        mount_container(container_name, storage_account_name, mount_point, storage_account_key)

        
        test_df = load_data(test_input_path)

        # Apply same preprocessing as train
        test_df = convert_timestamps(test_df)
        test_df = extract_time_features(test_df)
        test_df = add_count_window_features(test_df)

        
        inspect_dataframe(test_df)

        
        save_dataframe(test_df, test_output_path)

        logger.info("✅ Test data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"❌ Test preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    run_preprocessing_pipeline()

