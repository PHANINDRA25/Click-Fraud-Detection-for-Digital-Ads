import sys
import os
from databricks.sdk.runtime import dbutils

# Add parent directory of the src/ folder to Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

print("✅ Project root added to sys.path:", project_root)


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
import os


logger = get_logger(__name__)



container_name = "data"
storage_account_name = "frauddata160"
mount_point = f"/mnt/{container_name}"
path = f"{mount_point}/train_60Mn.parquet/"

storage_account_key = "" 


logger.info("Starting preprocessing pipeline...")

def run_preprocessing_pipeline():

    try:
        mount_container(
            container_name=container_name,
            storage_account_name=storage_account_name,
            mount_point=mount_point,
            storage_account_key=storage_account_key
        )

        train_df = load_data(path)
        train_df = convert_timestamps(train_df)
        inspect_dataframe(train_df)
        train_df = extract_time_features(train_df)
        inspect_dataframe(train_df)
        train_df = add_count_window_features(train_df)
        inspect_dataframe(train_df)
        output_path = f"{mount_point}/preprocess_feature_engg_60Mn.parquet"
        save_dataframe(train_df, output_path)


    except Exception as e:
        logger.error(f"❌ Preprocessing pipeline failed: {e}")
        raise

    logger.info("✅ Preprocessing script completed.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
