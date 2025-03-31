from src.data_utils import (
    mount_container, 
    load_data, 
    inspect_dataframe,
    convert_timestamps,
    extract_time_features,
    add_count_window_features,
    save_dataframe,
    get_top_20
)

from src.logger import get_logger
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pyspark.sql.functions import *

logger = get_logger(__name__)

def add_bar_trace(fig, df, x_col, y_col, col_pos, title):
    
    fig.add_trace(
        go.Bar(y=df[x_col].astype(str), x=df[y_col], name=title, orientation='h'),
        row=1, col=col_pos
    )

def create_top_20_charts(data_path, subplot_titles, is_attributed=None):
    
    df = load_data(data_path)
    if is_attributed is not None:
        df = df.filter(col("is_attributed") == is_attributed)
    
    fig = make_subplots(rows=1, cols=5, subplot_titles=subplot_titles)

    for i, col_name in enumerate(["app", "os", "channel", "device", "hour"]):
        top_data = get_top_20(df, col_name)
        add_bar_trace(fig, top_data, col_name, "count", i+1, f"Top {col_name.capitalize()}")

    fig.update_layout(
        title_text="Top Categories by Attributed Clicks",
        height=600,
        showlegend=False
    )
    fig.show()

def run_eda_pipeline():
    container_name = "data"
    storage_account_name = "frauddata160"
    mount_point = f"/mnt/{container_name}"
    train_input_path = f"{mount_point}/preprocess_feature_engg_60Mn.parquet/"
    test_output_path = f"{mount_point}/test_processed.parquet"

    storage_account_key = ""  

    logger.info("Starting preprocessing pipeline...")
    
    try:
        mount_container(
            container_name=container_name,
            storage_account_name=storage_account_name,
            mount_point=mount_point,
            storage_account_key=storage_account_key
        )
        
        create_top_20_charts(train_input_path, ["Top Apps", "Top OS", "Top Channels", "Top Devices", "Top Hours"])
        create_top_20_charts(train_input_path, ["Top Apps", "Top OS", "Top Channels", "Top Devices", "Top Hours"], is_attributed=1)
        create_top_20_charts(test_output_path, ["Top Apps", "Top OS", "Top Channels", "Top Devices", "Top Hours"])

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_eda_pipeline()
