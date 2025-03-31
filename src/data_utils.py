import pandas as pd
from pyspark.sql import DataFrame
from src.logger import get_logger
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import dayofweek,hour
from pyspark.sql import functions as F
from pyspark.sql.window import Window
global dbutils
from pyspark.sql.functions import col, count

try:
    dbutils
except NameError:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)

logger = get_logger(__name__)


    

def mount_container(container_name, storage_account_name, mount_point, storage_account_key):
    """
    Mount an Azure Blob Storage container to DBFS.
    """
    source_url = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net"

    mounts = [mount.mountPoint for mount in dbutils.fs.mounts()]
    if mount_point not in mounts:
        logger.info(f"Mount point {mount_point} not found. Mounting now...")
        dbutils.fs.mount(
            source=source_url,
            mount_point=mount_point,
            extra_configs={
                f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
            }
        )
        logger.info(f"Mounted {source_url} to {mount_point}")
        logger.info(f"Mounted contents: {dbutils.fs.ls(mount_point)}")
    else:
        logger.info(f"Mount point {mount_point} already exists.")

def load_data(path: str):

    logger.info(f"Loading data from {path}")
    try:
        if path.endswith(".csv") or path.endswith(".csv/"):
            df = spark.read.option("header", True).csv(path)
        elif path.endswith(".parquet") or path.endswith(".parquet/"):
            df = spark.read.parquet(path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")

        logger.info(f"Loaded DataFrame with {df.count()} rows and {len(df.columns)} columns.")
        return df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def inspect_dataframe(df: DataFrame, num_rows: int = 5):

    logger.info(f"Inspecting DataFrame with up to {num_rows} rows...")
    df.show(num_rows)
    logger.info(f"DataFrame shape: ({df.count()}, {len(df.columns)})")
    logger.info(f"Columns: {df.columns}")
    df.printSchema()

def convert_timestamps(df: DataFrame) -> DataFrame:

    if "click_time" in df.columns:
        logger.info("Converting 'click_time' to timestamp...")
        df = df.withColumn('click_time', F.to_timestamp(df['click_time'], 'yyyy-MM-dd HH:mm:ss'))
    
    if "attributed_time" in df.columns:
        logger.info("Converting 'attributed_time' to timestamp...")
        df = df.withColumn('attributed_time', F.to_timestamp(df['attributed_time'], 'yyyy-MM-dd HH:mm:ss'))

    return df

def extract_time_features(df: DataFrame) -> DataFrame:

    logger.info("Extracting day of week and hour from 'click_time'...")

    df = df.withColumn('wday', dayofweek(df.click_time))
    df = df.withColumn('hour', hour(df.click_time))
    

    return df

def add_count_window_features(df: DataFrame) -> DataFrame:

    logger.info("Adding count-based window features...")

    # Define window specs
    w_nip_day_h = Window.partitionBy("ip", "wday", "hour")
    w_nip_h_chan = Window.partitionBy("ip", "hour", "channel")
    w_nip_h_osr = Window.partitionBy("ip", "hour", "os")
    w_nip_h_app = Window.partitionBy("ip", "hour", "app")
    w_nip_h_dev = Window.partitionBy("ip", "hour", "device")

    # Add window count features
    df = df.withColumn("nip_day_h", F.count("ip").over(w_nip_day_h))
    df = df.withColumn("nip_h_chan", F.count("ip").over(w_nip_h_chan))
    df = df.withColumn("nip_h_osr", F.count("ip").over(w_nip_h_osr))
    df = df.withColumn("nip_h_app", F.count("ip").over(w_nip_h_app))
    df = df.withColumn("nip_h_dev", F.count("ip").over(w_nip_h_dev))

    df.show(5)

    logger.info("Finished adding count-based window features.")
    return df

def save_dataframe(df: DataFrame, output_path: str, mode: str = "overwrite"):

    try:
        df.write.mode(mode).parquet(output_path)
        logger.info(f"✅ DataFrame written to: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to write DataFrame to {output_path}: {e}")
        raise


def get_top_20(df, col_name):
    return (
        df.groupBy(col_name)
        .agg(count("*").alias("count"))
        .orderBy(col("count").desc())
        .limit(20)
        .toPandas()
    )









