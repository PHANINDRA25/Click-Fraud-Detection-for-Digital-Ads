from pyspark.ml.feature import VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.spark
from src.data_utils import load_data 
import numpy as np
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


logger = get_logger(__name__)
container_name = "data"
storage_account_name = "frauddata160"
mount_point = f"/mnt/{container_name}"
storage_account_key = "..."

def run_model_training():

    try:
        mount_container(
            container_name=container_name,
            storage_account_name=storage_account_name,
            mount_point=mount_point,
            storage_account_key=storage_account_key
        )

        mlflow.set_experiment("Click Ad Fraud Detection")
        mlflow.pyspark.ml.autolog()

        print("🔄 Loading data...")
        train_path = "/mnt/data/preprocess_feature_engg_60Mn.parquet"
        valid_path = "/mnt/data/validation.parquet"  # I have created validation data seperately maintaining the original target variable imbalance for hyperparameter tuning 
        
        train_df = load_data(train_path)
        valid_df = load_data(valid_path)

        drop_cols = ["is_attributed", "click_time", "attributed_time", "ip"]
        feature_cols = [col for col in train_df.columns if col not in drop_cols]

        # Assemble features
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        train_vector = vector_assembler.transform(train_df).select("features", "is_attributed")
        valid_vector = vector_assembler.transform(valid_df).select("features", "is_attributed")

        
        search_space = {
            "learningRate": hp.choice("learningRate", [0.01,  0.1]),
            "numLeaves": scope.int(hp.quniform("numLeaves", 5, 50, 1)),
            "minSumHessianInLeaf": hp.choice("minSumHessianInLeaf", [1, 5]),
            "maxDepth": scope.int(hp.quniform("maxDepth", 3,6,1))
            "lambdaL1": hp.uniform("lambdaL1", 0, 1),
            "lambdaL2": hp.uniform("lambdaL2", 0, 1)
        }

        run_counter = 0

        def objective(params):
            nonlocal run_counter
            run_counter += 1
            run_name = f"lightgbm_run_{run_counter}"

            with mlflow.start_run(run_name=run_name, nested=True):
                print(f" Run {run_counter} with params: {params}")

                lgbm = LightGBMClassifier(
                    labelCol="is_attributed",
                    featuresCol="features",
                    objective="binary",
                    metric="auc",
                    isUnbalance=True,
                    verbosity=-1,
                    **params
                )

                model = lgbm.fit(train_vector)
                predictions = model.transform(valid_vector)

                y_true = [int(row["is_attributed"]) for row in predictions.select("is_attributed").collect()]
                y_pred_proba = [float(row["probability"][1]) for row in predictions.select("probability").collect()]
                y_pred = [int(p >= 0.5) for p in y_pred_proba]

                # Log metrics
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred_proba)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", auc)

                print(f"✅ Run {run_counter}: AUC={auc:.4f}, F1={f1:.4f}")
                return {"status": STATUS_OK, "loss": -auc}

   
        with mlflow.start_run(run_name="lightgbm_hyperopt_sweep"):
            trials = Trials()
            best = fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials
            )

            print("Best hyperparameters found:", best)

    except Exception as e:
        logger.error(f"❌ Test preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    run_model_training()

