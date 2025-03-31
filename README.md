## Business Problem

Companies market their products on social media through various pricing models such as fixed costs , PPC( pay-per-click) , PPI etc. Businesses invest significant amounts in digital advertising, paying social media platforms based on the engagement their content receives, which includes clicks, views, and interactions.

### What is Click Fraud?

Click fraud occurs when these engagement metrics are artificially inflated by deceptive means. This type of fraud involves bots or automated scripts that simulate real user clicks on ads without any genuine interest or intention to engage with the content. These fraudulent clicks result in increased advertising costs for companies without providing any real engagement or return on investment.

![](path)

### Recent Data (2024)

•	TikTok Ads: 740 fraudulent clicks per 1,000 (74% fraud rate)

•	Twitter/X Ads: 610 fraudulent clicks per 1,000 (61% fraud rate)

•	Facebook Ads: 520–570 fraudulent clicks per 1,000 (52–57% fraud rate)

Sources - https://www.wd-strategies.com/articles/click-fraud-in-2024-a-growing-threat-to-digital-advertising

Click fraud not only wastes financial resources but also skews data analytics. 30% of ad spend is wasted on fraudulent clicks. In 2018, it was noted that the cost of ad fraud was $19Bn and by 2030, we should see numbers which reach $100Bn. 

### Impact

Implementing this click fraud detection system can save millions in wasted ad spend for companies. By ensuring that each click is legitimate, advertisers can reallocate their budgets towards more effective marketing strategies and audience engagements.
An end-to-end Machine elarning training and inference pipeline are developed for the click ad fraud detection system. 

## Project Structure

The click ad fraud detection system is designed with an end-to-end machine learning training and inference pipeline,  leveraging the capabilities of PySpark and Azure Databricks to manage and analyze large-scale data effectively following a modular approach. incorporating best practices in MLOps. This includes the utilization of MLflow for comprehensive experiment tracking, model versioning, and seamless model registration, ensuring a robust, scalable, and reproducible machine learning lifecycle.

![image.png](path)

## Data Overview

The dataset comprises approximately 3GB of data spanning 60 million observations, both training and testing, and is characterized by a high imbalance typical of rare events scenarios: 99.8% of the instances are non-fraudulent clicks (negative), while only 0.2% represent fraudulent clicks (positive). The dataset includes several features essential for analysis, such as IP address, app, device, operating system, channel, click time, attributed time, and the binary target variable 'is_attributed' which idnicates whether a click is legit or fraudulent.

Tech stack used - Azure Databricks (for compute and model management), ADLS (data storage) , Pyspark (to handle big data)
The entire data processing and model training pipeline are built using PySpark, due to its ability to efficiently process large datasets. Using pandas for such large-scale data operations is impractical due to its limitations with big data.

![](path)

## Feature Engineering

Before building the model, it is crucial to develop a robust feature set, particularly when addressing the significant imbalance present in our dataset. Understanding the data points that can indicate whether a user's behavior is fraudulent is essential for crafting effective features. [referred  online sources to understand the behavioral patterns] 

- **Spike in Clicks from a Single Operating System:** An unusually high number of clicks from a single operating system within a short time frame can suggest a botnet operation. We track this by counting the number of clicks per OS, per hour, for each IP address.

- **Frequent Clicks from a Single IP on One Channel:** Repeated clicks on one advertising channel by a single IP address within an hour could indicate scripted clicking. We measure this using a count of clicks per channel, per hour, for each IP.

- **High Activity Across Multiple Apps from a Single IP:** If one IP is accessing multiple apps in unusually high volumes, it might be executing a fraud script across various platforms. We capture this by counting app interactions per hour for each IP.

To quantify these behaviors, additional features were created.

Hourly Clicks per IP by Day and Hour (nip_day_h): Counts the total clicks from an IP grouped by day and hour, helping identify sudden spikes in activity.

Hourly Clicks per IP by Channel (nip_h_chan): Provides insights into how frequently an IP interacts with specific channels within an hour.

Hourly Clicks per IP by OS (nip_h_osr): Measures the concentration of clicks from a specific operating system, per IP, per hour.

Hourly Clicks per IP by App (nip_h_app): Quantifies how active an IP is on different apps within the same hour.

Hourly Clicks per IP by Device (nip_h_dev): Tracks device-specific activities to detect anomalies in device usage patterns.


## Model Training 

When dealing with click ad fraud, which is a rare event, we can also use anomaly detection methods. However,  LightGBM is chosen here because it has excellent ability to handle categorical variables like IP, OS, channel, and hour in our data. LightGBM uses a method called integer encoding for these categories, avoiding the problems that come with one-hot encoding (increase in dimensioanlity) . It doesn't make our model overly complex or slow it down, which is crucial when working with big datasets. This makes LightGBM not just fast but also quite effective at predicting based on our specific needs.

Some important components are used during  model training and management.
MLflow, LightGBM from SynapseML, and Hyperopt are used to handle large-scale data efficiently while ensuring robust model performance and manageability.  

**LightGBM from SynapseML (mmlspark):** This distributed version of LightGBM runs on Apache Spark, allowing it to handle very large datasets by distributing computations across multiple nodes, unlike its regular version which is limited to single-machine environments.

**Hyperopt:** Hyperopt automates hyperparameter optimization using Bayesian techniques, scaling well across multiple nodes to integrate seamlessly with Spark environments for efficient parameter exploration.

**MLflow:** MLflow manages the entire ML lifecycle, tracking every model run's parameters and results, and facilitates model versioning and selection of the best model for production use.

![](path)


## Model Selection process & pushing to Production

After training, the top three models with the highest validation AUC were registered in MLflow's STAGING phase. To identify the best model for PRODUCTION, an independent holdout set, not previously used in training or validation, was employed. Each of these models was rigorously evaluated against this set, calculating key performance metrics such as AUC, accuracy, precision, recall, and F1-score. One of the models achieved an impressive AUC of 0.92 demonstrating its effectiveness for real-world application. This best-performing model is then promoted to the PRODUCTION phase in the MLflow Model Registry.

**_Note -_** Ideally, A/B testing of the three models could be conducted by exposing them to real-time traffic, allowing the best-performing model to be selected based on a combination of technical and business metrics. This approach is generally preferred for its real-world applicability. However, due to constraints on data availability, the current method involves evaluating the models against a holdout set. Additionally, unit tests can also be included to test the code quality.

![image.png](path)
![](path)

## INFERENCE Pipeline

The end-to-end workflow culminates with the application of the trained model to incoming batch data, assessing the likelihood of fraudulent clicks. First, this batch data is pre-processed according to the model’s requirements, ensuring compatibility and accuracy. Subsequently, the best-model, which is stored in the MLflow Model Registry PRODUCTION phase, is loaded and used to generate predictions that indicate the probability of fraud for each click.

## Extension (Real-time Prediction - Will complete this later)
The completed model can be deployed as a Fast API, enabling the capability for real-time predictions. This setup allows live traffic to flow directly to the Fast API, where it can instantly evaluate and score incoming data. Such a system provides timely insights into potential fraudulent activities, significantly enhancing the responsiveness of the fraud detection process




