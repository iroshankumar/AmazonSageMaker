
# 🧠 Bank Marketing Prediction using AWS SageMaker (XGBoost)

This project demonstrates a **complete end-to-end Machine Learning workflow on AWS SageMaker**, using the **Bank Marketing Dataset** to predict whether a customer will subscribe to a term deposit after a marketing campaign.

It covers the full ML lifecycle — from data acquisition, preprocessing, training, deployment, prediction, and evaluation — all inside **Amazon SageMaker** using **XGBoost**.

---

## 🚀 Project Overview

| Stage                   | Description                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| **1. Data Acquisition** | Downloaded and loaded the bank marketing dataset from an AWS public S3 source.               |
| **2. Data Preparation** | Cleaned, split, and transformed data for binary classification (`y_yes` as target).          |
| **3. AWS Setup**        | Created an S3 bucket and SageMaker session for storing training/test data and model outputs. |
| **4. Data Upload**      | Uploaded processed train/test CSVs to S3.                                                    |
| **5. Model Training**   | Trained an **XGBoost** model using SageMaker’s built-in XGBoost container.                   |
| **6. Model Deployment** | Deployed the trained model as a live HTTPS endpoint using SageMaker hosting services.        |
| **7. Inference**        | Sent real-time and batch predictions to the endpoint.                                        |
| **8. Evaluation**       | Computed confusion matrix, accuracy, recall, precision, F1, and ROC-AUC.                     |
| **9. Cleanup**          | Safely deleted the endpoint and S3 bucket to avoid costs.                                    |

---

## 🧩 Tech Stack

* **Language:** Python 3.10+
* **Platform:** Amazon SageMaker Notebook / Studio
* **Libraries:**

  * `boto3` — AWS SDK for Python
  * `sagemaker` — SageMaker Python SDK
  * `pandas`, `numpy`, `scikit-learn` — Data handling and metrics
  * `matplotlib` — Visualization
* **Algorithm:** XGBoost (`binary:logistic` objective)

---

## 🧱 Code Breakdown

### 1️⃣ Environment Setup

Initializes SageMaker session, IAM role, region, and defines S3 bucket/prefix:

```python
session = sagemaker.Session()
region = boto3.Session().region_name
role = sagemaker.get_execution_role()
bucket_name = 'bankapplication-12345'
prefix = 'bank-marketing-model'
```

### 2️⃣ Data Download and Load

Downloads the **Bank Marketing dataset** and loads it into a Pandas DataFrame:

```python
urllib.request.urlretrieve(url, "bank_clean.csv")
model_data = pd.read_csv("bank_clean.csv", index_col=0)
```

### 3️⃣ Train/Test Split

Splits the dataset into 80% training and 20% testing:

```python
train_data, test_data = train_test_split(model_data, test_size=0.2, random_state=42)
```

### 4️⃣ Save and Upload to S3

Prepares CSVs in XGBoost format (target as first column) and uploads to S3:

```python
pd.concat([train_data['y_yes'], train_data.drop(['y_no','y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
s3.Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
```

### 5️⃣ Define XGBoost Container and Hyperparameters

Retrieves the right Docker image for XGBoost and sets training parameters:

```python
container = image_uris.retrieve('xgboost', region=region, version='1.0-1')
hyperparameters = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "objective": "binary:logistic",
    "num_round": "50"
}
```

### 6️⃣ Train Model

Creates an **Estimator** and launches a managed training job on SageMaker:

```python
estimator = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.2xlarge',
    use_spot_instances=True,
    max_run=300,
    max_wait=600,
    output_path=f's3://{bucket_name}/{prefix}/output',
    hyperparameters=hyperparameters
)
estimator.fit({'train': s3_input_train, 'validation': s3_input_test})
```

### 7️⃣ Deploy the Trained Model

Creates a real-time inference endpoint:

```python
xgb_predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')
```

### 8️⃣ Make Predictions

Performs both single-sample and batch predictions:

```python
payload = ','.join(map(str, model_data.drop(['y_no','y_yes'], axis=1).iloc[0].tolist()))
response = xgb_predictor.predict(payload, initial_args={'ContentType': 'text/csv'})
```

For the full test set:

```python
from sagemaker.serializers import CSVSerializer
xgb_predictor.serializer = CSVSerializer()
predictions = []
for i in range(0, len(test_data_array), 100):
    batch = test_data_array[i:i+100]
    payload = '\n'.join([','.join(map(str, row)) for row in batch])
    response = xgb_predictor.predict(payload)
    predictions.extend(np.fromstring(response.decode('utf-8'), sep=','))
predictions_array = np.array(predictions)
```

### 9️⃣ Evaluate Model Performance

Computes accuracy, confusion matrix, ROC, and prints a formatted summary:

```python
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array),
                 rownames=['Observed'], colnames=['Predicted'])
print(cm)
```

**Results Example:**

```
Confusion Matrix:
 Predicted   0.0  1.0
Observed            
0          7193  110
1           748  187

Overall Classification Rate: 89.58%
Precision (Purchase): ~63%
Recall (Purchase): ~20%
```

### 🔟 Cleanup

Deletes deployed endpoint and S3 resources:

```python
xgb_predictor.delete_endpoint()
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
```

---

## 📊 Key Results

| Metric                        | Value                     |
| ----------------------------- | ------------------------- |
| **Accuracy**                  | 89.58%                    |
| **Specificity (No Purchase)** | 98.5%                     |
| **Recall (Purchase)**         | 20.0%                     |
| **Precision (Purchase)**      | 63.0%                     |
| **Model Type**                | Binary Logistic (XGBoost) |

---

## 🧹 Cleanup Reminder

Always delete endpoints and S3 buckets after running:

```python
xgb_predictor.delete_endpoint()
```

to avoid incurring extra AWS costs.

---

## 📘 Future Improvements

* Use **Hyperparameter Tuning** (SageMaker Automatic Model Tuning)
* Add `scale_pos_weight` to improve minority class recall
* Try advanced models (e.g., LightGBM, XGBoost V2)
* Add deployment automation via **SageMaker Pipelines**

---

## 🧑‍💻 Author

**Roshan Kumar**
AWS SageMaker | Machine Learning Engineer
📍 Built with Python, AWS SageMaker, and XGBoost
🗓️ *Project Completed: November 2025*
