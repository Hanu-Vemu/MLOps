import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import dagshub

# Load the dataset
df = pd.read_csv("data.csv")

# Dropping columns that are not needed
df = df.drop(columns=['id', 'Unnamed: 32'])

# Map the target to binary values: 'M' to 1 (malignant), 'B' to 0 (benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target datasets
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models and their hyperparameters
models = {
    "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=10000, random_state=8888),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "SVC": SVC(kernel='linear', C=1, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Initialize dagshub
dagshub.init(repo_owner='v.hanu85', repo_name='MLOps', mlflow=True)

# Set the MLflow experiment and tracking URI
mlflow.set_experiment("Anomaly Detection")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Loop through each model, train, predict, and log with MLflow
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Generate the classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Start a new MLflow run
    with mlflow.start_run(run_name=model_name):
        # Set tags
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("run_type", "experiment")
        
        # Log the model parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Log the metrics
        mlflow.log_metrics({
            'accuracy': class_report['accuracy'],
            'recall_class_0': class_report['0']['recall'],
            'recall_class_1': class_report['1']['recall'],
            'f1_score': class_report['macro avg']['f1-score']
        })
        
        # Log the model itself
        mlflow.sklearn.log_model(model, model_name)
