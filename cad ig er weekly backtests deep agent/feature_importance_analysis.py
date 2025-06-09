import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Loading data and model...")

# Load the original data
data = pd.read_csv('~/Uploads/with_er_daily.csv')
print(f"Data shape: {data.shape}")

# Load the saved model with different encoding approaches
try:
    with open('best_trading_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
except Exception as e:
    print(f"First attempt failed: {e}")
    try:
        with open('best_trading_model.pkl', 'rb') as f:
            f.seek(0)
            model_data = pickle.load(f, encoding='latin1')
    except Exception as e2:
        print(f"Second attempt failed: {e2}")
        # Try to load with protocol 0
        with open('best_trading_model.pkl', 'rb') as f:
            f.seek(0)
            model_data = pickle.load(f, fix_imports=True, encoding='bytes')

print("Model data keys:", model_data.keys())

# Extract components
model = model_data['model']
feature_names = model_data['feature_names']
X_test = model_data['X_test']
y_test = model_data['y_test']

print(f"Number of features: {len(feature_names)}")
print(f"Test set shape: {X_test.shape}")
print(f"Model type: {type(model)}")

# Get model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
