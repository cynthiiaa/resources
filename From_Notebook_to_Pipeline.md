<div align="center">

# 📓 → 🚀 From Notebook to Pipeline

<img src="https://img.shields.io/badge/Level-Beginner%20to%20Advanced-9e6aed?style=for-the-badge" alt="Level Badge">
<img src="https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20by-cynscode.com-9e6aed?style=for-the-badge" alt="Made by cynscode.com">

### _The Complete Guide to Productionizing Your ML Code_

<p style="color: #9e6aed; font-size: 1.1em; font-weight: 500;">
Stop hearing "this will never make it to production" in code review 😤
</p>

</div>

---

## <span style="color: #9e6aed;">🎯 What You'll Learn</span>

<table>
<tr>
<td width="50%" style="border: 2px solid #9e6aed; padding: 15px; border-radius: 8px;">

**<span style="color: #9e6aed;">The Problem</span>**

Your notebook works perfectly on your laptop, but:

- 🤔 Hard to rerun with new data
- 😰 Impossible to test or debug
- 🚫 Can't deploy to production
- 📊 No way to monitor or log
- 🤖 "Works on my machine" syndrome

</td>
<td width="50%" style="border: 2px solid #9e6aed; padding: 15px; border-radius: 8px;">

**<span style="color: #9e6aed;">The Solution</span>**

A production-ready pipeline that:

- ✅ Runs automatically with new data
- ✅ Is tested and maintainable
- ✅ Can be deployed anywhere
- ✅ Has proper logging and monitoring
- ✅ Makes your team love you

</td>
</tr>
</table>

---

## <span style="color: #9e6aed;">📋 Table of Contents</span>

1. [The Anatomy of a Notebook](#the-anatomy-of-a-notebook)
2. [The 5-Step Conversion Framework](#the-5-step-conversion-framework)
3. [Step 1: Audit Your Notebook](#step-1-audit-your-notebook)
4. [Step 2: Extract & Modularize](#step-2-extract--modularize)
5. [Step 3: Add Configuration](#step-3-add-configuration)
6. [Step 4: Implement Error Handling & Logging](#step-4-implement-error-handling--logging)
7. [Step 5: Add Tests & Documentation](#step-5-add-tests--documentation)
8. [Complete Before & After Example](#complete-before--after-example)
9. [Production Deployment Patterns](#production-deployment-patterns)
10. [Bonus: Advanced Patterns](#bonus-advanced-patterns)

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 🔍 The Anatomy of a Notebook

> **Understanding what you're working with before you start converting**

</div>

Most ML notebooks follow this structure (yours probably does too):

```python
# 1. Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 2. Load Data
df = pd.read_csv('/Users/you/Desktop/data.csv')

# 3. Exploratory Data Analysis (EDA)
df.head()
df.describe()
df.plot()  # lots of visualizations

# 4. Data Cleaning
df = df.dropna()
df['feature'] = df['feature'].str.lower()

# 5. Feature Engineering
df['new_feature'] = df['feature1'] * df['feature2']

# 6. Train/Test Split
from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 7. Model Training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 8. Evaluation
score = model.score(X_test, y_test)
print(f"Accuracy: {score}")

# 9. Save Model
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```

**The problems with this approach:**

❌ Hardcoded paths everywhere  
❌ No error handling  
❌ Can't reuse functions  
❌ No logging (just print statements)  
❌ No tests  
❌ Magic numbers scattered throughout  
❌ Can't run on different environments

**Let's fix all of this! 💪**

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 🎯 The 5-Step Conversion Framework

> **A systematic approach that works every time**

</div>

<table style="width: 100%;">
<tr style="background: linear-gradient(135deg, #f5f0ff 0%, #e8dbff 100%);">
<td width="10%" align="center" style="padding: 15px;"><strong style="color: #9e6aed; font-size: 1.5em;">1</strong></td>
<td style="padding: 15px;"><strong style="color: #9e6aed;">Audit Your Notebook</strong><br>Identify all the moving parts and dependencies</td>
</tr>
<tr>
<td width="10%" align="center" style="padding: 15px;"><strong style="color: #9e6aed; font-size: 1.5em;">2</strong></td>
<td style="padding: 15px;"><strong style="color: #9e6aed;">Extract & Modularize</strong><br>Break code into reusable, testable functions</td>
</tr>
<tr style="background: linear-gradient(135deg, #f5f0ff 0%, #e8dbff 100%);">
<td width="10%" align="center" style="padding: 15px;"><strong style="color: #9e6aed; font-size: 1.5em;">3</strong></td>
<td style="padding: 15px;"><strong style="color: #9e6aed;">Add Configuration</strong><br>Externalize all parameters and paths</td>
</tr>
<tr>
<td width="10%" align="center" style="padding: 15px;"><strong style="color: #9e6aed; font-size: 1.5em;">4</strong></td>
<td style="padding: 15px;"><strong style="color: #9e6aed;">Implement Error Handling & Logging</strong><br>Make it production-ready and debuggable</td>
</tr>
<tr style="background: linear-gradient(135deg, #f5f0ff 0%, #e8dbff 100%);">
<td width="10%" align="center" style="padding: 15px;"><strong style="color: #9e6aed; font-size: 1.5em;">5</strong></td>
<td style="padding: 15px;"><strong style="color: #9e6aed;">Add Tests & Documentation</strong><br>Ensure reliability and maintainability</td>
</tr>
</table>

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 1️⃣ Step 1: Audit Your Notebook

</div>

### <span style="color: #9e6aed;">Create an inventory of everything in your notebook</span>

Use this checklist to identify all components:

**<span style="color: #9e6aed;">📊 Data Sources</span>**

- [ ] Where does data come from? (CSV, database, API, S3)
- [ ] What are the file paths or connection strings?
- [ ] How large is the data?
- [ ] How often does it update?

**<span style="color: #9e6aed;">🔧 Parameters & Hyperparameters</span>**

- [ ] What values are hardcoded? (test_size, n_estimators, learning_rate)
- [ ] Which parameters might change between runs?
- [ ] What are the magic numbers?

**<span style="color: #9e6aed;">🎯 Core Functions</span>**

- [ ] Data loading
- [ ] Data cleaning/preprocessing
- [ ] Feature engineering
- [ ] Model training
- [ ] Evaluation
- [ ] Prediction/inference

**<span style="color: #9e6aed;">📦 Dependencies</span>**

- [ ] List all imports
- [ ] Identify version-specific requirements
- [ ] Note any external tools or services

**<span style="color: #9e6aed;">💾 Outputs</span>**

- [ ] What artifacts are created? (models, plots, reports)
- [ ] Where are they saved?
- [ ] What format?

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 2️⃣ Step 2: Extract & Modularize

</div>

### <span style="color: #9e6aed;">Transform notebook cells into reusable functions</span>

#### **Before: Notebook Cell**

```python
# Cell 3: Load and clean data
df = pd.read_csv('/Users/you/Desktop/data.csv')
df = df.dropna()
df['text'] = df['text'].str.lower().str.strip()
df = df[df['age'] > 0]
df = df[df['age'] < 120]
print(f"Loaded {len(df)} rows")
```

#### **After: Modular Function**

```python
def load_and_clean_data(
    filepath: str,
    min_age: int = 0,
    max_age: int = 120
) -> pd.DataFrame:
    """
    Load and clean customer data.

    Args:
        filepath: Path to CSV file
        min_age: Minimum valid age
        max_age: Maximum valid age

    Returns:
        Cleaned DataFrame

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If DataFrame is empty after cleaning
    """
    logger.info(f"Loading data from {filepath}")

    df = pd.read_csv(filepath)
    initial_rows = len(df)

    # Remove missing values
    df = df.dropna()

    # Clean text columns
    df['text'] = df['text'].str.lower().str.strip()

    # Filter valid ages
    df = df[(df['age'] > min_age) & (df['age'] < max_age)]

    final_rows = len(df)
    logger.info(f"Cleaned data: {initial_rows} -> {final_rows} rows")

    if final_rows == 0:
        raise ValueError("No data remaining after cleaning")

    return df
```

### <span style="color: #9e6aed;">🎯 Key Improvements</span>

✅ **Type hints** - IDE can help with autocomplete  
✅ **Docstring** - Clear documentation  
✅ **Parameters** - No hardcoded values  
✅ **Logging** - Replaced print statements  
✅ **Error handling** - Fails gracefully  
✅ **Validation** - Checks output quality

---

### <span style="color: #9e6aed;">Project Structure After Extraction</span>

```
ml-project/
├── src/
│   ├── __init__.py
│   ├── data.py           # Data loading and preprocessing
│   ├── features.py       # Feature engineering
│   ├── models.py         # Model training and evaluation
│   └── utils.py          # Helper functions
├── config/
│   ├── config.yaml       # Configuration parameters
│   └── logging.yaml      # Logging configuration
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── notebooks/
│   └── exploration.ipynb # Keep for EDA only
├── scripts/
│   └── train.py          # Main training script
├── requirements.txt
└── README.md
```

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 3️⃣ Step 3: Add Configuration

</div>

### <span style="color: #9e6aed;">Externalize ALL hardcoded values</span>

#### **Create `config/config.yaml`**

```yaml
# Data Configuration
data:
  raw_data_path: "data/raw/customers.csv"
  processed_data_path: "data/processed/customers_clean.csv"
  train_test_split: 0.2
  random_state: 42

# Feature Engineering
features:
  numerical_features:
    - age
    - income
    - account_age
  categorical_features:
    - occupation
    - city
  text_features:
    - customer_feedback

# Model Configuration
model:
  type: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    random_state: 42

# Training Configuration
training:
  test_size: 0.2
  validation_size: 0.2
  cv_folds: 5

# Output Paths
output:
  model_path: "models/model.pkl"
  metrics_path: "results/metrics.json"
  plots_path: "results/plots/"

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### **Load Configuration in Code**

```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Usage
config = load_config()
data_path = config['data']['raw_data_path']
model_params = config['model']['hyperparameters']
```

#### **Or Use Environment Variables for Sensitive Data**

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
```

**`.env` file (never commit this!):**

```
DATABASE_URL=postgresql://user:pass@localhost:5432/db
API_KEY=your-secret-api-key
S3_BUCKET=my-ml-models
```

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 4️⃣ Step 4: Implement Error Handling & Logging

</div>

### <span style="color: #9e6aed;">Make it production-ready and debuggable</span>

#### **Set Up Logging**

```python
import logging
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure logging for the application."""

    # Create logs directory if needed
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

    return logging.getLogger(__name__)

# Usage
logger = setup_logging(log_level="INFO", log_file="logs/training.log")
logger.info("Starting training pipeline")
```

#### **Add Comprehensive Error Handling**

```python
from typing import Optional
import pandas as pd

def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load data with comprehensive error handling.

    Returns None if loading fails after retries.
    """
    try:
        logger.info(f"Attempting to load data from {filepath}")

        # Validate file exists
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Validate file extension
        if not filepath.endswith('.csv'):
            raise ValueError(f"Expected CSV file, got: {filepath}")

        # Load data
        df = pd.read_csv(filepath)

        # Validate data
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"CSV file is empty: {filepath}")

    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise ValueError(f"Invalid CSV format: {filepath}")

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}", exc_info=True)
        raise
```

#### **Add Data Validation**

```python
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    min_rows: int = 1
) -> None:
    """
    Validate DataFrame meets requirements.

    Raises:
        ValueError: If validation fails
    """
    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        logger.warning(f"Columns with all null values: {null_cols}")

    logger.info("DataFrame validation passed")
```

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 5️⃣ Step 5: Add Tests & Documentation

</div>

### <span style="color: #9e6aed;">Ensure reliability and maintainability</span>

#### **Write Unit Tests with pytest**

```python
# tests/test_data.py
import pytest
import pandas as pd
from pathlib import Path
from src.data import load_and_clean_data, validate_dataframe

@pytest.fixture
def sample_data(tmp_path):
    """Create sample CSV for testing."""
    data = pd.DataFrame({
        'age': [25, 30, -5, 150, 40],
        'text': ['Hello', 'World', 'Test', 'Data', 'Python'],
        'target': [0, 1, 0, 1, 0]
    })

    filepath = tmp_path / "test_data.csv"
    data.to_csv(filepath, index=False)
    return filepath

def test_load_and_clean_data(sample_data):
    """Test data loading and cleaning."""
    df = load_and_clean_data(sample_data, min_age=0, max_age=120)

    # Check invalid ages were removed
    assert len(df) == 3  # Should remove -5 and 150
    assert (df['age'] >= 0).all()
    assert (df['age'] <= 120).all()

def test_load_nonexistent_file():
    """Test that loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("nonexistent.csv")

def test_validate_dataframe():
    """Test DataFrame validation."""
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

    # Should pass
    validate_dataframe(df, required_columns=['col1', 'col2'], min_rows=1)

    # Should fail - missing column
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, required_columns=['col1', 'col3'])

    # Should fail - too few rows
    with pytest.raises(ValueError, match="minimum 10 required"):
        validate_dataframe(df, required_columns=['col1'], min_rows=10)
```

#### **Run Tests**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data.py

# Run tests matching pattern
pytest -k "test_load"
```

#### **Add Docstrings and Type Hints**

```python
from typing import Tuple, Optional
import pandas as pd
from sklearn.base import BaseEstimator

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    model_path: Optional[str] = None
) -> Tuple[BaseEstimator, dict]:
    """
    Train machine learning model.

    This function trains a model using the provided training data
    and hyperparameters. Optionally saves the trained model to disk.

    Args:
        X_train: Training features
        y_train: Training labels
        model_params: Dictionary of model hyperparameters
            Example: {'n_estimators': 100, 'max_depth': 10}
        model_path: Optional path to save trained model

    Returns:
        Tuple containing:
            - Trained model instance
            - Dictionary of training metrics

    Raises:
        ValueError: If training data is invalid
        RuntimeError: If model training fails

    Example:
        >>> model_params = {'n_estimators': 100}
        >>> model, metrics = train_model(X_train, y_train, model_params)
        >>> print(f"Training accuracy: {metrics['train_accuracy']:.3f}")
    """
    logger.info("Starting model training")

    # Implementation...

    return model, metrics
```

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 📊 Complete Before & After Example

</div>

### <span style="color: #9e6aed;">❌ Before: The Messy Notebook</span>

```python
# customer_churn_notebook.ipynb

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv('/Users/you/Desktop/churn_data.csv')
print(f"Loaded {len(df)} rows")

# Clean data
df = df.dropna()
df = df[df['age'] > 0]
df['tenure_months'] = df['tenure_months'].fillna(0)

# Feature engineering
df['revenue_per_month'] = df['total_revenue'] / (df['tenure_months'] + 1)
df['is_high_value'] = (df['total_revenue'] > 1000).astype(int)

# Split data
X = df[['age', 'tenure_months', 'revenue_per_month', 'is_high_value']]
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train accuracy: {train_score}")
print(f"Test accuracy: {test_score}")

# Save model
pickle.dump(model, open('churn_model.pkl', 'wb'))
print("Model saved!")
```

### <span style="color: #9e6aed;">✅ After: Production Pipeline</span>

**Project Structure:**

```
churn-prediction/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── pipeline.py
├── config/
│   └── config.yaml
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── scripts/
│   └── train.py
└── requirements.txt
```

**`src/data.py`**

```python
"""Data loading and preprocessing."""
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Load customer data from CSV."""
    logger.info(f"Loading data from {filepath}")

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df

def clean_data(
    df: pd.DataFrame,
    min_age: int = 0,
    max_age: int = 120
) -> pd.DataFrame:
    """Clean and validate customer data."""
    logger.info("Cleaning data")
    initial_rows = len(df)

    # Remove missing values
    df = df.dropna(subset=['age', 'churned'])

    # Filter valid ages
    df = df[(df['age'] >= min_age) & (df['age'] <= max_age)]

    # Fill missing tenure
    df['tenure_months'] = df['tenure_months'].fillna(0)

    final_rows = len(df)
    removed = initial_rows - final_rows
    logger.info(f"Removed {removed} invalid rows ({removed/initial_rows*100:.1f}%)")

    if final_rows == 0:
        raise ValueError("No data remaining after cleaning")

    return df
```

**`src/features.py`**

```python
"""Feature engineering."""
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for churn prediction."""
    logger.info("Creating features")

    df = df.copy()

    # Revenue per month
    df['revenue_per_month'] = df['total_revenue'] / (df['tenure_months'] + 1)

    # High value customer flag
    df['is_high_value'] = (df['total_revenue'] > 1000).astype(int)

    logger.info(f"Created {2} new features")

    return df

def get_feature_columns(config: dict) -> List[str]:
    """Get list of feature columns from config."""
    return config['features']['feature_columns']
```

**`src/models.py`**

```python
"""Model training and evaluation."""
import logging
import pickle
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict
) -> RandomForestClassifier:
    """Train Random Forest model."""
    logger.info(f"Training model with params: {model_params}")

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    logger.info("Model training complete")

    return model

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model performance."""
    logger.info("Evaluating model")

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    logger.info(f"Model metrics: {metrics}")

    return metrics

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """Save trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {filepath}")
```

**`src/pipeline.py`**

```python
"""Main training pipeline."""
import logging
from typing import Dict, Any

from sklearn.model_selection import train_test_split

from .data import load_data, clean_data
from .features import create_features, get_feature_columns
from .models import train_model, evaluate_model, save_model

logger = logging.getLogger(__name__)

def run_training_pipeline(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Execute complete training pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting training pipeline")

    # Load and clean data
    df = load_data(config['data']['raw_data_path'])
    df = clean_data(df)

    # Feature engineering
    df = create_features(df)

    # Prepare features and target
    feature_cols = get_feature_columns(config)
    X = df[feature_cols]
    y = df[config['data']['target_column']]

    # Split data
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train model
    model = train_model(X_train, y_train, config['model']['hyperparameters'])

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, config['output']['model_path'])

    logger.info("Pipeline complete")

    return metrics
```

**`scripts/train.py`**

```python
"""Training script entry point."""
import argparse
import logging
import yaml
from pathlib import Path

from src.pipeline import run_training_pipeline

def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Run pipeline
    try:
        metrics = run_training_pipeline(config)
        logger.info(f"Training complete. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
```

**Run the pipeline:**

```bash
python scripts/train.py --config config/config.yaml
```

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 🚀 Production Deployment Patterns

</div>

### <span style="color: #9e6aed;">1. Batch Prediction Pipeline</span>

```python
# scripts/predict.py
"""Batch prediction script."""
import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.data import load_data, clean_data
from src.features import create_features, get_feature_columns

def load_model(model_path: str):
    """Load trained model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_batch(
    data_path: str,
    model_path: str,
    output_path: str,
    config: dict
) -> None:
    """Run batch predictions."""
    logger = logging.getLogger(__name__)
    logger.info("Starting batch prediction")

    # Load data
    df = load_data(data_path)
    df = clean_data(df)
    df = create_features(df)

    # Prepare features
    feature_cols = get_feature_columns(config)
    X = df[feature_cols]

    # Load model and predict
    model = load_model(model_path)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Add predictions to dataframe
    df['churn_prediction'] = predictions
    df['churn_probability'] = probabilities

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # CLI implementation...
    pass
```

### <span style="color: #9e6aed;">2. REST API for Real-Time Serving</span>

```python
# api/app.py
"""FastAPI application for model serving."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import List

# Load model at startup
with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="Churn Prediction API")

class PredictionRequest(BaseModel):
    """Input schema for prediction."""
    age: int
    tenure_months: float
    total_revenue: float

class PredictionResponse(BaseModel):
    """Output schema for prediction."""
    churn_prediction: int
    churn_probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make churn prediction."""
    try:
        # Create features
        features = pd.DataFrame([{
            'age': request.age,
            'tenure_months': request.tenure_months,
            'revenue_per_month': request.total_revenue / (request.tenure_months + 1),
            'is_high_value': int(request.total_revenue > 1000)
        }])

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]

        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}
```

**Run the API:**

```bash
uvicorn api.app:app --reload --port 8000
```

**Test it:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "tenure_months": 12, "total_revenue": 1500}'
```

### <span style="color: #9e6aed;">3. Airflow DAG for Scheduled Training</span>

```python
# dags/train_churn_model.py
"""Airflow DAG for scheduled model training."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'train_churn_model',
    default_args=default_args,
    description='Train customer churn model',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'churn'],
)

# Extract fresh data
extract_data = BashOperator(
    task_id='extract_data',
    bash_command='python scripts/extract_data.py',
    dag=dag,
)

# Train model
train_model = BashOperator(
    task_id='train_model',
    bash_command='python scripts/train.py --config config/config.yaml',
    dag=dag,
)

# Validate model
validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model_performance,
    dag=dag,
)

# Deploy model
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command='python scripts/deploy_model.py',
    dag=dag,
)

# Set dependencies
extract_data >> train_model >> validate_model >> deploy_model
```

---

<div style="background: linear-gradient(90deg, #9e6aed 0%, #b89ef0 100%); padding: 20px; border-radius: 8px; color: white; margin: 20px 0;">

## 🎓 Bonus: Advanced Patterns

</div>

### <span style="color: #9e6aed;">1. Feature Store Integration</span>

```python
# src/feature_store.py
"""Feature store integration."""
from typing import List, Dict
import pandas as pd

class FeatureStore:
    """Simple feature store interface."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def save_features(
        self,
        df: pd.DataFrame,
        feature_group: str,
        entity_column: str
    ) -> None:
        """Save features to the feature store."""
        path = f"{self.storage_path}/{feature_group}.parquet"
        df.to_parquet(path, index=False)

    def load_features(
        self,
        feature_group: str,
        features: List[str],
        entity_ids: List[str] = None
    ) -> pd.DataFrame:
        """Load features from the feature store."""
        path = f"{self.storage_path}/{feature_group}.parquet"
        df = pd.read_parquet(path)

        if entity_ids:
            df = df[df['entity_id'].isin(entity_ids)]

        return df[features]
```

### <span style="color: #9e6aed;">2. Model Registry Pattern</span>

```python
# src/model_registry.py
"""Model versioning and registry."""
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class ModelRegistry:
    """Simple model registry for versioning."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a new model version."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save metadata
        metadata_doc = {
            'version': version,
            'model_name': model_name,
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat()
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata_doc, f, indent=2)

        return version

    def load_model(self, model_name: str, version: str = "latest"):
        """Load a specific model version."""
        if version == "latest":
            version = self._get_latest_version(model_name)

        model_path = self.registry_path / model_name / version / "model.pkl"

        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def _get_latest_version(self, model_name: str) -> str:
        """Get the latest model version."""
        versions = sorted([d.name for d in (self.registry_path / model_name).iterdir()])
        return versions[-1]
```

### <span style="color: #9e6aed;">3. Experiment Tracking with MLflow</span>

```python
# src/tracking.py
"""MLflow experiment tracking."""
import mlflow
from typing import Dict, Any

def train_with_tracking(
    X_train, y_train,
    model_params: Dict[str, Any],
    experiment_name: str = "churn_prediction"
):
    """Train model with MLflow tracking."""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_params)

        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Log metrics
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        return model
```

---

<div style="background: linear-gradient(135deg, #9e6aed 0%, #7c4fd4 100%); padding: 30px; border-radius: 12px; color: white; margin: 30px 0; box-shadow: 0 4px 15px rgba(158, 106, 237, 0.3);">

## 🎯 Key Takeaways

<table style="color: white; width: 100%;">
<tr>
<td width="50%" style="padding: 15px;">

**<span style="font-size: 1.2em;">✅ DO</span>**

- Break code into modular functions
- Use configuration files
- Implement proper logging
- Write tests for critical functions
- Add type hints and docstrings
- Handle errors gracefully
- Version your models

</td>
<td width="50%" style="padding: 15px;">

**<span style="font-size: 1.2em;">❌ DON'T</span>**

- Leave everything in notebooks
- Hardcode paths and parameters
- Use print() for debugging
- Skip testing "because it works"
- Ignore error handling
- Commit secrets to git
- Deploy without validation

</td>
</tr>
</table>

</div>

---

## <span style="color: #9e6aed;">🛠️ Recommended Tools</span>

<table>
<tr style="background: linear-gradient(135deg, #f5f0ff 0%, #e8dbff 100%);">
<td width="25%" align="center" style="padding: 20px;">
<strong style="color: #9e6aed; font-size: 1.2em;">Code Quality</strong><br><br>
• black / ruff<br>
• pylint<br>
• mypy<br>
• pre-commit
</td>
<td width="25%" align="center" style="padding: 20px;">
<strong style="color: #9e6aed; font-size: 1.2em;">Testing</strong><br><br>
• pytest<br>
• pytest-cov<br>
• hypothesis<br>
• tox
</td>
<td width="25%" align="center" style="padding: 20px;">
<strong style="color: #9e6aed; font-size: 1.2em;">Workflow</strong><br><br>
• MLflow<br>
• Weights & Biases<br>
• Airflow<br>
• Prefect
</td>
<td width="25%" align="center" style="padding: 20px;">
<strong style="color: #9e6aed; font-size: 1.2em;">Deployment</strong><br><br>
• FastAPI<br>
• Docker<br>
• GitHub Actions<br>
• AWS/GCP/Azure
</td>
</tr>
</table>

---

## <span style="color: #9e6aed;">📚 Further Reading</span>

- **Blog Posts** on [cynscode.com](https://cynscode.com)

  - [Getting Started with AWS CDK](https://cynscode.com/getting-started-with-aws-cdk/)
  - [A Guide to Python's Virtual Environment](https://cynscode.com/a-guide-to-pythons-virtual-environment/)
  - [The Importance of Documentation](https://cynscode.com/documentation/)

- **Books**

  - "Building Machine Learning Powered Applications" by Emmanuel Ameisen
  - "Designing Machine Learning Systems" by Chip Huyen
  - "Clean Code" by Robert Martin

- **Resources**
  - [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
  - [ML Ops Principles](https://ml-ops.org/)
  - [12 Factor App](https://12factor.net/)

---

## <span style="color: #9e6aed;">📬 Stay Connected</span>

<div align="center">

[![Blog](https://img.shields.io/badge/📝_Blog-cynscode.com-9e6aed?style=for-the-badge)](https://cynscode.com)
[![LinkedIn](https://img.shields.io/badge/💼_LinkedIn-Connect-9e6aed?style=for-the-badge)](https://www.linkedin.com/in/cynthiiaa/)

<!-- [![Newsletter](https://img.shields.io/badge/📧_Newsletter-Subscribe-9e6aed?style=for-the-badge)](https://cynscode.com/newsletter) -->

**Want more practical ML engineering content?** Follow along at [cynscode.com](https://cynscode.com) for weekly tips! 🚀

</div>

---

<div align="center" style="padding: 30px; background: linear-gradient(135deg, #f5f0ff 0%, #e8dbff 100%); border-radius: 12px; margin: 30px 0;">

<h3 style="color: #9e6aed; margin-bottom: 15px;">💜 Made with love by cynscode.com</h3>

<p style="color: #666; font-size: 1.1em;">
If this guide saved you from production nightmares, give it a ⭐️ and share it!
</p>

<p style="color: #9e6aed; font-weight: bold; margin-top: 20px; font-size: 1.2em;">
Your notebook → production pipeline transformation starts now! 🚀
</p>

</div>

---

<div align="center">
<sub>Last updated: October 2025 | Made for data professionals who want to ship real code</sub>
</div>
