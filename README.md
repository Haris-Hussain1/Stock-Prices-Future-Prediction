# Tesla Stock Price Prediction using Linear Regression and Random Forest

## Project Overview

This project demonstrates **Machine Learning** techniques for predicting Tesla's **next-day closing stock price** using historical financial data. By implementing and comparing **Linear Regression** and **Random Forest Regressor** models, we explore how different algorithms perform on time-series stock prediction tasks.

###  What's Done
- **Fetch historical Tesla stock data** from Yahoo Finance (2018-2024)
- **Preprocess and prepare** the dataset for machine learning
- **Train two ML models**: Linear Regression and Random Forest
- **Compare model performance** using RMSE and other metrics
- **Predict next-day closing prices** with trained models

---

## Key Highlights

- **Real-world financial data** from Yahoo Finance
- **Two ML algorithms** compared side by side
- **Comprehensive evaluation** with multiple metrics
- **Visual analysis** with professional plots
- **Beginner-friendly** code and explanations
- **Portfolio-ready** project structure

---

## Objectives

1. **Data Acquisition**: Fetch historical Tesla stock data from Yahoo Finance
2. **Data Preprocessing**: Clean and prepare the dataset for ML training
3. **Model Training**: Train Linear Regression model for stock prediction
4. **Forest Training**: Train Random Forest Regressor for comparison
5. **Model Evaluation**: Compare both models using RMSE and other metrics
6. **Price Prediction**: Predict next day closing stock price accurately

---

## Technologies & Libraries Used

### Core Technologies
| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Main programming language | 3.8+ |
| **Google collab Notebook** | Development environment | Latest |

### Essential Libraries
| Library | Use Case | Badge |
|---------|----------|-------|
| **Pandas** | Data manipulation & analysis | ![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange) |
| **NumPy** | Numerical computations | ![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue) |
| **Matplotlib** | Data visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-blue) |
| **Seaborn** | Statistical visualization | ![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-red) |
| **Scikit-learn** | Machine learning algorithms | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-orange) |
| **yfinance** | Yahoo Finance data API | ![yfinance](https://img.shields.io/badge/yfinance-0.1%2B-green) |

---

## Dataset Information

### Data Source & Specifications
- **Source**: Yahoo Finance API (via `yfinance` library)
- **Company**: Tesla, Inc. (TSLA)
- **Time Period**: 2018 – 2024 (6 years of historical data)
- **Frequency**: Daily stock prices

### Features & Target
| Column | Type | Description |
|--------|------|-------------|
| **Open** | Feature | Opening price of the day |
| **High** | Feature | Highest price of the day |
| **Low** | Feature | Lowest price of the day |
| **Volume** | Feature | Trading volume for the day |
| **Close** | Target | **Next day's closing price** |

### Dataset Statistics
- **Total Records**: ~1,500+ trading days
- **Features**: 4 input variables
- **Target**: 1 output variable (next-day close)
- **Data Type**: Time series financial data

---

## Step-by-Step Workflow

### 1Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

### 2 Downloading Tesla Stock Data
```python
# Download Tesla stock data
ticker = 'TSLA'
start_date = '2018-01-01'
end_date = '2024-01-01'

# Fetch data from Yahoo Finance
tesla_data = yf.download(ticker, start=start_date, end=end_date)
print(f"Downloaded {len(tesla_data)} days of data")
```

### 3 Dataset Inspection
```python
# Display basic information
print("First 5 rows:")
print(tesla_data.head())

print("\nDataset Info:")
print(tesla_data.info())

print("\nStatistical Summary:")
print(tesla_data.describe())
```

### 4 Statistical Analysis
```python
# Correlation analysis
correlation_matrix = tesla_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

### 5 Creating Target Variable
```python
# Create target variable (next day's closing price)
tesla_data['Target'] = tesla_data['Close'].shift(-1)

# Remove last row (no target available)
tesla_data = tesla_data[:-1]

# Features and target
features = ['Open', 'High', 'Low', 'Volume']
X = tesla_data[features]
y = tesla_data['Target']
```

### 6 Sequential Train-Test Split
```python
# Sequential split for time series data (80% train, 20% test)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

### 7 Training Linear Regression
```python
# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)

print("Linear Regression model trained successfully!")
```

### 8 Evaluating Model
```python
# Calculate evaluation metrics
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)

print("Linear Regression Results:")
print(f"MAE: {lr_mae:.2f}")
print(f"MSE: {lr_mse:.2f}")
print(f"RMSE: {lr_rmse:.2f}")
```

### 9 Visualization
```python
# Plot actual vs predicted prices
plt.figure(figsize=(15, 8))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue', alpha=0.7)
plt.plot(y_test.index, lr_predictions, label='Predicted Price', color='red', alpha=0.7)
plt.title('Tesla Stock Price: Actual vs Predicted (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 10 Next-Day Prediction
```python
# Get the most recent data for prediction
latest_data = X.iloc[-1:].values
next_day_prediction = lr_model.predict(latest_data)

print(f"Predicted next-day closing price: ${next_day_prediction[0]:.2f}")
```

### 1.1 Training Random Forest
```python
# Initialize and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

print("Random Forest model trained successfully!")
```

### 1.2 Comparing Both Models
```python
# Calculate Random Forest metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)

print("Random Forest Results:")
print(f"MAE: {rf_mae:.2f}")
print(f"MSE: {rf_mse:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
```

---

## Model Evaluation Results

### Performance Comparison Table

| Metric | Linear Regression | Random Forest | Winner |
|--------|------------------|---------------|--------|
| **MAE**  | 6.59            | 8.19   | Linear Regression |
| **MSE**  | 89.51           | 160.69 | Linear Regression |
| **RMSE** | 9.46           | 12.67  | Linear Regression |

### Performance Analysis
- **Linear Regression** achieved **lower error metrics** across all evaluations
- **RMSE difference**: 3.21 points in favor of Linear Regression
- **Consistency**: Linear Regression showed more stable predictions

---

## Comparison Conclusion

### Why Linear Regression Performed Better

**Linear Regression** outperformed Random Forest in this stock prediction task due to several key factors:

####  **Linear Nature of Stock Trends**
- Stock prices often follow **trending patterns** that are well captured by linear relationships
- The relationship between opening, high, low prices and closing price tends to be **approximately linear**

#### **Simplicity Advantage**
- **Less prone to overfitting** on noisy financial data
- **Better generalization** to unseen market conditions
- **More stable predictions** across different time periods

####  **Random Forest Limitations**
- **Complex model** may overfit to noise in financial data
- **Higher variance** in predictions for volatile stocks like Tesla
- **Tree-based approach** less suited for continuous trend prediction

####  **Key Insight**
For **time-series financial prediction** with strong trending behavior, **simpler linear models** often outperform complex ensemble methods when the underlying relationships are predominantly linear.

---

##  Visualizations Section

### 1. Actual vs Predicted Prices
- **Line plot** comparing actual closing prices with model predictions
- **Time series visualization** showing prediction accuracy over time
- **Color-coded lines** for easy distinction between actual and predicted values

### 2. Model Comparison Graph
- **Side-by-side bar charts** comparing RMSE, MAE, and MSE
- **Visual performance metrics** for quick model assessment
- **Clear winner indication** through visual comparison

### 3. Feature Importance (Random Forest)
- **Bar plot** showing feature importance scores
- **Insights** into which features contribute most to predictions
- **Data-driven feature selection** guidance

---

##  Future Improvements

### Advanced Modeling Techniques
| Improvement | Description | Expected Benefit |
|-------------|-------------|------------------|
| **LSTM/Deep Learning** | Implement neural networks for sequence learning | Better capture of temporal patterns |
| **Technical Indicators** | Add RSI, MACD, Bollinger Bands | More predictive features |
| **Hyperparameter Tuning** | Grid search for optimal parameters | Improved model performance |
| **Ensemble Methods** | Combine multiple models | More robust predictions |

### System Enhancements
- ** Real-time Prediction System**: Live stock price prediction dashboard
- ** Mobile App**: User-friendly prediction interface
- ** Alert System**: Price movement notifications
- ** Portfolio Integration**: Multi-stock prediction capabilities

### Data Improvements
- ** Sentiment Analysis**: Incorporate news and social media sentiment
- ** Economic Indicators**: Add market-wide economic factors
- ** Alternative Data**: Include satellite imagery, supply chain data

---

##  Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Installation
```bash

# Install required packages
pip install yfinance pandas numpy matplotlib seaborn scikit-learn

# Run the main script
python tesla_prediction.py
```

### Alternative Installation
```bash
# Install packages individually
pip install yfinance
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

---

##  Sample Code Section

### Downloading Tesla Stock Data
```python
import yfinance as yf
import pandas as pd

def download_tesla_data(start_date='2018-01-01', end_date='2024-01-01'):
    """
    Download Tesla stock data from Yahoo Finance
    
    Parameters:
    - start_date: Start date for data collection (default: '2018-01-01')
    - end_date: End date for data collection (default: '2024-01-01')
    
    Returns:
    - DataFrame with Tesla stock data
    """
    
    # Download Tesla (TSLA) stock data
    tesla = yf.Ticker('TSLA')
    data = tesla.history(start=start_date, end=end_date)
    
    # Display basic information
    print(f" Successfully downloaded {len(data)} days of Tesla data")
    print(f" Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f" Columns: {list(data.columns)}")
    
    return data

# Example usage
if __name__ == "__main__":
    tesla_data = download_tesla_data()
    print("\nFirst 5 rows:")
    print(tesla_data.head())
```

### Quick Model Training Example
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def quick_prediction(data):
    """Quick example of model training and prediction"""
    
    # Prepare features and target
    features = ['Open', 'High', 'Low', 'Volume']
    X = data[features]
    y = data['Close'].shift(-1).dropna()
    X = X[:-1]  # Align with target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f" Model RMSE: {rmse:.2f}")
    
    return model

# Run the example
model = quick_prediction(tesla_data)
```

---

##  Project Structure

```
tesla-stock-prediction/
├──  README.md                 # Project documentation
├──  data/
│   ├──  tesla_stock_data.csv  # Downloaded stock data
│   └──  processed_data.csv    # Preprocessed data
├──  notebooks/
│   ├──  exploratory_analysis.ipynb
│   ├──  model_training.ipynb
│   └──  visualization.ipynb
├──  src/
│   ├──  data_loader.py       # Data downloading functions
│   ├──  preprocessing.py     # Data cleaning utilities
│   ├──  models.py            # ML model definitions
│   ├──  visualization.py     # Plotting functions
│   └──  prediction.py        # Main prediction pipeline
├──  models/
│   ├──  linear_regression.pkl
│   └──  random_forest.pkl
├──  outputs/
│   ├──  plots/
│   │   ├── actual_vs_predicted.png
│   │   └── model_comparison.png
│   └──  results/
│       └── model_metrics.csv
├──  requirements.txt          # Python dependencies
├──  main.py                   # Main execution script
├──  test_prediction.py       # Testing script
└──  config.py               # Configuration settings
```

---

##  Author Section

### **Haris Hussain**
 **AI/ML Intern** |  **Machine Learning Enthusiast**

 **Email**: harishussain631@gmail.com  


### About Me
Passionate AI/ML intern with expertise in **financial modeling**, **time series prediction**, and **machine learning algorithms**. This project demonstrates my ability to apply ML techniques to real-world financial data and deliver actionable insights through data-driven approaches.

### Skills Demonstrated
-  **Machine Learning**: Linear Regression, Random Forest
-  **Data Analysis**: Pandas, NumPy, Statistical Analysis
-  **Financial Modeling**: Stock price prediction, Time series
-  **Data Visualization**: Matplotlib, Seaborn
-  **Python Programming**: Clean, efficient code

---


### License Summary
 **Commercial use** allowed  
 **Modification** allowed  
 **Distribution** allowed  
 **Private use** allowed  

 **Liability**: No warranty provided  
 **Trademark**: No trademark rights granted

---

##  Acknowledgments

- ** Yahoo Finance** for providing reliable stock data API
- ** Scikit-learn Team** for excellent ML library
- ** Python Community** for amazing data science ecosystem
- ** Tesla Inc.** for being an interesting stock to analyze

---

##  Contact & Collaboration

Interested in **collaborating** or have **questions** about this project?

 **Email**: harishussain631@gmail.com  

---

##  Show Your Support

If this project helped you learn about **stock price prediction** or **machine learning**, please consider:

 **Starring this repository** on GitHub  
 **Forking** for your own experiments  
 **Sharing** with your network  

---

