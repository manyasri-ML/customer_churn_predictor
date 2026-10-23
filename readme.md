# Customer Churn Prediction

A Machine Learning web application that predicts whether a telecom customer is likely to leave a service provider using customer information and a trained Decision Tree classification model.

## Project Overview

Customer churn is a major challenge for telecom companies because losing customers affects revenue and business growth. Identifying customers who are likely to leave helps businesses take proactive steps to improve customer retention.

This project uses Python and Machine Learning to analyze customer data, train a Decision Tree classifier, evaluate its performance, and deploy the saved model through a Flask web application.

## Objectives

- Inspect and clean customer data.
- Perform Exploratory Data Analysis (EDA) to understand customer behavior.
- Build a Decision Tree classification model.
- Evaluate the model using classification metrics.
- Optimize model hyperparameters using GridSearchCV and cross-validation.
- Save the trained model for reuse.
- Develop a web interface to predict customer churn.

## Project Structure

```text
customer_churn_predictor/
│
├── 01_data/
│   ├── 01_raw/
│   │   └── customer_churn_data.csv
│   │
│   └── 02_processed/
│       └── customer_churn_data_cleaned.csv
│
├── 02_notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_decision_tree_classification.ipynb
│
├── 03_models/
│   └── decision_tree_model.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── check_model.py
├── gitignore
├── readme.md
└── requirement.txt
```

## Machine Learning Workflow

### 1. Data Inspection

- Load the original customer dataset.
- Examine the dataset shape, columns, and data types.
- Identify missing values, duplicates, and inconsistent entries.

### 2. Data Cleaning

- Handle missing or invalid values.
- Convert columns to appropriate data types.
- Prepare the cleaned dataset for further analysis.

### 3. Exploratory Data Analysis (EDA)

- Analyze customer churn distribution.
- Explore relationships between churn and customer tenure.
- Investigate monthly charges, contract types, and other customer characteristics.
- Identify patterns associated with customers leaving the service.

### 4. Data Preprocessing

- Separate input features from the target variable.
- Remove the customer ID column because it is an identifier rather than a useful predictive feature.
- Encode categorical variables into numerical representations.
- Split the dataset into training and testing sets.

### 5. Model Training

- Train a Decision Tree Classifier.
- Establish baseline model performance.
- Evaluate predictions on the test dataset.

### 6. Hyperparameter Tuning

- Use GridSearchCV to search for suitable Decision Tree hyperparameters.
- Apply cross-validation to evaluate candidate configurations.
- Select the best estimator based on the chosen scoring metric.

### 7. Model Evaluation

Evaluate the classifier using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Recall and F1-score are particularly useful for assessing churn detection because missing customers who are likely to leave can be costly for a business.

### 8. Model Saving

Save the trained model as a serialized `.pkl` file using Joblib so that the web application can load it without retraining the model on every request.

### 9. Web Application

Use Flask to create a web application that:

- Displays an HTML form for entering customer details.
- Accepts customer information from the user.
- Loads the saved model.
- Processes the submitted data in the format expected by the model.
- Displays the churn prediction.
- AI Acknowledgment

  AI tools were used to assist in generating the frontend code, including the HTML templates and user interface, as I am currently learning frontend development. The AI-assisted frontend was integrated with the Flask backend to create a functional web application for customer churn prediction.

  This project provided an opportunity to apply Machine Learning concepts and understand how a trained model can be integrated into a web application for real-world use.

## Technologies Used

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Programming language                   |
| Pandas           | Data manipulation and cleaning         |
| NumPy            | Numerical operations                   |
| Matplotlib       | Data visualization                     |
| Seaborn          | Exploratory data analysis              |
| Scikit-learn     | Model training, evaluation, and tuning |
| Joblib           | Saving and loading the trained model   |
| Flask            | Web application backend                |
| HTML/CSS         | Web interface                          |
| Jupyter Notebook | Experimentation and analysis           |
| Git and GitHub   | Version control and project hosting    |

## Installation and Setup

### Prerequisites

- Python installed on your system
- Visual Studio Code or another Python editor
- Git (optional, for version control)

### 1. Clone the Repository

```bash
git clone <customer_churn_predictor>
cd customer_churn_predictor
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```cmd
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirement.txt
```

### 4. Verify the Saved Model

Ensure that `03_models/decision_tree_model.pkl` exists before starting the application.

You can run the model verification script:

```bash
python check_model.py
```

### 5. Run the Flask Application

```bash
python app.py
```

Open the local URL displayed in your terminal, commonly:

```text
https://customer-churn-predictor-srum.onrender.com/
```

Enter the customer information on the webpage and submit the form to view the prediction.

## Dataset

The project uses a customer churn dataset containing information about telecom customers, such as:

- Customer demographics
- Tenure
- Phone and internet services
- Contract type
- Payment method
- Monthly and total charges
- Churn status

The raw dataset is stored in `01_data/01_raw/`, while the cleaned dataset is stored in `01_data/02_processed/`.

## Model Deployment Notes

The web application must use the same preprocessing steps, feature names, and feature order used during model training. Categorical encoding and missing-value handling must also remain consistent.

The saved model should be loaded from a project-relative path rather than a hard-coded path tied to a particular computer.

If the application uses a preprocessing pipeline saved with the classifier, load that complete pipeline. Otherwise, reproduce the training-time preprocessing correctly in the Flask backend.

Only load `.pkl` files from trusted sources because deserializing untrusted files can execute malicious code.

## Future Improvements

- Compare Decision Tree performance with Random Forest and other classification algorithms.
- Deploy the application to a cloud hosting platform.
- Develop customer retention recommendations based on churn risk.

##

**Rapelli Manyasri**

