# Employee Salary Prediction System

## Project Overview

The **Employee Salary Prediction System** is a Machine Learning project that predicts an estimated employee salary based on experience, education level, and job role.

The project uses a Random Forest Regression model to learn patterns from the provided dataset and generate salary predictions for new employee details.

## Objectives

- Predict estimated employee salaries using employee details.
- Apply data preprocessing and categorical feature encoding.
- Train and evaluate a Machine Learning regression model.
- Understand model performance using Mean Absolute Error (MAE).

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas
- **Model:** Random Forest Regressor
- **Development Environment:** Visual Studio Code

## Dataset Description

The project uses `salary.csv`, which contains the following columns:

| Column | Description |
|---|---|
| `experience_years` | Employee's years of experience |
| `education_level` | Employee's education qualification |
| `job_role` | Employee's job role |
| `salary` | Salary value used as the prediction target |

The supplied dataset contains 100 records and is intended for learning and demonstration purposes.

## Project Workflow

1. Load the salary dataset using Pandas.
2. Separate employee features and the target salary.
3. Encode categorical features using One-Hot Encoding.
4. Split the dataset into training and testing sets.
5. Train a Random Forest Regression model.
6. Evaluate the model using Mean Absolute Error.
7. Enter employee details to generate an estimated salary.

## Project Structure

```text
Employee-Salary-Prediction/
├── app.py
├── employee_salary.py
├── salary.csv
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Employee-Salary-Prediction
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Python prediction program

```bash
python employee_salary.py
```

### 4. Run the Streamlit web application

```bash
python -m streamlit run app.py
```

The Streamlit application allows users to enter employee details and view an estimated salary.

## Model Evaluation

The initial execution of the supplied Python program produced the following result:

- **Mean Absolute Error (MAE):** 31,323.32
- **Example Input:** 3 years of experience, BTech, Developer
- **Example Predicted Salary:** 477,133.10

These values are from the initial model run and may vary if the dataset, model configuration, or train-test split changes.

## Limitations

- The model is trained on a small, supplied demonstration dataset.
- Predictions depend on the patterns and quality of the available data.
- The predicted salary is an estimate, not a guaranteed market salary or job offer.
- The model's performance on real-world employee data has not been established.

## Future Enhancements

- Compare multiple regression models.
- Evaluate performance using additional regression metrics.
- Improve the dataset using reliable and representative data.
- Enhance the web application with additional visualizations.

## Acknowledgement

This project was developed as part of a summer internship learning activity using the provided dataset, source code, and project instructions.