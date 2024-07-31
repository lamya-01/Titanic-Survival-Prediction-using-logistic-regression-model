# Titanic-Survival-Prediction-using-logistic-regression-model

This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset used for this project is from the Kaggle Titanic competition.

## Dataset

The dataset can be found [here](https://www.kaggle.com/competitions/titanic/data). The data is split into training and testing datasets. The training data is used to build the model, and the testing data is used to evaluate the model's performance.

## Project Steps

1. Data Loading: Load the dataset using pandas.
2. Data Preprocessing:
    - Fill missing values in the Age column with the mean.
    - Fill missing values in the Embarked column with the mode.
    - Drop irrelevant columns such as Name and PassengerId and Ticket.
    - Check and drop duplicate rows.
3. Data Splitting: Split the data into training and testing sets using train_test_split.
4. Model Building and Training: Create and train a Logistic Regression model.
5. Model Evaluation: Use the accuracy_score to evaluate the model's performance on the test data.

## Usage

### Prerequisites

- Python 
- Jupyter Notebook or Google Colab
- Required Python libraries: pandas, sklearn

### Instructions

1. Clone the repository:
   
    git clone <repository_url>
    cd <repository_name>
    
2. Install the required libraries:
   
    pip install pandas scikit-learn
    
3. Run the Jupyter Notebook or Colab:
    - If using Jupyter Notebook:
       
        jupyter notebook Titanic_Survival_Prediction.ipynb
        
    - If using Google Colab, open the provided link and run the cells sequentially.

## Project Steps in Detail

### Data Loading

Load the dataset using pandas to analyze and preprocess it.

### Data Preprocessing

Handle missing values:
- Fill missing values in the Age column with the mean age of the passengers.
- Fill missing values in the Embarked column with the mode (most frequent value) of the column.

### Drop Useless Columns

Remove columns that are not relevant to the prediction, such as Name and PassengerId and Ticket.

### Data Splitting

Separate the features and the target variable. Split the data into training and testing sets.

### Model Building and Training

Create and train a Logistic Regression model on the training data.

### Model Evaluation

Make predictions on the test set and evaluate the model's performance using metrics such as accuracy score.

## Conclusion

This project demonstrates a basic workflow for building and evaluating a machine learning model to predict Titanic passenger survival. Further improvements can be made by exploring more advanced feature engineering, hyperparameter tuning, and trying different machine learning algorithms.
