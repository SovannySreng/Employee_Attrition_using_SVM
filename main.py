import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.eda import eda
from src.evaluation import metrics_score
from src.model_training import train_logistic_regression, train_svm
from src.visualizations import plot_histograms, plot_correlation_heatmap

def main():
    # Load and preprocess data
    df = load_data(r'H:\My Drive\BISI II\Data Science\Term Assignments\Employee_Attrition_using_SVM\data\HR_Employee_Attrition.xlsx')
    X, Y = preprocess_data(df)

    # Perform EDA
    eda(df)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)

    # Train Logistic Regression model
    lg_model = train_logistic_regression(x_train, y_train)

    # Evaluate Logistic Regression model
    y_pred_train_lg = lg_model.predict(x_train)
    y_pred_test_lg = lg_model.predict(x_test)
    print("Logistic Regression - Training Data")
    metrics_score(y_train, y_pred_train_lg)
    print("Logistic Regression - Test Data")
    metrics_score(y_test, y_pred_test_lg)

    # Train and Evaluate SVM models
    for kernel in ['linear', 'rbf', 'poly']:
        print(f"SVM with {kernel} kernel")
        svm_model = train_svm(x_train, y_train, kernel=kernel)
        y_pred_train_svm = svm_model.predict(x_train)
        y_pred_test_svm = svm_model.predict(x_test)
        print("SVM - Training Data")
        metrics_score(y_train, y_pred_train_svm)
        print("SVM - Test Data")
        metrics_score(y_test, y_pred_test_svm)

if __name__ == "__main__":
    main()