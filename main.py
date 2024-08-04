from src.data_preprocessing import load_data, preprocess_data
from src.eda import eda
from src.model_training import train_logistic_regression, train_svm
from src.evaluation import metrics_score
from src.utils import setup_logging, log_error
from src.visualizations import plot_histograms, plot_categorical_distribution

def main():
    setup_logging()
    
    try:
        df = load_data('H:\My Drive\BISI II\Data Science\Term Assignments\Employee_Attrition_using_SVM\data\HR_Employee_Attrition.xlsx')
        
        # Exploratory Data Analysis
        eda(df)
        
        # Preprocessing Data
        x_train, x_test, y_train, y_test = preprocess_data(df)
        
        # Visualizations
        num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']
        cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']
        
        plot_histograms(df, num_cols)
        plot_categorical_distribution(df, cat_cols)
        
        # Training and Evaluating Logistic Regression
        lg = train_logistic_regression(x_train, y_train)
        y_pred_train = lg.predict(x_train)
        metrics_score(y_train, y_pred_train)
        y_pred_test = lg.predict(x_test)
        metrics_score(y_test, y_pred_test)
        
        # Training and Evaluating SVM (Linear Kernel)
        svm_linear = train_svm(x_train, y_train, kernel='linear')
        y_pred_train_svm = svm_linear.predict(x_train)
        metrics_score(y_train, y_pred_train_svm)
        y_pred_test_svm = svm_linear.predict(x_test)
        metrics_score(y_test, y_pred_test_svm)
        
    except Exception as e:
        log_error(e)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()