
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df: pd.DataFrame):
    # Creating numerical columns
    num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears',
                'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']
    
    # Creating categorical variables
    cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']

    # Checking summary statistics
    print(df[num_cols].describe().T)

    # Creating histograms
    df[num_cols].hist(figsize=(14, 14))
    plt.show()

    # Printing the % sub categories of each category
    for i in cat_cols:
        print(df[i].value_counts(normalize=True))
        print('*' * 40)

    for i in cat_cols:
        if i != 'Attrition':
            (pd.crosstab(df[i], df['Attrition'], normalize='index') * 100).plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Percentage Attrition %')
            plt.show()

    # Mean of numerical variables grouped by attrition
    print(df.groupby(['Attrition'])[num_cols].mean())

    # Plotting the correlation between numerical variables
    plt.figure(figsize=(15, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
    plt.show()