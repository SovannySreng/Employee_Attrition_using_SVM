import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df: pd.DataFrame):
    print(df.sample(5))
    print(df.info())
    print(df.describe().T)
    df.hist(figsize=(14,14))
    plt.show()
    cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']
    for col in cat_cols:
        if col != 'Attrition':
            pd.crosstab(df[col], df['Attrition'], normalize='index').plot(kind='bar', figsize=(8,4), stacked=True)
            plt.ylabel('Attrition Percentage')
            plt.show()