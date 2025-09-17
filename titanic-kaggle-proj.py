# ===========================
# Titanic Data Analysis - EDA
# ===========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# 1. Load Data
df = pd.read_csv(r'E:\Rev-DataScience\titanic.csv')

# 2. Basic Info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.dtypes)

# ===========================
# Data Cleaning
# ===========================
# Convert categorical features
cat_cols = ['Embarked', 'Cabin', 'Ticket', 'Sex', 'Name']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Handle missing values
df['Age'] = df['Age'].fillna(round(df['Age'].mean()))
df['Cabin'] = df['Cabin'].fillna('G6')   # most frequent
df['Embarked'] = df['Embarked'].fillna('S')  # mode

print("Missing values after cleaning:\n", df.isnull().sum())

# ===========================
# Exploratory Analysis
# ===========================
# 1) Survival counts by sex
survived_female = df.loc[(df['Survived']==1) & (df['Sex']=='female'),'Survived'].sum()
survived_male = df.loc[(df['Survived']==1) & (df['Sex']=='male'),'Survived'].sum()

print(f"Female survived: {survived_female}")
print(f"Male survived: {survived_male}")

# 2) Survival by class
survived_class1 = df.loc[(df['Survived']==1) & (df['Pclass']==1),'Survived'].sum()
survived_class2 = df.loc[(df['Survived']==1) & (df['Pclass']==2),'Survived'].sum()
survived_class3 = df.loc[(df['Survived']==1) & (df['Pclass']==3),'Survived'].sum()

print(f"1st Class survived: {survived_class1}")
print(f"2nd Class survived: {survived_class2}")
print(f"3rd Class survived: {survived_class3}")

# ===========================
# Visualizations
# ===========================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Sex')
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(data=df, x='Pclass', y='Survived')
plt.title('Survival Rate by Pclass')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()


"""

Summary =>
The analysis shows that women survived at much higher rates than men (233 vs. 109).

First-class passengers had the best chance of survival compared to second and third class.

Next steps include feature engineering (e.g., family size) and building predictive ML models.

"""