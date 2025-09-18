# Titanic Data Analysis (Kaggle Project)

##  Project Overview
This project explores the famous **Titanic dataset** from Kaggle.  
The goal is to perform **Exploratory Data Analysis (EDA)**, clean the dataset,
and extract insights about survival rates based on passenger features.

##  Dataset
- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset?select=Titanic-Dataset.csv)
- **Rows**: ~891 passengers  
- **Columns**: 12 features (Name, Age, Sex, Ticket, Cabin, Embarked, etc.)

##  Steps
1. **Load & Inspect Data**  
   - Checked data shape, dtypes, missing values.
2. **Data Cleaning**  
   - Filled missing Age with mean.  
   - Filled missing Embarked with mode (S).  
   - Filled missing Cabin with most frequent (G6).  
   - Converted categorical features (Sex, Embarked, Ticket, Cabin, Name).
3. **Exploratory Analysis**  
   - Distribution of Age, Sex, Pclass.  
   - Survival rates by Sex and Class.  
   - Identified duplicates (none found).
4. **Key Insights**  
   - Female passengers had a much higher survival rate (233 survived vs. 109 males).  
   - Passengers in **1st class** had the highest survival chance.  
   - Most passengers embarked from **S port**.

##  Visualizations
- Survival by Sex  
- Survival rate by Class  
- Age distribution of passengers  

##  Next Steps
- Feature engineering (e.g., family size from SibSp & Parch).  
- Apply ML models for survival prediction (Logistic Regression, Random Forest).  
- Compare baseline vs. tuned models.

---
 Author: *NauRaa*  
 Date: *18 September 2025*
