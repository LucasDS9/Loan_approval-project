# ** 🏦 Loan Approval Prediction**

Complete **Machine Learning** project focused on predicting the approval of bank loans.  
In addition to approval prediction, the project also performs **regression** to estimate the ideal **interest rate** for each approved customer, offering a complete solution for credit decision-making.

The problem addressed involves automatically and accurately identifying which customers are more likely to have their loan approved and understanding the factors that contribute the most to credit approval.

The solution helps financial institutions **reduce risk**, **automate processes**, **optimize offered interest rates**, and **improve operational efficiency** in their credit granting policies.

---

## 🎯 **Project Objectives**
- Build a model that correctly classifies clients into **approved** or **not approved**.
- Perform regression to predict the **interest rate** for approved customers.
- Identify variables with the greatest impact on the approval process.
- Create a complete overview, from EDA to predictive models and evaluation.

---

## 🧱 **Project Steps**

### 1️⃣ **Imports and Initial Understanding of the Dataset**
- Data loading
- First inspections: variable types, initial statistics, identification of inconsistencies
- General understanding of the dataset structure

---

### 2️⃣ **Exploratory Data Analysis (EDA)**
- Generation of charts for variables and their relationships with the *approval* variable
- Evaluation of numerical and categorical variable distributions
- Verification of patterns, trends, and possible outliers
- Analyses that help understand the behavior of approved vs non-approved clients

---

### 3️⃣ **Preprocessing**
- Cleaning and organizing the data  
- Handling missing values, including use of **KNN Imputer** when needed  
- Encoding categorical variables with **OrdinalEncoder**  
- Analysis of the **correlation matrix with the target** to identify the most relevant variables  
- Standardization and final preparation of the dataset for modeling  

---

### 4️⃣ **Training and Evaluation of the Model (Classification)**

The ideal model was selected using **GridSearchCV** and evaluated using classic classification metrics:

