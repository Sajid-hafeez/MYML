# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 02:37:31 2023

@author: Sajid
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from plotly import graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
def z_score_method(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_cols))
    filtered_entries = (z_scores < 3).all(axis=1)
    new_data = data[filtered_entries]
    return new_data

def iqr_method(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1
    new_data = data[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
    return new_data

def isolation_forest(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    clf = IsolationForest(contamination='auto')
    pred = clf.fit_predict(numeric_cols)
    new_data = data[pred == 1]
    return new_data

def lof_method(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    clf = LocalOutlierFactor()
    pred = clf.fit_predict(numeric_cols)
    new_data = data[pred == 1]
    return new_data

def dbscan_method(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    clustering = DBSCAN(eps=3, min_samples=2).fit(numeric_cols)
    labels = clustering.labels_
    new_data = data[labels != -1]
    return new_data

def winsorization_method(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    new_data_numeric = winsorize(numeric_cols, limits=[0.05, 0.05])
    new_data_numeric = pd.DataFrame(new_data_numeric, columns=numeric_cols.columns)
    for col in numeric_cols.columns:
        data[col] = new_data_numeric[col]
    return data

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    st.write( "Mean Squared Error (MSE): ", mse)
    st.write("Root Mean Squared Error (RMSE): ", rmse)
    st.write("Mean Absolute Error (MAE): ", mae)


def normalize(variables):
    """
    Normalize a list of variables using z-score normalization.
    
    Args:
        variables (list): A list of numerical variables.
        
    Returns:
        list: The normalized variables.
    """
    mean = np.mean(variables)
    std = np.std(variables)
    
    normalized_vars = [(x - mean) / std for x in variables]
    
    return normalized_vars

def detect_outliers(dataframe):
    """
    Detect outliers in a DataFrame using IQR method.

    Args:
        dataframe (DataFrame): A DataFrame.

    Returns:
        DataFrame: A DataFrame with the count of outliers for each column and their indices.
    """
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3 - Q1

    is_outlier = (dataframe < (Q1 - 1.5 * IQR)) | (dataframe > (Q3 + 1.5 * IQR))
    outliers = dataframe[is_outlier]

    # Get count of outliers in each column
    outliers_count = outliers.count()

    # Get indices of outliers in each column
    outliers_indices = {col: outliers.index[outliers[col].notnull()].tolist() for col in outliers.columns}

    return outliers_count, outliers_indices


def standardize(variables):
    """
    Standardize a list of variables using Standard Scaler.

    Args:
        variables (DataFrame): A DataFrame of numerical variables.

    Returns:
        DataFrame: The standardized variables.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    standardized_vars = scaler.fit_transform(variables)
    
    return standardized_vars


def main():
    st.title('My ML buddy in Progress')

    file = st.file_uploader("Upload your dataset", type=['csv'])
    
    if file is not None:
        data = pd.read_csv(file)
        task = st.sidebar.selectbox("Tasks", ["EDA","Preprocessing" ,"Machine Learning Models"])
        if task == "EDA":
            st.write("First five data rows")
            st.write(data.head())
            st.write("Missing values table")
            st.write(data.isna().sum())
            st.write("Data Shape")
            st.write(data.shape)
    #        st.write(data.info())
            st.write("Basic Statistical summary")        
            st.write(data.describe())
            st.header("Data Visualization")
            st.write("Numeric Variables")
    
            # Define your columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
    
            # Create a button in each column
            hist_button = col1.button("Histogram | Barplot")
            scatter_button = col2.button("Scatter Plot")
            scatter_ols_button = col3.button("Scatter Plot with ols")
            box_button = col4.button("Box Plot")
            heat_map_button = col5.button("Correlation Heatmap")
            violin_button = col6.button("Violin Plot")


            selected_columns = st.multiselect("Select Columns To Plot",data.columns)
    

    
            bins = st.slider("Number of bins for histogram", 5, 100, 20)
            if hist_button:
                  # Min: 5, Max: 100, Default: 20
                    
                if len(selected_columns) > 0:
                    df_melt = pd.melt(data[selected_columns])
                    fig, ax = plt.subplots(figsize=(10,10)) 
                    sns.histplot(data=df_melt, x='value', hue='variable', bins=bins, element="step", stat="density", common_norm=False, ax=ax)
                    plt.xticks(rotation=90)  # rotate x-axis labels for better visibility
                    st.pyplot(fig)


            elif scatter_button:
                if len(selected_columns) == 2:
                    fig = px.scatter(data, x=selected_columns[0], y=selected_columns[1])
                    st.plotly_chart(fig)
            elif scatter_ols_button:
                if len(selected_columns) == 2:
                    fig = px.scatter(data, x=selected_columns[0], y=selected_columns[1], trendline="ols")
                    st.plotly_chart(fig)
            elif box_button:
                if len(selected_columns) > 0:
                    fig = go.Figure()
                    for column in selected_columns:
                        fig.add_trace(go.Box(y=data[column], name=column))
                    st.plotly_chart(fig)
            elif heat_map_button:
                if len(selected_columns) > 1:
                    fig, ax = plt.subplots(figsize=(10,10)) 
                    correlation_matrix = data[selected_columns].corr().round(2)
                    sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
            elif violin_button:
                if len(selected_columns) > 0:
                    # Melt the DataFrame into a format where one or more columns are identifier variables
                    # and all other columns, considered measured variables, are unpivoted to the row axis,
                    # leaving just two non-identifier columns, "variable" and "value".
                    df_melt = pd.melt(data[selected_columns])
                    fig, ax = plt.subplots(figsize=(10,10)) 
                    sns.violinplot(x='variable', y='value', data=df_melt, ax=ax)
                    plt.xticks(rotation=90)  # rotate x-axis labels for better visibility
                    st.pyplot(fig)



        if task == 'Preprocessing':
            st.header("Welcome to preprocessing tab")
            st.write("Data Shape")
            st.write(data.shape)

            numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
            selected_columns1 = st.multiselect("Please choose the variables from normalization", numeric_columns)
            selected_data = data[selected_columns1].values  # Extract the selected data as a NumPy array
            normalized_data = normalize(selected_data)  # Apply normalization to the selected data
            data[selected_columns1] = normalized_data  # Update the DataFrame with the normalized values
            st.write(data.head())
            selected_columns1 = st.multiselect("Select independent variables for standardization", numeric_columns)
            if selected_columns1:  # Check if any variables were selected
                selected_data = data[selected_columns1]  # Extract the selected data as a DataFrame
                standardized_data = standardize(selected_data)  # Apply standardization to the selected data
                data[selected_columns1] = standardized_data  # Update the DataFrame with the standardized values
                st.write(data.head())
            else:
                st.warning("No independent variable selected")

            st.write("Note:  Normalization should be used when we hae different units across variables.")
            st.write("Note:  Standardization is a common preprocessing technique used in machine learning and data analysis that modifies numerical features so they have a mean of 0 and standard deviation of 1.")

            st.subheader('Outlier Detection')
            outliers_count, outliers_indices = detect_outliers(data[numeric_columns])
            st.write('Count of outliers in each selected column:')
            st.write(outliers_count)
            st.write('Indices of outliers in each selected column:')
            st.write(outliers_indices)
  
        
            st.header("Outlier Detection & Removal")
            st.write("The following buttons will detect the outliers with the mentioned method and also remove it from the data.")
            col1, col2, col3 = st.columns(3)
        
            with col1:
                if st.button('Z-Score Method'):
                    cleaned_data = z_score_method(data)
                    rows_before = data.shape[0]
                    rows_after = cleaned_data.shape[0]
                    rows_removed = rows_before - rows_after
                    data = cleaned_data
                    st.write(data)
                    st.write('Rows before cleaning: ', rows_before)
                    st.write('Rows after cleaning: ', rows_after)
                    st.write('Rows removed during cleaning: ', rows_removed)

        
            with col2:
                if st.button('IQR Method'):
                    cleaned_data = iqr_method(data)
                    rows_before = data.shape[0]
                    rows_after = cleaned_data.shape[0]
                    rows_removed = rows_before - rows_after
                    data = cleaned_data
                    st.write(data)
                    st.write('Rows before cleaning: ', rows_before)
                    st.write('Rows after cleaning: ', rows_after)
                    st.write('Rows removed during cleaning: ', rows_removed)

            with col3:
                if st.button('Isolation Forest'):
                    cleaned_data = isolation_forest(data)
                    rows_before = data.shape[0]
                    rows_after = cleaned_data.shape[0]
                    rows_removed = rows_before - rows_after
                    data = cleaned_data
                    st.write(data)
                    st.write('Rows before cleaning: ', rows_before)
                    st.write('Rows after cleaning: ', rows_after)
                    st.write('Rows removed during cleaning: ', rows_removed)
        
            col4, col5, col7 = st.columns(3)
        
            with col4:
                if st.button('Local Outlier Factor'):
                    cleaned_data = lof_method(data)
                    rows_before = data.shape[0]
                    rows_after = cleaned_data.shape[0]
                    rows_removed = rows_before - rows_after
                    data = cleaned_data
                    st.write(data)
                    st.write('Rows before cleaning: ', rows_before)
                    st.write('Rows after cleaning: ', rows_after)
                    st.write('Rows removed during cleaning: ', rows_removed)
        
            with col5:
                if st.button('DBSCAN Clustering'):
                    cleaned_data = dbscan_method(data)
                    rows_before = data.shape[0]
                    rows_after = cleaned_data.shape[0]
                    rows_removed = rows_before - rows_after
                    data = cleaned_data
                    st.write(data)
                    st.write('Rows before cleaning: ', rows_before)
                    st.write('Rows after cleaning: ', rows_after)
                    st.write('Rows removed during cleaning: ', rows_removed)
        

            with col7:
                if st.button('Reset to Orignal Data'):
                     data = data
                     st.write(data.head())
                     st.write('Shape of Data: ', data.shape)
                     
            st.header("Recoding-cat to numeric (One Hot Encoding)")
                     
            #########################################################################
        # Create multiselect menu for categorical columns
            categorical_cols = st.multiselect("Select the categorical columns you want to encode", data.columns)
        
            if categorical_cols:
        
                if st.button('One-Hot Encoding'):
                    # Perform One-Hot Encoding:
                    data = pd.get_dummies(data, columns=categorical_cols)
                    st.write('Data after one-hot encoding:')
                    st.write(data)
        
                if st.button('Label Encoding'):
                    # Create a label encoder object for Label Encoding
                    le = LabelEncoder()
        
                    for col in categorical_cols:
                        # Fit and transform the selected columns
                        data[col] = le.fit_transform(data[col])
                    
                    st.write('Data after label encoding:')
                    st.write(data)
            else:
                st.write("No categorical column selected!")
            #########################################################################
            data.to_csv('preprocessed_data.csv', index=False)
            

            ###########################
            
            
            
            
        if task == "Machine Learning Models":
            # Load your preprocessed data from the CSV file
            try:
                data = pd.read_csv('preprocessed_data.csv')
            except FileNotFoundError:
                st.warning("Please run preprocessing before training the model!")
                return
                   
            st.header("Machine Learning")
            st.write("Data Shape")
            st.write(data.shape)
            model_name = st.sidebar.selectbox("Select Model", ["Linear Regression", "KNN Regression", "Decision Tree Regression", "Random Forest Regression",  "KNN Classifier", "Decision Tree Classifier", "Random Forest Classifier", "Logistic Regression Classifier"])
            target = st.selectbox("Select the target variable", data.columns)
            selected_columns1 = st.multiselect("Select independent variables",data.columns)
            x = data[selected_columns1] #data.drop(target, axis=1)
            y = data[target]
            split_ratio = st.number_input('Enter a test/train split ratio', min_value=0.1, max_value=0.9, value=0.2, step=0.1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=42)

            if model_name == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                evaluate_regression(y_test, y_pred)
                b0 = model.intercept_
                b1 = model.coef_
                equation = "y = {:.2f} ".format(b0)
                for i, coef in enumerate(b1):
                    equation += "+ {:.2f}*{} ".format(coef, selected_columns1[i])
                st.write("Linear Regression Model Equation: ", equation)
                
                X_train_with_intercept = sm.add_constant(X_train)  # Add constant term to the features
                model_sm = sm.OLS(y_train, X_train_with_intercept)  # Create a statsmodels OLS model
                results = model_sm.fit()  # Fit the model
                st.write("Regression Results:")
                st.write(results.summary())
                #########################################################################
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            
                # Plotting residuals
                residuals = y_test - y_pred
                fig, ax = plt.subplots()
                sns.distplot(residuals)
                plt.xlabel('Residuals')
                st.pyplot(fig)
                #########################################################################
                
            elif model_name == "KNN Regression":
                model = KNeighborsRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                evaluate_regression(y_test, y_pred)
                
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            
                # Plotting residuals
                residuals = y_test - y_pred
                fig, ax = plt.subplots()
                sns.distplot(residuals)
                plt.xlabel('Residuals')
                st.pyplot(fig)
            
            # Note that in KNN Regression, we don't have the model equation and OLS regression results. 
            # KNN is a non-parametric model and doesn't provide coefficients like a linear regression model. 
            
            elif model_name == "Decision Tree Regression":
                model = DecisionTreeRegressor(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                evaluate_regression(y_test, y_pred)
            
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            
                # Plotting residuals
                residuals = y_test - y_pred
                fig, ax = plt.subplots()
                sns.distplot(residuals)
                plt.xlabel('Residuals')
                st.pyplot(fig)

                # Just as with KNN, we don't have the model equation and OLS regression results. 
                # Decision Tree is also a non-parametric model and doesn't provide coefficients like a linear regression model.
            
            elif model_name == "Random Forest Regression":
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                evaluate_regression(y_test, y_pred)
            
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            
                # Plotting residuals
                residuals = y_test - y_pred
                fig, ax = plt.subplots()
                sns.distplot(residuals)
                plt.xlabel('Residuals')
                st.pyplot(fig)

                
            elif model_name == "KNN Classifier":
                model = KNeighborsClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred)*100, "%")
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            

                # Confusion matrix
                cm = metrics.confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
                ax.set(xlabel="Predicted", ylabel="Actual", xticklabels=model.classes_, yticklabels=model.classes_, title="Confusion Matrix")
                st.pyplot(fig)
                report = classification_report(y_test, y_pred, target_names=model.classes_)
                st.text(report)

            elif model_name == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred)*100, "%")
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            
                # Confusion matrix
                cm = metrics.confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
                ax.set(xlabel="Predicted", ylabel="Actual", xticklabels=model.classes_, yticklabels=model.classes_, title="Confusion Matrix")
                st.pyplot(fig)
                report = classification_report(y_test, y_pred, target_names=model.classes_)
                st.text(report)

            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred)*100, "%")
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            
                # Confusion matrix
                cm = metrics.confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
                ax.set(xlabel="Predicted", ylabel="Actual", xticklabels=model.classes_, yticklabels=model.classes_, title="Confusion Matrix")
                st.pyplot(fig)
                report = classification_report(y_test, y_pred, target_names=model.classes_)
                st.text(report)

            elif model_name == "Logistic Regression Classifier":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred)*100, "%")
                # Plotting predicted vs actual values
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(fig)
            
                # Confusion matrix
                cm = metrics.confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
                ax.set(xlabel="Predicted", ylabel="Actual", xticklabels=model.classes_, yticklabels=model.classes_, title="Confusion Matrix")
                st.pyplot(fig)
                report = classification_report(y_test, y_pred, target_names=model.classes_)
                st.text(report)
                        
            

if __name__ == '__main__':
    main()
