# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 02:37:31 2023

@author: Sajid
"""
from sklearn.svm import SVR
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
import base64
# import streamlit as st
# import pandas as pd
# import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
st.set_page_config(
    page_title="My ML Buddy",
    page_icon="⌛",
    layout="centered",
    initial_sidebar_state="auto",
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

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
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def main():
    # Load Image
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "preprocessed_data" not in st.session_state:
        st.session_state["preprocessed_data"] = None
    add_bg_from_local('background.png')  
    st.title('My ML buddy')
    st.write("A video of MY ML Buddy in action is live [here](https://youtu.be/j7-GOT51-e0?si=G_AOHacjJo0A8T3Z)")
    #file = st.file_uploader("Upload your dataset", type=['csv'])
    file = st.file_uploader("Upload your dataset", type=['csv', 'txt', 'json', 'xlsx', 'xls'])
    if file is not None:
        file_details = {"FileName":file.name,"FileType":file.type,"FileSize":file.size}
        
        if file_details["FileType"] == "text/csv":
            data = pd.read_csv(file)
       #     st.write(data)
        elif file_details["FileType"] == "application/vnd.ms-excel" or file_details["FileType"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(file)
       #     st.write(data)
        elif file_details["FileType"] == "application/json":
            data = pd.read_json(file)
        #    st.write(data)
    
    # if file is not None:
    
    #     data = pd.read_csv(file)
        task = st.sidebar.selectbox("Tasks", ["EDA","Preprocessing" ,"Machine Learning Models","About Me"])
        if task == "EDA":
            st.header("Welcome to the Future of Predictive Analytics! 🚀.")
            st.write("In the constantly evolving digital world, the true power lies within data. It shapes strategies, inspires innovations, and propels businesses to greater heights. But unlocking the full potential of data can be a daunting task. That's where we come in! I am thrilled to welcome you to our revolutionary web application, designed to simplify and amplify the way you work with data. Whether you're a data science expert or just getting started, our application is your gateway to the exciting world of machine learning.The best part? It's absolutely FREE to use! Load your data, effortlessly preprocess it, implement a wide array of supervised machine learning models, and instantly visualize the results. Our user-friendly interface transforms complex data operations into a few clicks, making data more accessible than ever before. It's a tool built to empower you, to transform your decisions with the power of predictive analytics. Beyond this dynamic tool, we offer expert freelance data science services. With extensive experience in Python and R programming, we're here to support you through your data journey, from start to finish. From guidance on intricate data projects to hands-on coding assistance, consider us your go-to resource. So, why wait? Experience the power of data at your fingertips. Dive into the world of insights and predictive power right now! Remember, no matter where you are on your data journey, we're here to help you make the most of it. Don't just predict the future, define it with data. For more information, feel free to contact us. We look forward to embarking on this exciting journey with you! Let's turn data into decisions. Welcome to the future of predictive analytics! 🌐")
            st.write(file_details)
            st.write("First five data rows")
            st.write(data.head())
            st.write("Missing values table")
            st.write(data.isna().sum())
            st.write("Data Shape")
            st.write(data.shape)
    #        st.write(data.info())
            st.write("Basic Statistical summary")        
            st.write(data.describe())
            #########################################
            # st.write("Test code starts here")
            # st.write(data)
            # st.write("Test code ends here")
            
            
            ##########################################
            
            
            
            st.header("Data Visualization")
            st.write("Numeric Variables")
            selected_columns = st.multiselect("Select Columns To Plot",data.columns)
    
            # Define your columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
    
            # Create a button in each column
            hist_button = col1.button("Histogram | Barplot")
            scatter_button = col2.button("Scatter Plot")
            scatter_ols_button = col3.button("Scatter Plot with ols")
            box_button = col4.button("Box Plot")
            heat_map_button = col5.button("Correlation Heatmap")
            violin_button = col6.button("Violin Plot")


            

    
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
                    
            st.session_state["data"] = data

#########################################################################################################################################################################################################################################################################
# Pre-Processing
#########################################################################################################################################################################################################################################################################

        if task == 'Preprocessing':
            if st.session_state["data"] is not None:
                st.header("Welcome to preprocessing tab")
                st.write("Data Shape")
                st.write(data.shape)
                st.write('---------------------------------------------')
                st.header('Missing Values imputation')
                st.write('---------------------------------------------')
    
                st.subheader('Raw data')
                st.write(data)
            
                imputation_methods = ['Remove rows with missing values',
                                      'Mean for numerical, Mode for categorical',
                                      'Zero imputation',
                                      'Iterative (multivariate) imputation',
                                      'Most frequent/constant']
            
                # Show only those variables that have missing values
                missing_cols = data.columns[data.isna().any()].tolist()
                if not missing_cols:
                    st.write("No columns with missing values in the uploaded file")
                else:
                    st.subheader('Columns with missing values')
                    st.write(missing_cols)
            
                    # Choose column for imputation
                    selected_columns = st.multiselect('Choose columns for imputation', missing_cols)
            
                    
                    for column in selected_columns:
                        temp_data = data.copy()  # reset to original data each time
            
                        method = st.selectbox(f'Choose an imputation method for {column}', imputation_methods)
                        if method == 'Remove rows with missing values':
                            temp_data = temp_data.dropna()
                            data = data.dropna()
            
                        elif method == 'Mean for numerical, Mode for categorical':
                            if data[column].dtype == 'object':
                                temp_data[column].fillna(data[column].mode()[0], inplace=True)
                            else:
                                temp_data[column].fillna(data[column].mean(), inplace=True)
            
                        elif method == 'Zero imputation':
                            temp_data[column].fillna(0, inplace=True)
            
                        elif method == 'Iterative (multivariate) imputation':
                            imp = IterativeImputer(max_iter=10, random_state=0)
                            temp_data[column] = imp.fit_transform(data[[column]])
            
                        elif method == 'Most frequent/constant':
                            temp_data[column].fillna(data[column].value_counts().index[0], inplace=True)
            
                        data[column] = temp_data[column]  # update only the specific column
            
                        st.subheader('Imputed data')
                        st.write(data)            
              #  del temp_data
                ##################################################################
    
                numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
                st.write('---------------------------------------------')
                st.header('Normalization')
                st.write('---------------------------------------------')
                selected_columns1 = st.multiselect("Please choose the variables from normalization", numeric_columns)
                selected_data = data[selected_columns1].values  # Extract the selected data as a NumPy array
                normalized_data = normalize(selected_data)  # Apply normalization to the selected data
                data[selected_columns1] = normalized_data  # Update the DataFrame with the normalized values
                st.write(data.head())
                st.write('---------------------------------------------')
                st.header('Standardization')
                st.write('---------------------------------------------')
    
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
                st.write('---------------------------------------------')
                st.header('Outliers index based on IQR method')
                st.write('---------------------------------------------')
    
                outliers_count, outliers_indices = detect_outliers(data[numeric_columns])
                st.write('Count of outliers in each selected column:')
                st.write(outliers_count)
                st.write('Indices of outliers in each selected column:')
                st.write(outliers_indices)
                st.write('---------------------------------------------')
                st.header('One Hot Encoding & Label Encoding')
                st.write('---------------------------------------------')
                
                one_hot_cols = st.multiselect("Select the categorical columns you want to one-hot encode", data.columns)
                label_cols = st.multiselect("Select the categorical columns you want to label encode", data.columns)
                
                if one_hot_cols:
                    # Perform One-Hot Encoding:
                    data = pd.get_dummies(data, columns=one_hot_cols)
                    st.write('Data after one-hot encoding:')
                    st.write(data)
                
                if label_cols:
                    # Create a label encoder object for Label Encoding
                    le = LabelEncoder()
                
                    for col in label_cols:
                        # Fit and transform the selected columns
                        data[col] = le.fit_transform(data[col])
                    st.write('Data after label encoding:')
                    st.write(data)
    
    
      ##############################################################
                st.dataframe(data)  # Use dataframe() to make it more pretty
                st.write('---------------------------------------------')
                st.header('Column creation based on existing variables')
                st.write('---------------------------------------------')
                st.write('You can create new variable based on the mathematical operations from pervious variables')
                st.write('It supports ADDITION, SUBSTRACTION, DIVIDE, MULTIPLICATION, EXPONENTIAL, EXP/LOG, ABSOLUTE, TRANOMETRIC FUNCTIONS AND CONDITIONS')
                st.write('E.g.New variable can be created from equations like this')
                st.markdown("if equation is: $new\_variable = 3 \cdot var1 + var2^2$")
                st.write("You can write this in formula 3* var1 + var2**2")
                # Input for new variable
                new_var = st.text_input('Enter new variable name')
        
                # Input for formula
                formula = st.text_input('Enter formula for new variable (use var names from the data)')
        
                if new_var and formula:  # Only try to create the column if the variable name and formula are provided
                    try:
                        # Calculate the new variable
                        data[new_var] = data.eval(formula)
                        st.dataframe(data)  # Use dataframe() to make it more pretty
                    except KeyError as e:
                        st.write(f'Invalid variable in formula: {str(e)}')
                    except Exception as e:
                        st.write(f'Error occurred: {str(e)}')
          
            
        ##############################################################
                st.write('---------------------------------------------')
                st.header('Outlier Detection & Removal')
                st.write('---------------------------------------------')
    
                # st.header("Outlier Detection & Removal")
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
                         
                st.header("Recoding variables-cat to numeric")
                         
                #########################################################################
            # Create multiselect menu for categorical columns
                # categorical_cols = st.multiselect("Select the categorical columns you want to encode", data.columns)
            
                # if categorical_cols:
            
                #     if st.button('One-Hot Encoding'):
                #         # Perform One-Hot Encoding:
                #         data = pd.get_dummies(data, columns=categorical_cols)
                #         st.write('Data after one-hot encoding:')
                #         st.write(data)
            
                #     if st.button('Label Encoding'):
                #         # Create a label encoder object for Label Encoding
                #         le = LabelEncoder()
            
                #         for col in categorical_cols:
                #             # Fit and transform the selected columns
                #             data[col] = le.fit_transform(data[col])
                        
                #         st.write('Data after label encoding:')
                #         st.write(data)
                # else:
                #     st.write("No categorical column selected!")
                
    
                #########################################################################
                data.to_csv('preprocessed_data.csv', index=False)
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                st.markdown(href, unsafe_allow_html=True)
                st.session_state["preprocessed_data"] = data  # Store the preprocessed data in the session state
            else:
                st.warning('Please upload a file first.')    
            

 #           ###########################
            
  ###############################################################################################################################################################################################################################################          
            
            
        if task == "Machine Learning Models":
             if st.session_state["preprocessed_data"] is not None:
                data = st.session_state["preprocessed_data"]
                # Load your preprocessed data from the CSV file
                try:
                    data = pd.read_csv('preprocessed_data.csv')
                except FileNotFoundError:
                    st.warning("Please run preprocessing before training the model!")
                    return
                       
                st.header("Machine Learning")
                st.write("Data Shape")
                st.write(data.shape)
                model_name = st.sidebar.selectbox("Select Model", ["Linear Regression", "KNN Regression", "Decision Tree Regression", "Random Forest Regression", "SVM Regressor",  "KNN Classifier", "Decision Tree Classifier", "Random Forest Classifier", "Logistic Regression Classifier","SVM Classifier"])
                target = st.selectbox("Select the target variable", data.columns)
                selected_columns1 = st.multiselect("Select independent variables",data.columns)
                if not selected_columns1:
                    st.info("Please select at least one variable.")
                    return
    
                x = data[selected_columns1] #data.drop(target, axis=1)
                y = data[target]
                split_ratio = st.number_input('Enter a test/train split ratio', min_value=0.1, max_value=0.9, value=0.2, step=0.1)
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=42)
    
                if model_name == "Linear Regression":
                    # Reserve a slot for the progress bar
                    placeholder = st.empty()

                    # Insert your custom progress bar
                    progress_bar_html = """
                    <div style="width: 100%; background-color: #ddd;">
                    <div id="myBar" style="width: 0%; height: 30px; background: repeating-linear-gradient(
                            45deg,
                            red,
                            red 10px,
                            black 10px,
                            black 20px
                        ); text-align: center;">0%</div>
                    </div>
                    """

                    # CSS code for animation
                    progress_bar_css = """
                    <style>
                    #myBar {
                        animation: lightning 1.5s infinite linear;
                    }
                    @keyframes lightning {
                        0% { background-position: 0 0; }
                        100% { background-position: 50px 50px; }
                    }
                    </style>
                    """

                    # Note the use of placeholder to write to the reserved slot
                    placeholder.markdown(progress_bar_css, unsafe_allow_html=True)
                    placeholder.markdown(progress_bar_html, unsafe_allow_html=True)

                    # Initialize the progress bar
                    progress = 0

                    # Update the progress bar
                    def update_progress(new_val):
                        update_script = f"""
                        <script>
                            var elem = document.getElementById("myBar");
                            elem.style.width = "{new_val}%";
                            elem.innerHTML = "{new_val}%";
                        </script>
                        """
                        placeholder.markdown(update_script, unsafe_allow_html=True)

                    # Update progress to 10%
                    update_progress(10)

                    # Fit the model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Update progress to 50%
                    update_progress(50)

                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Update progress to 70%
                    update_progress(70)

                    # Evaluate the model
                    evaluate_regression(y_test, y_pred)
                    
                    # Update progress to 90%
                    update_progress(90)
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
                    update_progress(100)
                    #########################################################################
                elif model_name == "SVM Regressor":
                    
                    model = SVR(kernel='rbf')
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
                    
                elif model_name == "SVM Classifier":
                    from sklearn.svm import SVC
                    model = SVC()
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
             else:
                    st.warning("Please run preprocessing before training the model!")
########################################################################################################################################################
# About Me
#######################################################################################################################################################        


        # Function to convert image to base64
        def image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()

        #task = 'About Me'  # Replace with your task logic

        if task == 'About Me':
            st.markdown(
                "<h2 style='font-family:Times New Roman, Times, serif; font-style:italic;'>Data Scientist behind the app</h2>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='font-family:Times New Roman, Times, serif; font-style:italic;'>"
                "Hello everyone, this app is created and managed by Sajid Hafeez, Data scientist at Rprogrammers.com.<br>"
                "I offer services related to Data science and statistical analysis using R, Python, Stata, SPSS, Weka and Power BI. Feel free to contact me on the following."
                "</div>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            email_logo_base64 = image_to_base64("email.png")
            linkedin_logo_base64 = image_to_base64("whats.png")
            website_logo_base64 = image_to_base64("web.png")

            col1.markdown(
                f"""
                <div style='font-family:Times New Roman, Times, serif; font-style:italic;'>
                    <p><img src='data:image/png;base64,{email_logo_base64}' style='width:20px; vertical-align:middle;'/> Email: <a href='mailto:Sajidhafeex@gmail.com'>Sajidhafeex@gmail.com</a></p>
                    <p><img src='data:image/png;base64,{linkedin_logo_base64}' style='width:20px; vertical-align:middle;'/> LinkedIn: <a href='https://www.linkedin.com/in/sajid-hafeex'>https://www.linkedin.com/in/sajid-hafeex</a></p>
                    <p><img src='data:image/png;base64,{website_logo_base64}' style='width:20px; vertical-align:middle;'/> Website: <a href='https://Rprogrammers.com'>https://Rprogrammers.com</a></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col2.image("giphy.gif")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
