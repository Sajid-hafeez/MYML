
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Title
st.title("PCA Step-by-Step App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.dataframe(df)

    # Filter out non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    st.write("Data with only numeric columns:")
    st.dataframe(numeric_df)

    # Check for missing values (NaN) or infinite values (inf)
    if numeric_df.isnull().values.any():
        st.warning("Dataset contains NaN values. These will be filled with the mean of each column.")
        numeric_df = numeric_df.fillna(numeric_df.mean())

    if np.isinf(numeric_df.values).any():
        st.warning("Dataset contains infinite values. These will be replaced with finite maximum/minimum values.")
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.fillna(numeric_df.mean())

    if numeric_df.empty:
        st.warning("No numeric columns found in the dataset!")
    else:
        # Step 1: Standardize the data
        st.subheader("Step 1: Standardize the Data")
        if st.checkbox("Show Standardized Data"):
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(numeric_df)
            st.write(pd.DataFrame(df_scaled, columns=numeric_df.columns))

        # Step 2: Calculate the covariance matrix
        st.subheader("Step 2: Covariance Matrix")
        if st.checkbox("Show Covariance Matrix"):
            cov_matrix = np.cov(df_scaled.T)
            st.write(pd.DataFrame(cov_matrix, index=numeric_df.columns, columns=numeric_df.columns))

            # Check for NaN or Inf in the covariance matrix before proceeding
            if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                st.error("Covariance matrix contains NaN or infinite values. PCA cannot be performed.")
            else:
                # Step 3: Eigenvalues and Eigenvectors
                st.subheader("Step 3: Eigenvalues and Eigenvectors")
                eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
                st.write("Eigenvalues:")
                st.write(eig_vals)
                st.write("Eigenvectors:")
                st.write(pd.DataFrame(eig_vecs, index=numeric_df.columns))

                # Step 4: Explained Variance
                st.subheader("Step 4: Explained Variance")
                pca = PCA()
                pca.fit(df_scaled)
                explained_variance = pca.explained_variance_ratio_
                st.write("Explained Variance:")
                st.write(explained_variance)

                # Plot the explained variance
                st.subheader("Explained Variance Plot")
                plt.figure(figsize=(8, 4))
                plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Explained Variance')
                plt.ylabel('Explained variance ratio')
                plt.xlabel('Principal components')
                plt.title('Explained Variance by Principal Components')
                st.pyplot(plt)

                # Step 5: Visualize the Principal Components
                st.subheader("Step 5: Visualizing the Principal Components")
                if st.checkbox("Show PCA Transformation"):
                    pca_data = pca.transform(df_scaled)
                    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i}' for i in range(1, pca_data.shape[1] + 1)])
                    st.write("PCA Transformed Data:")
                    st.dataframe(pca_df)

                    # Scatter plot for the first two components
                    st.subheader("Scatter Plot of First Two Principal Components")
                    plt.figure(figsize=(8, 6))
                    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                    plt.title('PCA: First Two Principal Components')
                    st.pyplot(plt)

        # Step 6: Mathematical Explanation
        st.subheader("Mathematics Behind PCA")
        st.markdown("""
            **Step 1**: Standardize the data to have a mean of 0 and a variance of 1.
            \n**Step 2**: Calculate the covariance matrix to understand how features vary together.
            \n**Step 3**: Compute the eigenvalues and eigenvectors from the covariance matrix to identify principal components.
            \n**Step 4**: Project the data onto the principal components to reduce dimensionality.
        """)

else:
    st.info("Please upload a CSV file to proceed.")
