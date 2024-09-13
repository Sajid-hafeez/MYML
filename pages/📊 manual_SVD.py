# # import numpy as np
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import streamlit as st
# # import plotly.express as px

# # # Streamlit App Layout
# # st.title("Custom Matrix SVD Analysis App")

# # # Step 1: Select number of rows and columns
# # st.write("### Step 1: Select the number of rows and columns")
# # num_rows = st.number_input('Select number of rows', min_value=2, max_value=10, value=3)
# # num_cols = st.number_input('Select number of columns', min_value=2, max_value=10, value=3)

# # # Step 2: Manually input the values for the matrix
# # st.write(f"### Step 2: Input values for a {int(num_rows)} x {int(num_cols)} matrix")
# # matrix = np.zeros((int(num_rows), int(num_cols)))

# # for i in range(int(num_rows)):
# #     for j in range(int(num_cols)):
# #         matrix[i, j] = st.number_input(f'Enter value for row {i+1}, column {j+1}', value=0.0)

# # # Step 3: Perform SVD analysis
# # st.write("### Step 3: SVD Analysis")
# # st.write("Original Matrix (A):")
# # st.write(matrix)

# # # Perform SVD with full matrices
# # U, Sigma, VT = np.linalg.svd(matrix, full_matrices=True)

# # # Convert Sigma to a diagonal matrix (for heatmap)
# # Sigma_matrix = np.zeros((U.shape[1], VT.shape[0]))  # Full matrix size
# # np.fill_diagonal(Sigma_matrix, Sigma)

# # # Calculate variance explained
# # variance_explained = (Sigma**2) / np.sum(Sigma**2)

# # # Prepare UΣ for scatter plot (if matrix is 2D or higher)
# # U_Sigma = U @ np.diag(Sigma)
# # if num_cols >= 2:
# #     df_u_sigma = pd.DataFrame(U_Sigma[:, :2], columns=['Component 1', 'Component 2'])

# # # Step 4: Handle matrix reconstruction
# # # Only use the first `min(num_rows, num_cols)` singular values to reconstruct the original matrix
# # A_reconstructed = (U[:, :len(Sigma)] @ np.diag(Sigma) @ VT[:len(Sigma), :])

# # # Step 5: Plot heatmaps and results
# # st.write("### Step 5: Visualizations and Analysis")

# # # Heatmap for original matrix (A)
# # st.write("#### Heatmap for Original Matrix (A)")
# # fig, ax = plt.subplots()
# # sns.heatmap(matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# # st.pyplot(fig)

# # # Heatmap for U matrix
# # st.write("#### Heatmap for U Matrix")
# # fig, ax = plt.subplots()
# # sns.heatmap(U, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# # st.pyplot(fig)

# # # Heatmap for Sigma matrix
# # st.write("#### Heatmap for Sigma Matrix")
# # fig, ax = plt.subplots()
# # sns.heatmap(Sigma_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# # st.pyplot(fig)

# # # Heatmap for V^T matrix (Full size)
# # st.write("#### Heatmap for V^T Matrix (Full Size)")
# # fig, ax = plt.subplots()
# # sns.heatmap(VT, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# # st.pyplot(fig)

# # # Variance Explained Plot
# # st.write("#### Variance Explained by Singular Values")
# # fig, ax = plt.subplots()
# # ax.bar(range(1, len(Sigma) + 1), variance_explained * 100)
# # ax.set_xlabel('Singular Value Index')
# # ax.set_ylabel('Variance Explained (%)')
# # st.pyplot(fig)

# # # Scatter plots to visualize the transformation at each step (only if columns >= 2)
# # if num_cols >= 2:
# #     st.write("#### Scatter Plot: Transformed Data (UΣ, First 2 Components)")
# #     fig = px.scatter(df_u_sigma, x='Component 1', y='Component 2', title='Transformed Data (U Σ)')
# #     st.plotly_chart(fig)

# #     st.write("#### Scatter Plot: Reconstructed Data (U Σ V^T, First 2 Columns)")
# #     df_reconstructed = pd.DataFrame(A_reconstructed[:, :2], columns=['Column 1', 'Column 2'])
# #     fig = px.scatter(df_reconstructed, x='Column 1', y='Column 2', title='Reconstructed Data (U Σ V^T)')
# #     st.plotly_chart(fig)
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import streamlit as st
# import plotly.express as px

# # Streamlit App Layout
# st.title("Custom Matrix SVD Analysis App")

# # Step 1: Select number of rows and columns
# st.write("### Step 1: Select the number of rows and columns")
# num_rows = st.number_input('Select number of rows', min_value=2, max_value=10, value=3)
# num_cols = st.number_input('Select number of columns', min_value=2, max_value=10, value=3)

# # Step 2: Display matrix with editable cells
# st.write(f"### Step 2: Input values for a {int(num_rows)} x {int(num_cols)} matrix")

# # Initialize a dataframe to store matrix values for input
# matrix_df = pd.DataFrame(np.zeros((int(num_rows), int(num_cols))), 
#                          columns=[f'Col {i+1}' for i in range(int(num_cols))], 
#                          index=[f'Row {i+1}' for i in range(int(num_rows))])

# # Use Streamlit's built-in data editor for matrix input (works as a grid)
# edited_matrix = st.experimental_data_editor(matrix_df)

# # Convert the dataframe back to a numpy array
# matrix = edited_matrix.values

# # Step 3: Perform SVD analysis
# st.write("### Step 3: SVD Analysis")
# st.write("Original Matrix (A):")
# st.write(matrix)

# # Perform SVD with full matrices
# U, Sigma, VT = np.linalg.svd(matrix, full_matrices=True)

# # Convert Sigma to a diagonal matrix (for heatmap)
# Sigma_matrix = np.zeros((U.shape[1], VT.shape[0]))  # Full matrix size
# np.fill_diagonal(Sigma_matrix, Sigma)

# # Calculate variance explained
# variance_explained = (Sigma**2) / np.sum(Sigma**2)

# # Prepare UΣ for scatter plot (if matrix is 2D or higher)
# U_Sigma = U @ np.diag(Sigma)
# if num_cols >= 2:
#     df_u_sigma = pd.DataFrame(U_Sigma[:, :2], columns=['Component 1', 'Component 2'])

# # Step 4: Handle matrix reconstruction
# # Only use the first `min(num_rows, num_cols)` singular values to reconstruct the original matrix
# A_reconstructed = (U[:, :len(Sigma)] @ np.diag(Sigma) @ VT[:len(Sigma), :])

# # Step 5: Plot heatmaps and results
# st.write("### Step 5: Visualizations and Analysis")

# # Heatmap for original matrix (A)
# st.write("#### Heatmap for Original Matrix (A)")
# fig, ax = plt.subplots()
# sns.heatmap(matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# st.pyplot(fig)

# # Heatmap for U matrix
# st.write("#### Heatmap for U Matrix")
# fig, ax = plt.subplots()
# sns.heatmap(U, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# st.pyplot(fig)

# # Heatmap for Sigma matrix
# st.write("#### Heatmap for Sigma Matrix")
# fig, ax = plt.subplots()
# sns.heatmap(Sigma_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# st.pyplot(fig)

# # Heatmap for V^T matrix (Full size)
# st.write("#### Heatmap for V^T Matrix (Full Size)")
# fig, ax = plt.subplots()
# sns.heatmap(VT, cmap='viridis', annot=True, fmt=".2f", ax=ax)
# st.pyplot(fig)

# # Variance Explained Plot
# st.write("#### Variance Explained by Singular Values")
# fig, ax = plt.subplots()
# ax.bar(range(1, len(Sigma) + 1), variance_explained * 100)
# ax.set_xlabel('Singular Value Index')
# ax.set_ylabel('Variance Explained (%)')
# st.pyplot(fig)

# # Scatter plots to visualize the transformation at each step (only if columns >= 2)
# if num_cols >= 2:
#     st.write("#### Scatter Plot: Transformed Data (UΣ, First 2 Components)")
#     fig = px.scatter(df_u_sigma, x='Component 1', y='Component 2', title='Transformed Data (U Σ)')
#     st.plotly_chart(fig)

#     st.write("#### Scatter Plot: Reconstructed Data (U Σ V^T, First 2 Columns)")
#     df_reconstructed = pd.DataFrame(A_reconstructed[:, :2], columns=['Column 1', 'Column 2'])
#     fig = px.scatter(df_reconstructed, x='Column 1', y='Column 2', title='Reconstructed Data (U Σ V^T)')
#     st.plotly_chart(fig)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Streamlit App Layout
st.title("Custom Matrix SVD Analysis App")

# Step 1: Select number of rows and columns
st.write("### Step 1: Select the number of rows and columns")
num_rows = st.number_input('Select number of rows', min_value=2, max_value=10, value=3)
num_cols = st.number_input('Select number of columns', min_value=2, max_value=10, value=3)

# Step 2: Display matrix with editable cells
st.write(f"### Step 2: Input values for a {int(num_rows)} x {int(num_cols)} matrix")

# Initialize a dataframe to store matrix values for input
matrix_df = pd.DataFrame(np.zeros((int(num_rows), int(num_cols))), 
                         columns=[f'Col {i+1}' for i in range(int(num_cols))], 
                         index=[f'Row {i+1}' for i in range(int(num_rows))])

# Use Streamlit's stable data editor for matrix input (works as a grid)
edited_matrix = st.data_editor(matrix_df)

# Convert the dataframe back to a numpy array
matrix = edited_matrix.values

# Step 3: Perform SVD analysis
st.write("### Step 3: SVD Analysis")
st.write("Original Matrix (A):")
st.write(matrix)

# Perform SVD with full matrices
U, Sigma, VT = np.linalg.svd(matrix, full_matrices=True)

# Convert Sigma to a diagonal matrix (for heatmap)
Sigma_matrix = np.zeros((U.shape[1], VT.shape[0]))  # Full matrix size
np.fill_diagonal(Sigma_matrix, Sigma)

# Calculate variance explained
variance_explained = (Sigma**2) / np.sum(Sigma**2)

# Prepare UΣ for scatter plot (if matrix is 2D or higher)
U_Sigma = U @ np.diag(Sigma)
if num_cols >= 2:
    df_u_sigma = pd.DataFrame(U_Sigma[:, :2], columns=['Component 1', 'Component 2'])

# Step 4: Handle matrix reconstruction
# Only use the first `min(num_rows, num_cols)` singular values to reconstruct the original matrix
A_reconstructed = (U[:, :len(Sigma)] @ np.diag(Sigma) @ VT[:len(Sigma), :])

# Step 5: Plot heatmaps and results
st.write("### Step 5: Visualizations and Analysis")

# Heatmap for original matrix (A)
st.write("#### Heatmap for Original Matrix (A)")
fig, ax = plt.subplots()
sns.heatmap(matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Heatmap for U matrix
st.write("#### Heatmap for U Matrix")
fig, ax = plt.subplots()
sns.heatmap(U, cmap='viridis', annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Heatmap for Sigma matrix
st.write("#### Heatmap for Sigma Matrix")
fig, ax = plt.subplots()
sns.heatmap(Sigma_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Heatmap for V^T matrix (Full size)
st.write("#### Heatmap for V^T Matrix (Full Size)")
fig, ax = plt.subplots()
sns.heatmap(VT, cmap='viridis', annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Variance Explained Plot
st.write("#### Variance Explained by Singular Values")
fig, ax = plt.subplots()
ax.bar(range(1, len(Sigma) + 1), variance_explained * 100)
ax.set_xlabel('Singular Value Index')
ax.set_ylabel('Variance Explained (%)')
st.pyplot(fig)

# Scatter plots to visualize the transformation at each step (only if columns >= 2)
if num_cols >= 2:
    st.write("#### Scatter Plot: Transformed Data (UΣ, First 2 Components)")
    fig = px.scatter(df_u_sigma, x='Component 1', y='Component 2', title='Transformed Data (U Σ)')
    st.plotly_chart(fig)

    st.write("#### Scatter Plot: Reconstructed Data (U Σ V^T, First 2 Columns)")
    df_reconstructed = pd.DataFrame(A_reconstructed[:, :2], columns=['Column 1', 'Column 2'])
    fig = px.scatter(df_reconstructed, x='Column 1', y='Column 2', title='Reconstructed Data (U Σ V^T)')
    st.plotly_chart(fig)
