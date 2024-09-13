
# # # ###########################################################
# # # #############################################################
# # # import numpy as np
# # # import dash
# # # from dash import dcc, html
# # # from dash.dependencies import Input, Output
# # # import plotly.express as px
# # # import plotly.graph_objects as go
# # # from sklearn.datasets import load_iris
# # # from sklearn.preprocessing import StandardScaler
# # # import pandas as pd

# # # # Load the dataset
# # # data = load_iris()
# # # A = data.data  # Original data matrix (A) with 4 features
# # # y = data.target  # Labels for visualization

# # # # Step 1: Standardize the data
# # # scaler = StandardScaler()
# # # A_standardized = scaler.fit_transform(A)

# # # # Step 2: Compute SVD
# # # U, Sigma, VT = np.linalg.svd(A_standardized, full_matrices=False)
# # # U_Sigma = U @ np.diag(Sigma)
# # # A_reconstructed = U @ np.diag(Sigma) @ VT

# # # # Variance explained by the first two singular values
# # # variance_explained = (Sigma**2) / np.sum(Sigma**2)
# # # variance_explained_cumsum = np.cumsum(variance_explained)

# # # # Initialize the Dash app
# # # app = dash.Dash(__name__)

# # # # Define the layout of the app
# # # app.layout = html.Div([
# # #     html.H1("Interactive SVD Data Transformation"),
# # #     html.P("Explore how data is reduced from 4 dimensions to 2 dimensions using SVD."),

# # #     # Dropdown to select transformation step
# # #     dcc.Dropdown(
# # #         id='transformation-select',
# # #         options=[
# # #             {'label': 'Original Data (4D)', 'value': '4D'},
# # #             {'label': 'Reduced Data (2D)', 'value': '2D'}
# # #         ],
# # #         value='4D',  # Default value
# # #         style={'width': '50%'}
# # #     ),

# # #     # Side-by-Side comparison graph
# # #     dcc.Graph(id='side-by-side-graph'),

# # #     # Parallel coordinates plot for 4D to 2D transition
# # #     html.H2("Parallel Coordinates: From 4D to 2D"),
# # #     dcc.Graph(id='parallel-coordinates-graph'),

# # #     # Pie chart for variance explained
# # #     html.H2("Variance Explained by Components"),
# # #     dcc.Graph(id='variance-pie-chart'),

# # #     # Explanation text
# # #     html.Div(id='explanation', style={'padding': '20px'}),
# # # ])

# # # # Define callback for the side-by-side SVD transformation graph
# # # @app.callback(
# # #     Output('side-by-side-graph', 'figure'),
# # #     Output('explanation', 'children'),
# # #     Input('transformation-select', 'value')
# # # )
# # # def update_graph(selected_transformation):
# # #     if selected_transformation == '4D':
# # #         # Project original 4D data onto the first two components for visualization
# # #         fig = px.scatter_matrix(
# # #             A_standardized, dimensions=range(4), color=y.astype(str),
# # #             labels={str(i): f'Feature {i + 1}' for i in range(4)},
# # #             title="Original Data (4D projected onto 2D)"
# # #         )
# # #         explanation = (
# # #             "This plot shows the original 4-dimensional data (Iris dataset) projected onto 2D. "
# # #             "The axes represent the original four features, and the color indicates the Iris species."
# # #         )
# # #     else:
# # #         # Show reduced 2D data (UΣ)
# # #         fig = px.scatter(
# # #             x=U_Sigma[:, 0], y=U_Sigma[:, 1], color=y.astype(str),
# # #             labels={'x': 'Component 1', 'y': 'Component 2'},
# # #             title="Reduced Data (2D after SVD)"
# # #         )
# # #         explanation = (
# # #             "This plot shows the data after dimensionality reduction to 2D using SVD. "
# # #             "The two new components retain most of the variance from the original data."
# # #         )

# # #     fig.update_layout(transition_duration=500)
# # #     return fig, explanation

# # # # Define callback for the parallel coordinates plot
# # # @app.callback(
# # #     Output('parallel-coordinates-graph', 'figure'),
# # #     Input('transformation-select', 'value')
# # # )
# # # def update_parallel_coordinates(selected_transformation):
# # #     if selected_transformation == '4D':
# # #         # Parallel coordinates for original 4D data
# # #         fig = px.parallel_coordinates(
# # #             data_frame=pd.DataFrame(A_standardized, columns=[f'Feature {i+1}' for i in range(4)]),
# # #             color=y,
# # #             labels={str(i): f'Feature {i+1}' for i in range(4)},
# # #             title="Parallel Coordinates: Original Data (4D)"
# # #         )
# # #     else:
# # #         # Parallel coordinates for reduced 2D data
# # #         df_2d = pd.DataFrame(U_Sigma[:, :2], columns=['Component 1', 'Component 2'])
# # #         df_2d['Species'] = y
# # #         fig = px.parallel_coordinates(
# # #             df_2d, color='Species', labels={'Component 1': 'Component 1', 'Component 2': 'Component 2'},
# # #             title="Parallel Coordinates: Reduced Data (2D)"
# # #         )

# # #     fig.update_layout(transition_duration=500)
# # #     return fig

# # # # Define callback for the variance explained pie chart
# # # @app.callback(
# # #     Output('variance-pie-chart', 'figure'),
# # #     Input('transformation-select', 'value')
# # # )
# # # def update_variance_pie_chart(selected_transformation):
# # #     labels = [f'Component {i + 1}' for i in range(len(Sigma))]
# # #     fig = go.Figure(data=[go.Pie(labels=labels, values=variance_explained, hole=.3)])
# # #     fig.update_layout(title="Variance Explained by Each Component")
# # #     return fig

# # # # Run the Dash app
# # # if __name__ == '__main__':
# # #     app.run_server(debug=True)
# # # import numpy as np
# # # import pandas as pd
# # # import streamlit as st
# # # import plotly.express as px
# # # import plotly.graph_objects as go
# # # from sklearn.datasets import load_iris
# # # from sklearn.preprocessing import StandardScaler

# # # # Load the dataset
# # # data = load_iris()
# # # df = pd.DataFrame(data['data'], columns=data['feature_names'])

# # # # Add a species column to the DataFrame (using the target)
# # # df['species'] = pd.Categorical.from_codes(data['target'], data['target_names'])

# # # # Now use st.write() or st.table() to display the table
# # # st.header("Original Data")
# # # st.write(df)
# # # A = data.data  # Original data matrix (A) with 4 features
# # # y = data.target  # Labels for visualization

# # # # Standardize the data
# # # scaler = StandardScaler()
# # # A_standardized = scaler.fit_transform(A)

# # # # Perform SVD
# # # U, Sigma, VT = np.linalg.svd(A_standardized, full_matrices=False)
# # # U_Sigma = U @ np.diag(Sigma)
# # # A_reconstructed = U @ np.diag(Sigma) @ VT

# # # # Variance explained by the first two singular values
# # # variance_explained = (Sigma**2) / np.sum(Sigma**2)
# # # variance_explained_cumsum = np.cumsum(variance_explained)

# # # # Streamlit App Layout
# # # st.title("Interactive SVD Data Transformation")
# # # st.write("Explore how data is reduced from 4 dimensions to 2 dimensions using SVD.")

# # # # Dropdown to select transformation
# # # transformation_select = st.selectbox(
# # #     "Select Data Transformation:",
# # #     ("Original Data (4D)", "Reduced Data (2D)")
# # # )
# # # column_names = data.feature_names
# # # # Side-by-Side Comparison
# # # if transformation_select == "Original Data (4D)":
# # #     # Original Data (4D projected onto 2D)
# # #     fig_4d = px.scatter_matrix(
# # #         A_standardized, dimensions=range(4), color=y.astype(str),
# # #         labels={str(i): column_names[i] for i in range(4)},
# # #         title="Original Data (4D projected onto 2D)"
# # #     )
# # #     st.plotly_chart(fig_4d)

# # #     st.write("This plot shows the original 4-dimensional data (Iris dataset) projected onto 2D. "
# # #              "The axes represent the original four features, and the color indicates the Iris species.")

# # # else:
# # #     # Reduced Data (UΣ in 2D)
# # #     fig_2d = px.scatter(
# # #         x=U_Sigma[:, 0], y=U_Sigma[:, 1], color=y.astype(str),
# # #         labels={'x': 'Component 1', 'y': 'Component 2'},
# # #         title="Reduced Data (2D after SVD)"
# # #     )
# # #     st.plotly_chart(fig_2d)

# # #     st.write("This plot shows the data after dimensionality reduction to 2D using SVD. "
# # #              "The two new components retain most of the variance from the original data.")

# # # # Parallel Coordinates Plot
# # # st.header("Parallel Coordinates: From 4D to 2D")
# # # if transformation_select == "Original Data (4D)":
# # #     df_4d = pd.DataFrame(A_standardized, columns=[f'Feature {i + 1}' for i in range(4)])
# # #     df_4d['Species'] = y
# # #     fig_parallel_4d = px.parallel_coordinates(
# # #         df_4d, color='Species',
# # #         labels={str(i): f'Feature {i + 1}' for i in range(4)},
# # #         title="Parallel Coordinates: Original Data (4D)"
# # #     )
# # #     st.plotly_chart(fig_parallel_4d)
# # # else:
# # #     df_2d = pd.DataFrame(U_Sigma[:, :2], columns=['Component 1', 'Component 2'])
# # #     df_2d['Species'] = y
# # #     fig_parallel_2d = px.parallel_coordinates(
# # #         df_2d, color='Species',
# # #         title="Parallel Coordinates: Reduced Data (2D)"
# # #     )
# # #     st.plotly_chart(fig_parallel_2d)

# # # # Variance Explained by Components
# # # st.header("Variance Explained by Components")
# # # fig_variance = go.Figure(data=[go.Pie(labels=[f'Component {i + 1}' for i in range(len(Sigma))],
# # #                                       values=variance_explained, hole=.3)])
# # # fig_variance.update_layout(title="Variance Explained by Each Component")
# # # st.plotly_chart(fig_variance)

# # # st.write("This pie chart shows the proportion of variance explained by each component in the SVD. "
# # #          "The first two components typically explain most of the variance in the data.")
# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # import plotly.express as px
# # import plotly.graph_objects as go
# # from sklearn.datasets import load_iris
# # from sklearn.preprocessing import StandardScaler
# # from io import StringIO

# # st.title("Interactive SVD Data Transformation")

# # # File uploader for any dataset
# # uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")

# # # Option to select the Iris dataset as an example
# # use_iris = st.checkbox("Use Iris dataset as an example")

# # if uploaded_file is not None:
# #     # Read the uploaded dataset
# #     df = pd.read_csv(uploaded_file)
# #     st.write("Uploaded Data:")
# #     st.write(df)
# # elif use_iris:
# #     # Load the Iris dataset as an example
# #     data = load_iris()
# #     df = pd.DataFrame(data['data'], columns=data['feature_names'])
# #     st.write("Iris Dataset:")
# #     st.write(df)
# # else:
# #     st.write("Please upload a dataset or select the Iris dataset to continue.")

# # if 'df' in locals():
# #     # Standardize the data (ignoring non-numeric columns)
# #     numeric_cols = df.select_dtypes(include=[np.number]).columns
# #     A = df[numeric_cols].values

# #     scaler = StandardScaler()
# #     A_standardized = scaler.fit_transform(A)

# #     # Perform SVD
# #     U, Sigma, VT = np.linalg.svd(A_standardized, full_matrices=False)
# #     U_Sigma = U @ np.diag(Sigma)
# #     A_reconstructed = U @ np.diag(Sigma) @ VT

# #     # Create a new dataframe for the transformed data
# #     df_transformed = pd.DataFrame(U_Sigma, columns=[f'Component {i+1}' for i in range(U_Sigma.shape[1])])

# #     # Variance explained by the first two singular values
# #     variance_explained = (Sigma**2) / np.sum(Sigma**2)
# #     variance_explained_cumsum = np.cumsum(variance_explained)

# #     # Dropdown to select transformation
# #     transformation_select = st.selectbox(
# #         "Select Data Transformation:",
# #         ("Original Data (4D)", "Reduced Data (2D)")
# #     )

# #     # Side-by-Side Comparison
# #     if transformation_select == "Original Data (4D)" and A.shape[1] >= 4:
# #         # Original Data (4D projected onto 2D)
# #         fig_4d = px.scatter_matrix(
# #             A_standardized, dimensions=range(min(4, A.shape[1])),
# #             labels={str(i): numeric_cols[i] for i in range(min(4, A.shape[1]))},
# #             title="Original Data (Projected onto 2D)"
# #         )
# #         st.plotly_chart(fig_4d)

# #         st.write("This plot shows the original data projected onto 2D.")
# #     else:
# #         # Reduced Data (UΣ in 2D)
# #         fig_2d = px.scatter(
# #             x=U_Sigma[:, 0], y=U_Sigma[:, 1],
# #             labels={'x': 'Component 1', 'y': 'Component 2'},
# #             title="Reduced Data (2D after SVD)"
# #         )
# #         st.plotly_chart(fig_2d)

# #         st.write("This plot shows the data after dimensionality reduction to 2D using SVD.")

# #     # Variance Explained by Components
# #     st.header("Variance Explained by Components")
# #     fig_variance = go.Figure(data=[go.Pie(labels=[f'Component {i + 1}' for i in range(len(Sigma))],
# #                                           values=variance_explained, hole=.3)])
# #     fig_variance.update_layout(title="Variance Explained by Each Component")
# #     st.plotly_chart(fig_variance)

# #     st.write("This pie chart shows the proportion of variance explained by each component in the SVD.")

# #     # Create a CSV to download the transformed data
# #     csv = df_transformed.to_csv(index=False)
# #     st.download_button(
# #         label="Download Transformed Data",
# #         data=csv,
# #         file_name='transformed_data.csv',
# #         mime='text/csv'
# #     )
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from io import StringIO

# st.title("Interactive SVD Data Transformation")

# # File uploader for any dataset
# uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")

# # Option to select the Iris dataset as an example
# use_iris = st.checkbox("Use Iris dataset as an example")

# if uploaded_file is not None:
#     # Read the uploaded dataset
#     df = pd.read_csv(uploaded_file)
#     st.write("Uploaded Data:")
#     st.write(df)
# elif use_iris:
#     # Load the Iris dataset as an example
#     data = load_iris()
#     df = pd.DataFrame(data['data'], columns=data['feature_names'])
#     st.write("Iris Dataset:")
#     st.write(df)
# else:
#     st.write("Please upload a dataset or select the Iris dataset to continue.")

# if 'df' in locals():
#     # Standardize the data (ignoring non-numeric columns)
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     A = df[numeric_cols].values

#     scaler = StandardScaler()
#     A_standardized = scaler.fit_transform(A)

#     # Perform SVD
#     U, Sigma, VT = np.linalg.svd(A_standardized, full_matrices=False)
#     U_Sigma = U @ np.diag(Sigma)
#     A_reconstructed = U @ np.diag(Sigma) @ VT

#     # Create a new dataframe for the transformed data
#     df_transformed = pd.DataFrame(U_Sigma, columns=[f'Component {i+1}' for i in range(U_Sigma.shape[1])])

#     # Variance explained by the first two singular values
#     variance_explained = (Sigma**2) / np.sum(Sigma**2)
#     variance_explained_cumsum = np.cumsum(variance_explained)

#     # Dropdown to select transformation
#     transformation_select = st.selectbox(
#         "Select Data Transformation:",
#         ("Original Data (Full Dimensions)", "Reduced Data (2D)")
#     )

#     # Plot according to the number of dimensions in the dataset
#     if transformation_select == "Original Data (Full Dimensions)" and A.shape[1] >= 2:
#         # Original Data (Dynamically handle the number of columns)
#         max_columns_to_display = min(A.shape[1], 10)  # Limit the plot to 10 columns to avoid overloading
#         fig_full_dim = px.scatter_matrix(
#             A_standardized, dimensions=range(max_columns_to_display),
#             labels={str(i): numeric_cols[i] for i in range(max_columns_to_display)},
#             title=f"Original Data ({max_columns_to_display}D projected)"
#         )
#         st.plotly_chart(fig_full_dim)

#         st.write(f"This plot shows the original data projected onto {max_columns_to_display}D.")
#     else:
#         # Reduced Data (UΣ in 2D)
#         fig_2d = px.scatter(
#             x=U_Sigma[:, 0], y=U_Sigma[:, 1],
#             labels={'x': 'Component 1', 'y': 'Component 2'},
#             title="Reduced Data (2D after SVD)"
#         )
#         st.plotly_chart(fig_2d)

#         st.write("This plot shows the data after dimensionality reduction to 2D using SVD.")

#     # Variance Explained by Components
#     st.header("Variance Explained by Components")
#     fig_variance = go.Figure(data=[go.Pie(labels=[f'Component {i + 1}' for i in range(len(Sigma))],
#                                           values=variance_explained, hole=.3)])
#     fig_variance.update_layout(title="Variance Explained by Each Component")
#     st.plotly_chart(fig_variance)

#     st.write("This pie chart shows the proportion of variance explained by each component in the SVD.")

#     # Create a CSV to download the transformed data
#     csv = df_transformed.to_csv(index=False)
#     st.download_button(
#         label="Download Transformed Data",
#         data=csv,
#         file_name='transformed_data.csv',
#         mime='text/csv'
#     )
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from io import StringIO

st.title("Interactive SVD Data Transformation with Heatmaps")

# File uploader for any dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")

# Option to select the Iris dataset as an example
use_iris = st.checkbox("Use Iris dataset as an example")

if uploaded_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df)
elif use_iris:
    # Load the Iris dataset as an example
    data = load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    st.write("Iris Dataset:")
    st.write(df)
else:
    st.write("Please upload a dataset or select the Iris dataset to continue.")

if 'df' in locals():
    # Standardize the data (ignoring non-numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    A = df[numeric_cols].values

    scaler = StandardScaler()
    A_standardized = scaler.fit_transform(A)

    # Perform SVD
    U, Sigma, VT = np.linalg.svd(A_standardized, full_matrices=False)
    U_Sigma = U @ np.diag(Sigma)
    A_reconstructed = U @ np.diag(Sigma) @ VT

    # Create a new dataframe for the transformed data
    df_transformed = pd.DataFrame(U_Sigma, columns=[f'Component {i+1}' for i in range(U_Sigma.shape[1])])

    # Variance explained by the first two singular values
    variance_explained = (Sigma**2) / np.sum(Sigma**2)
    variance_explained_cumsum = np.cumsum(variance_explained)

    # Dropdown to select transformation
    transformation_select = st.selectbox(
        "Select Data Transformation:",
        ("Original Data (Full Dimensions)", "Reduced Data (2D)")
    )

    # Plot according to the number of dimensions in the dataset
    if transformation_select == "Original Data (Full Dimensions)" and A.shape[1] >= 2:
        # Original Data (Dynamically handle the number of columns)
        max_columns_to_display = min(A.shape[1], 10)  # Limit the plot to 10 columns to avoid overloading
        fig_full_dim = px.scatter_matrix(
            A_standardized, dimensions=range(max_columns_to_display),
            labels={str(i): numeric_cols[i] for i in range(max_columns_to_display)},
            title=f"Original Data ({max_columns_to_display}D projected)"
        )
        st.plotly_chart(fig_full_dim)

        st.write(f"This plot shows the original data projected onto {max_columns_to_display}D.")
    else:
        # Reduced Data (UΣ in 2D)
        fig_2d = px.scatter(
            x=U_Sigma[:, 0], y=U_Sigma[:, 1],
            labels={'x': 'Component 1', 'y': 'Component 2'},
            title="Reduced Data (2D after SVD)"
        )
        st.plotly_chart(fig_2d)

        st.write("This plot shows the data after dimensionality reduction to 2D using SVD.")

    # Variance Explained by Components
    st.header("Variance Explained by Components")
    fig_variance = go.Figure(data=[go.Pie(labels=[f'Component {i + 1}' for i in range(len(Sigma))],
                                          values=variance_explained, hole=.3)])
    fig_variance.update_layout(title="Variance Explained by Each Component")
    st.plotly_chart(fig_variance)

    st.write("This pie chart shows the proportion of variance explained by each component in the SVD.")

    # Heatmap for Original Data
    st.header("Heatmap of Original Data")
    fig, ax = plt.subplots()
    sns.heatmap(A_standardized, ax=ax, cmap="coolwarm", cbar=True)
    st.pyplot(fig)

    # Heatmap for Matrix U
    st.header("Heatmap of Matrix U")
    fig, ax = plt.subplots()
    sns.heatmap(U, ax=ax, cmap="coolwarm", cbar=True)
    st.pyplot(fig)

    # Heatmap for Matrix Σ (Sigma)
    st.header("Heatmap of Matrix Σ (Diagonal Matrix)")
    Sigma_matrix = np.diag(Sigma)
    fig, ax = plt.subplots()
    sns.heatmap(Sigma_matrix, annot=True, fmt=".2f", ax=ax, cmap="coolwarm", cbar=True)
    st.pyplot(fig)

    # Heatmap for Matrix V^T
    st.header("Heatmap of Matrix V^T")
    fig, ax = plt.subplots()
    sns.heatmap(VT, annot=True, fmt=".2f", ax=ax, cmap="coolwarm", cbar=True)
    st.pyplot(fig)

    # Create a CSV to download the transformed data
    csv = df_transformed.to_csv(index=False)
    st.download_button(
        label="Download Transformed Data",
        data=csv,
        file_name='transformed_data.csv',
        mime='text/csv'
    )
