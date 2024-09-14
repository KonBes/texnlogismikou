import streamlit as st
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

# Function to check class alignment
def check_class_alignment(y_true, y_pred_prob):
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    if y_pred_prob.shape[1] != n_classes:
        return y_pred_prob[:, :n_classes]
    return y_pred_prob

# Set up the Streamlit app
st.set_page_config(page_title="Data Mining and Analysis", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS for better aesthetics
st.markdown("""
    <style>
    .reportview-container {
        background: #1e1e1e;
        color: #e0e0e0;
    }
    .sidebar .sidebar-content {
        background: #2c2c2c;
    }
    .st-bb {
        background: #3b3b3b;
        color: #e0e0e0;
    }
    .st-at {
        color: #e0e0e0;
    }
    .st-ae {
        color: #e0e0e0;
    }
    .st-bk {
        background: #3b3b3b;
        color: #e0e0e0;
    }
    .st-cv {
        background: #2c2c2c;
        border: 1px solid #3b3b3b;
    }
    .st-cd {
        color: #e0e0e0;
    }
    .st-dd {
        background: #2c2c2c;
        color: #e0e0e0;
    }
    .st-bb, .st-cg {
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("Web-Based Data Mining and Analysis App")

# Upload data
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, TSV)", type=["csv", "xlsx", "tsv"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, delimiter='\t')

    # Check if the dataset is valid
    if df.shape[1] < 2:
        st.error("The dataset must contain at least one feature and one label.")
    else:
        st.success("Dataset loaded successfully!")
        st.write(df)

        # Encode categorical labels
        label_encoder = LabelEncoder()
        df[df.columns[-1]] = label_encoder.fit_transform(df[df.columns[-1]])
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        # Create tabs
        tabs = st.tabs(["Visualization", "Feature Selection", "Classification", "Info"])

        with tabs[0]:
            # Visualization Tab
            st.subheader("Dimensionality Reduction Visualization")

            # Create columns for 2D and 3D plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PCA Visualization")

                # PCA
                pca = PCA(n_components=3)
                features_scaled = StandardScaler().fit_transform(features)
                pca_result = pca.fit_transform(features_scaled)

                # 2D PCA
                fig, ax = plt.subplots()
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab20', alpha=0.7)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('PCA 2D')
                cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(np.unique(labels))))
                cbar.set_label('Label')
                st.pyplot(fig)

                # 3D PCA
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=labels, cmap='tab20', alpha=0.7)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_zlabel('Principal Component 3')
                ax.set_title('PCA 3D')
                cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(np.unique(labels))))
                cbar.set_label('Label')
                st.pyplot(fig)

            with col2:
                st.subheader("UMAP Visualization")

                # UMAP
                umap_model = umap.UMAP(n_neighbors=15, n_components=3, random_state=42)
                umap_result = umap_model.fit_transform(features_scaled)

                # 2D UMAP
                fig, ax = plt.subplots()
                scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='tab20', alpha=0.7)
                ax.set_xlabel('UMAP Component 1')
                ax.set_ylabel('UMAP Component 2')
                ax.set_title('UMAP 2D')
                cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(np.unique(labels))))
                cbar.set_label('Label')
                st.pyplot(fig)

                # 3D UMAP
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=labels, cmap='tab20', alpha=0.7)
                ax.set_xlabel('UMAP Component 1')
                ax.set_ylabel('UMAP Component 2')
                ax.set_zlabel('UMAP Component 3')
                ax.set_title('UMAP 3D')
                cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(np.unique(labels))))
                cbar.set_label('Label')
                st.pyplot(fig)

            st.subheader("Exploratory Data Analysis (EDA)")
            st.write("Distribution of Features")
            fig, ax = plt.subplots()
            df.iloc[:, :-1].hist(ax=ax, bins=30, figsize=(10, 8))
            st.pyplot(fig)
            st.write("Pairplot of Features")
            try:
                fig = sns.pairplot(df, hue=df.columns[-1], palette='tab20')
                st.pyplot(fig)
            except ValueError as e:
                st.error(f"Pairplot error: {e}")

        with tabs[1]:
            # Feature Selection Tab
            st.subheader("Feature Selection")
            k_features = st.slider("Select number of features to retain:", min_value=1, max_value=features.shape[1], value=2)
            selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
            X_new = selector.fit_transform(features, labels)

            # Display the reduced dataset
            reduced_df = pd.DataFrame(X_new, columns=[f'Feature{i+1}' for i in range(k_features)])
            reduced_df[df.columns[-1]] = labels
            st.write("Dataset after Feature Selection:")
            st.write(reduced_df)

            # Prepare data for classification
            X_train, X_test, y_train, y_test = train_test_split(X_new, labels, test_size=0.2, random_state=42)

        with tabs[2]:
            # Classification Tab
            st.subheader("Classification")

            # Parameters
            k_value = st.slider("Select k for K-Nearest Neighbors:", min_value=1, max_value=20, value=5)
            c_value = st.slider("Select C for Logistic Regression:", min_value=0.01, max_value=10.0, value=1.0)

            # Initialize metrics dictionary
            metrics = {
                "K-Nearest Neighbors (Original Data)": {},
                "Logistic Regression (Original Data)": {},
                "K-Nearest Neighbors (Feature-Selected Data)": {},
                "Logistic Regression (Feature-Selected Data)": {}
            }

            def safe_roc_auc_score(y_true, y_prob):
                try:
                    y_true_bin = LabelBinarizer().fit_transform(y_true)
                    y_prob_bin = check_class_alignment(y_true_bin, y_prob)
                    return roc_auc_score(y_true_bin, y_prob_bin, multi_class='ovr')
                except ValueError:
                    return None

            # K-Nearest Neighbors on Original Data
            st.subheader("K-Nearest Neighbors on Original Data")
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(features_scaled, labels)
            y_pred_knn_orig = knn.predict(features_scaled)
            y_prob_knn_orig = knn.predict_proba(features_scaled)

            metrics["K-Nearest Neighbors (Original Data)"]["Accuracy"] = accuracy_score(labels, y_pred_knn_orig)
            metrics["K-Nearest Neighbors (Original Data)"]["F1 Score"] = f1_score(labels, y_pred_knn_orig, average='weighted')
            metrics["K-Nearest Neighbors (Original Data)"]["ROC AUC Score"] = safe_roc_auc_score(labels, y_prob_knn_orig)

            st.write("K-Nearest Neighbors Metrics on Original Data:")
            st.write(f"Accuracy: {metrics['K-Nearest Neighbors (Original Data)']['Accuracy']:.2f}")
            st.write(f"F1 Score: {metrics['K-Nearest Neighbors (Original Data)']['F1 Score']:.2f}")
            roc_auc_knn_orig = metrics["K-Nearest Neighbors (Original Data)"]["ROC AUC Score"]
            st.write(f"ROC AUC Score: {roc_auc_knn_orig:.2f}" if roc_auc_knn_orig is not None else "ROC AUC Score: N/A")

            # Logistic Regression on Original Data
            st.subheader("Logistic Regression on Original Data")
            log_reg = LogisticRegression(max_iter=1000, C=c_value, multi_class='ovr')
            log_reg.fit(features_scaled, labels)
            y_pred_log_reg_orig = log_reg.predict(features_scaled)
            y_prob_log_reg_orig = log_reg.predict_proba(features_scaled)

            metrics["Logistic Regression (Original Data)"]["Accuracy"] = accuracy_score(labels, y_pred_log_reg_orig)
            metrics["Logistic Regression (Original Data)"]["F1 Score"] = f1_score(labels, y_pred_log_reg_orig, average='weighted')
            metrics["Logistic Regression (Original Data)"]["ROC AUC Score"] = safe_roc_auc_score(labels, y_prob_log_reg_orig)

            st.write("Logistic Regression Metrics on Original Data:")
            st.write(f"Accuracy: {metrics['Logistic Regression (Original Data)']['Accuracy']:.2f}")
            st.write(f"F1 Score: {metrics['Logistic Regression (Original Data)']['F1 Score']:.2f}")
            roc_auc_log_reg_orig = metrics["Logistic Regression (Original Data)"]["ROC AUC Score"]
            st.write(f"ROC AUC Score: {roc_auc_log_reg_orig:.2f}" if roc_auc_log_reg_orig is not None else "ROC AUC Score: N/A")

            # K-Nearest Neighbors on Feature-Selected Data
            st.subheader("K-Nearest Neighbors on Feature-Selected Data")
            knn.fit(X_train, y_train)
            y_pred_knn_feat = knn.predict(X_test)
            y_prob_knn_feat = knn.predict_proba(X_test)

            metrics["K-Nearest Neighbors (Feature-Selected Data)"]["Accuracy"] = accuracy_score(y_test, y_pred_knn_feat)
            metrics["K-Nearest Neighbors (Feature-Selected Data)"]["F1 Score"] = f1_score(y_test, y_pred_knn_feat, average='weighted')
            metrics["K-Nearest Neighbors (Feature-Selected Data)"]["ROC AUC Score"] = safe_roc_auc_score(y_test, y_prob_knn_feat)

            st.write("K-Nearest Neighbors Metrics on Feature-Selected Data:")
            st.write(f"Accuracy: {metrics['K-Nearest Neighbors (Feature-Selected Data)']['Accuracy']:.2f}")
            st.write(f"F1 Score: {metrics['K-Nearest Neighbors (Feature-Selected Data)']['F1 Score']:.2f}")
            roc_auc_knn_feat = metrics["K-Nearest Neighbors (Feature-Selected Data)"]["ROC AUC Score"]
            st.write(f"ROC AUC Score: {roc_auc_knn_feat:.2f}" if roc_auc_knn_feat is not None else "ROC AUC Score: N/A")

            # Logistic Regression on Feature-Selected Data
            st.subheader("Logistic Regression on Feature-Selected Data")
            log_reg.fit(X_train, y_train)
            y_pred_log_reg_feat = log_reg.predict(X_test)
            y_prob_log_reg_feat = log_reg.predict_proba(X_test)

            metrics["Logistic Regression (Feature-Selected Data)"]["Accuracy"] = accuracy_score(y_test, y_pred_log_reg_feat)
            metrics["Logistic Regression (Feature-Selected Data)"]["F1 Score"] = f1_score(y_test, y_pred_log_reg_feat, average='weighted')
            metrics["Logistic Regression (Feature-Selected Data)"]["ROC AUC Score"] = safe_roc_auc_score(y_test, y_prob_log_reg_feat)

            st.write("Logistic Regression Metrics on Feature-Selected Data:")
            st.write(f"Accuracy: {metrics['Logistic Regression (Feature-Selected Data)']['Accuracy']:.2f}")
            st.write(f"F1 Score: {metrics['Logistic Regression (Feature-Selected Data)']['F1 Score']:.2f}")
            roc_auc_log_reg_feat = metrics["Logistic Regression (Feature-Selected Data)"]["ROC AUC Score"]
            st.write(f"ROC AUC Score: {roc_auc_log_reg_feat:.2f}" if roc_auc_log_reg_feat is not None else "ROC AUC Score: N/A")

            # Determine the best performing algorithm
            st.subheader("Algorithm Comparison")
            comparison_df = pd.DataFrame(metrics).T
            st.write("Performance Comparison of Different Algorithms:")
            st.write(comparison_df)

            # Calculate the average scores for each algorithm
            avg_metrics = comparison_df.mean(axis=1)
            best_algorithm = avg_metrics.idxmax()
            st.write(f"The best performing algorithm based on average metrics is: **{best_algorithm}**")

        with tabs[3]:
            # Info Tab
            st.subheader("Information")
            st.write("""
            **Application Information:**
            - This application provides tools for data visualization and machine learning.
            - Users can upload a dataset, perform dimensionality reduction using PCA and UMAP, and visualize data in 2D and 3D.
            - The application supports feature selection and classification using K-Nearest Neighbors and Logistic Regression.
            - ROC AUC scores and other performance metrics are provided for evaluating classification models.

            **Development Team:**
            - Developed by Konstantinos Besios P2018048 for Ionian University.
            """)
