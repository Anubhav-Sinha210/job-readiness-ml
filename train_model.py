import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.cluster import KMeans
# Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, silhouette_score

def handle_outliers(df, cols):
    # Cap outliers using IQR
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

def save_plots(fig, filename):
    os.makedirs('outputs', exist_ok=True)
    fig.savefig(f'outputs/{filename}')
    plt.close(fig)

def main():
    print("Loading data...")
    df = pd.read_csv('data/student_data.csv')
    
    # --- 1. DATA PREPROCESSING & 2. FEATURE ENGINEERING ---
    # Handle Outliers before splits (for simplified flow)
    num_cols = ['CGPA', 'Technical_Skills', 'Aptitude_Score', 'Communication_Skills', 'Coding_Score']
    df = handle_outliers(df, num_cols)
    
    X = df.drop('Job_Ready', axis=1)
    y = df['Job_Ready']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = ['Degree_Stream']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Preprocessing Data (Target Imbalance: SMOTE applies after split)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit preprocessor
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Fetch feature names post one-hot encoding
    cat_enc = preprocessor.named_transformers_['cat']['onehot']
    cat_names = cat_enc.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_names)
    
    X_train_df = pd.DataFrame(X_train_prep, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_prep, columns=feature_names)
    
    # Handle Imbalance
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_df, y_train)
    
    # Variance Threshold for feature selection
    selector = VarianceThreshold(threshold=0.01)
    X_train_sm = selector.fit_transform(X_train_sm)
    X_test_sel = selector.transform(X_test_df)
    
    feature_names = np.array(feature_names)[selector.get_support()]
    
    X_train_sm = pd.DataFrame(X_train_sm, columns=feature_names)
    X_test_sel = pd.DataFrame(X_test_sel, columns=feature_names)
    
    print(f"Features after selection: {feature_names.tolist()}")
    
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    plt_sns.heatmap(X_train_sm.corr(), cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    save_plots(plt.gcf(), 'correlation_matrix.png')
    
    # Feature Importance (Tree-based)
    rf_fi = RandomForestClassifier(random_state=42)
    rf_fi.fit(X_train_sm, y_train_sm)
    importances = rf_fi.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    save_plots(plt.gcf(), 'feature_importance.png')
    
    # --- 3. DIMENSIONALITY REDUCTION (PCA demo) ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_sm)
    print(f"PCA Explained Variance Ratio (2 components): {pca.explained_variance_ratio_}")
    
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_sm, cmap='viridis', alpha=0.5)
    plt.title('PCA 2D Projection')
    save_plots(plt.gcf(), 'pca_projection.png')
    
    # --- 6. REGRESSION ---
    print("\nRunning Linear Regression demo on arbitrary readiness scale...")
    lin_reg = LinearRegression()
    y_reg_train = X_train_sm['CGPA'] * 10 + X_train_sm['Technical_Skills'] * 1.5 + X_train_sm['Coding_Score'] * 1.5
    lin_reg.fit(X_train_sm, y_reg_train)
    print(f"Linear Regression R2 Score: {lin_reg.score(X_train_sm, y_reg_train):.2f}")
    
    # --- 9. UNSUPERVISED LEARNING (K-Means) ---
    print("\nRunning K-Means Clustering...")
    errors, sil_scores = [], []
    clusters = range(2, 6)
    for k in clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_sm)
        errors.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_train_sm, kmeans.labels_))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(clusters, errors, marker='o')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(clusters, sil_scores, marker='o')
    plt.title('Silhouette Score')
    save_plots(plt.gcf(), 'kmeans_eval.png')
    
    # --- 4 & 5. MODELS & EVALUATION ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        save_plots(plt.gcf(), f'roc_{name.lower().replace(" ", "_")}.png')
        
    print("\nModel Evaluation:")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by='F1', ascending=False))
    
    # --- 7 & 8. ENSEMBLE & HYPERPARAMETER TUNING ---
    print("\nHyperparameter Tuning Random Forest...")
    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train_sm, y_train_sm)
    best_rf = grid_search.best_estimator_
    print(f"Best RF Params: {grid_search.best_params_}")
    
    print("\nVoting Classifier Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=2000))
        ],
        voting='soft'
    )
    voting_clf.fit(X_train_sm, y_train_sm)
    ensemble_acc = accuracy_score(y_test, voting_clf.predict(X_test_sel))
    print(f"Voting Classifier Accuracy: {ensemble_acc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('models/selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
        
    results_df.to_csv('outputs/model_metrics.csv', index=False)
    print("\nTraining completed and best model saved successfully.")

if __name__ == "__main__":
    main()
