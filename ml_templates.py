"""
General Machine Learning Templates and Pipelines Cheatsheet
Preprocessing, feature engineering, model selection, ensembles, and production pipelines
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    f_classif, f_regression, chi2, mutual_info_classif
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier, StackingClassifier,
    AdaBoostClassifier
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import joblib

# ============================================================================
# BASIC PREPROCESSING PIPELINE
# ============================================================================

def create_basic_preprocessing_pipeline():
    """Basic preprocessing pipeline for numerical data"""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    return pipeline

def create_full_preprocessing_pipeline(numerical_features, categorical_features):
    """Comprehensive preprocessing for mixed data types"""
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

# ============================================================================
# ADVANCED PREPROCESSING
# ============================================================================

def create_advanced_preprocessing_pipeline(numerical_features, categorical_features):
    """Advanced preprocessing with feature engineering"""
    
    # Numerical transformations
    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler())
    ])
    
    # Categorical transformations
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')
    
    return preprocessor

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_feature_engineering_pipeline():
    """Pipeline with feature engineering steps"""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=20))
    ])
    
    return pipeline

def add_custom_features(df):
    """Add custom engineered features"""
    df_new = df.copy()
    
    # Date features
    if 'date' in df.columns:
        df_new['year'] = pd.to_datetime(df['date']).dt.year
        df_new['month'] = pd.to_datetime(df['date']).dt.month
        df_new['day'] = pd.to_datetime(df['date']).dt.day
        df_new['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek
        df_new['quarter'] = pd.to_datetime(df['date']).dt.quarter
    
    # Interaction features (example)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        df_new[f'{numeric_cols[0]}_x_{numeric_cols[1]}'] = df[numeric_cols[0]] * df[numeric_cols[1]]
        df_new[f'{numeric_cols[0]}_div_{numeric_cols[1]}'] = df[numeric_cols[0]] / (df[numeric_cols[1]] + 1e-8)
    
    # Aggregation features
    if len(numeric_cols) >= 3:
        df_new['sum_features'] = df[numeric_cols].sum(axis=1)
        df_new['mean_features'] = df[numeric_cols].mean(axis=1)
        df_new['std_features'] = df[numeric_cols].std(axis=1)
    
    return df_new

# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_univariate(X, y, k=10, method='f_classif'):
    """Univariate feature selection"""
    if method == 'f_classif':
        selector = SelectKBest(f_classif, k=k)
    elif method == 'chi2':
        selector = SelectKBest(chi2, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_features = selector.get_support(indices=True)
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    return X_selected, selector

def select_features_model_based(X, y, model, threshold='mean'):
    """Model-based feature selection"""
    selector = SelectFromModel(model, threshold=threshold)
    X_selected = selector.fit_transform(X, y)
    
    selected_features = selector.get_support(indices=True)
    print(f"Selected {len(selected_features)} features using {model.__class__.__name__}")
    
    return X_selected, selector

def select_features_rfe(X, y, model, n_features_to_select=10):
    """Recursive Feature Elimination"""
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    X_selected = rfe.fit_transform(X, y)
    
    selected_features = rfe.get_support(indices=True)
    print(f"RFE selected features: {selected_features}")
    print(f"Feature ranking: {rfe.ranking_}")
    
    return X_selected, rfe

def select_features_rfecv(X, y, model, cv=5):
    """RFE with Cross-Validation"""
    rfecv = RFECV(model, cv=cv, scoring='accuracy')
    X_selected = rfecv.fit_transform(X, y)
    
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Selected features: {rfecv.get_support(indices=True)}")
    
    return X_selected, rfecv

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def apply_pca(X, n_components=0.95):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {X_reduced.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_reduced, pca

def apply_truncated_svd(X, n_components=50):
    """Apply TruncatedSVD (works with sparse matrices)"""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    
    return X_reduced, svd

# ============================================================================
# COMPLETE ML PIPELINE
# ============================================================================

def create_complete_ml_pipeline(model, numerical_features, categorical_features):
    """End-to-end ML pipeline"""
    
    # Preprocessing
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=20)),
        ('model', model)
    ])
    
    return pipeline

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def grid_search_tuning(model, param_grid, X, y, cv=5, scoring='accuracy'):
    """Grid search for hyperparameter tuning"""
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search

def randomized_search_tuning(model, param_distributions, X, y, 
                             n_iter=100, cv=5, scoring='accuracy'):
    """Randomized search for hyperparameter tuning"""
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search

# Example parameter grids
def get_rf_param_grid():
    """Parameter grid for Random Forest"""
    return {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

def get_gb_param_grid():
    """Parameter grid for Gradient Boosting"""
    return {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

def create_voting_classifier(models):
    """Voting classifier ensemble"""
    voting_clf = VotingClassifier(
        estimators=models,
        voting='soft'  # 'hard' for majority vote, 'soft' for probability average
    )
    
    return voting_clf

def create_stacking_classifier(base_models, meta_model, cv=5):
    """Stacking classifier ensemble"""
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv
    )
    
    return stacking_clf

def create_bagging_classifier(base_model, n_estimators=10):
    """Bagging classifier"""
    bagging_clf = BaggingClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=42
    )
    
    return bagging_clf

# Example ensemble setup
def create_ensemble_models():
    """Create various ensemble models"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    
    # Voting ensemble
    voting_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(random_state=42))
    ]
    voting_clf = create_voting_classifier(voting_models)
    
    # Stacking ensemble
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    meta_model = LogisticRegression()
    stacking_clf = create_stacking_classifier(base_models, meta_model)
    
    return {
        'voting': voting_clf,
        'stacking': stacking_clf
    }

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model, filename='model.pkl'):
    """Save trained model to disk"""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='model.pkl'):
    """Load trained model from disk"""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def save_pipeline(pipeline, filename='pipeline.pkl'):
    """Save complete pipeline"""
    joblib.dump(pipeline, filename)
    print(f"Pipeline saved to {filename}")

# ============================================================================
# AUTOMATED ML PIPELINE
# ============================================================================

class AutoMLPipeline:
    """Automated ML pipeline with preprocessing, feature selection, and model selection"""
    
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.best_pipeline = None
        self.best_score = 0
        
    def create_preprocessing(self):
        """Create preprocessing steps"""
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        return ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])
    
    def get_candidate_models(self):
        """Get candidate models to try"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        return {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier()
        }
    
    def fit(self, X, y, cv=5):
        """Fit and evaluate multiple models"""
        from sklearn.model_selection import cross_val_score
        
        preprocessor = self.create_preprocessing()
        models = self.get_candidate_models()
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            
            results[name] = {
                'mean_score': mean_score,
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"{name}: {mean_score:.4f} ± {scores.std():.4f}")
            
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_pipeline = pipeline
                pipeline.fit(X, y)
        
        print(f"\nBest model: {self.best_pipeline.named_steps['model'].__class__.__name__}")
        print(f"Best score: {self.best_score:.4f}")
        
        return results
    
    def predict(self, X):
        """Make predictions with best pipeline"""
        if self.best_pipeline is None:
            raise ValueError("Pipeline not fitted yet. Call fit() first.")
        return self.best_pipeline.predict(X)

# ============================================================================
# IMBALANCED DATA HANDLING
# ============================================================================

def handle_imbalanced_data_smote(X, y):
    """Handle imbalanced data using SMOTE"""
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def handle_imbalanced_data_adasyn(X, y):
    """Handle imbalanced data using ADASYN"""
    from imblearn.over_sampling import ADASYN
    
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def handle_imbalanced_data_undersampling(X, y):
    """Handle imbalanced data using undersampling"""
    from imblearn.under_sampling import RandomUnderSampler
    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

# ============================================================================
# OUTLIER DETECTION AND REMOVAL
# ============================================================================

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        print(f"Removing {len(outliers)} outliers from {col}")
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def detect_outliers_isolation_forest(X, contamination=0.1):
    """Detect outliers using Isolation Forest"""
    from sklearn.ensemble import IsolationForest
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(X)
    
    n_outliers = (outliers == -1).sum()
    print(f"Detected {n_outliers} outliers ({n_outliers/len(X)*100:.2f}%)")
    
    return outliers

# ============================================================================
# PRODUCTION PIPELINE TEMPLATE
# ============================================================================

class ProductionMLPipeline:
    """Production-ready ML pipeline template"""
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.pipeline = None
        self.is_fitted = False
        
    def build_pipeline(self):
        """Build the complete pipeline"""
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
        
    def fit(self, X, y):
        """Fit the pipeline"""
        if self.pipeline is None:
            self.build_pipeline()
        
        self.pipeline.fit(X, y)
        self.is_fitted = True
        print("Pipeline fitted successfully")
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.pipeline.predict_proba(X)
    
    def save(self, filename='production_pipeline.pkl'):
        """Save pipeline to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")
        
        joblib.dump(self.pipeline, filename)
        print(f"Pipeline saved to {filename}")
    
    @classmethod
    def load(cls, filename='production_pipeline.pkl'):
        """Load pipeline from disk"""
        pipeline = joblib.load(filename)
        
        # Create instance with loaded pipeline
        instance = cls(None, None)
        instance.pipeline = pipeline
        instance.is_fitted = True
        
        print(f"Pipeline loaded from {filename}")
        return instance

# ============================================================================
# QUICK START TEMPLATE
# ============================================================================

def quick_ml_workflow(X_train, y_train, X_test, y_test, 
                     numerical_features, categorical_features):
    """Quick ML workflow from data to predictions"""
    
    # 1. Create preprocessing pipeline
    print("Step 1: Creating preprocessing pipeline...")
    preprocessor = create_full_preprocessing_pipeline(numerical_features, categorical_features)
    
    # 2. Create model
    print("\nStep 2: Creating model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 3. Create complete pipeline
    print("\nStep 3: Building complete pipeline...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # 4. Train
    print("\nStep 4: Training...")
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    print("\nStep 5: Evaluating...")
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # 6. Predictions
    print("\nStep 6: Making predictions...")
    predictions = pipeline.predict(X_test)
    
    # 7. Save model
    print("\nStep 7: Saving model...")
    save_pipeline(pipeline, 'trained_pipeline.pkl')
    
    return pipeline, predictions
