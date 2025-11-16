"""
ML Validation and K-Fold Cross-Validation Techniques Cheatsheet
Comprehensive guide to model validation, cross-validation strategies, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedShuffleSplit,
    cross_val_score, cross_validate, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

# ============================================================================
# BASIC TRAIN-TEST SPLIT
# ============================================================================

def simple_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """Basic train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify  # Use y for stratified split
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """Split into train, validation, and test sets"""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from train
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================

def k_fold_cv(model, X, y, n_splits=5, random_state=42, scoring='accuracy'):
    """Standard K-Fold Cross-Validation"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    
    print(f"K-Fold CV Scores: {scores}")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    return scores

def k_fold_cv_detailed(model, X, y, n_splits=5, random_state=42):
    """K-Fold with detailed metrics"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        fold_results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"Fold {fold}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    
    # Aggregate results
    df_results = pd.DataFrame(fold_results)
    print("\nAverage across folds:")
    print(df_results.mean())
    
    return df_results

# ============================================================================
# STRATIFIED K-FOLD
# ============================================================================

def stratified_k_fold_cv(model, X, y, n_splits=5, random_state=42):
    """Stratified K-Fold for imbalanced datasets"""
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
    
    print(f"Stratified K-Fold CV Scores: {scores}")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    return scores

def stratified_k_fold_with_class_distribution(X, y, n_splits=5):
    """Stratified K-Fold showing class distribution in each fold"""
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y), 1):
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"\nFold {fold}:")
        print(f"Train distribution: {np.bincount(y_train)}")
        print(f"Val distribution: {np.bincount(y_val)}")

# ============================================================================
# GROUP K-FOLD
# ============================================================================

def group_k_fold_cv(model, X, y, groups, n_splits=5):
    """Group K-Fold - ensures groups don't appear in both train and val"""
    gkfold = GroupKFold(n_splits=n_splits)
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkfold.split(X, y, groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold}: Score={score:.4f}")
        print(f"Train groups: {np.unique(groups[train_idx])}")
        print(f"Val groups: {np.unique(groups[val_idx])}\n")
    
    print(f"Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    
    return scores

# ============================================================================
# TIME SERIES CROSS-VALIDATION
# ============================================================================

def time_series_cv(model, X, y, n_splits=5, test_size=None):
    """Time Series Split - respects temporal order"""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold}: Train size={len(train_idx)}, Val size={len(val_idx)}, Score={score:.4f}")
    
    print(f"\nMean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    
    return scores

def walk_forward_validation(model, X, y, initial_train_size, step_size=1):
    """Walk-forward validation for time series"""
    scores = []
    
    for i in range(initial_train_size, len(X) - step_size + 1, step_size):
        X_train, y_train = X[:i], y[:i]
        X_test, y_test = X[i:i+step_size], y[i:i+step_size]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        score = accuracy_score(y_test, predictions)
        scores.append(score)
    
    print(f"Walk-forward validation: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")
    
    return scores

# ============================================================================
# LEAVE-ONE-OUT & LEAVE-P-OUT
# ============================================================================

def leave_one_out_cv(model, X, y):
    """Leave-One-Out Cross-Validation"""
    loo = LeaveOneOut()
    
    scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
    
    print(f"Leave-One-Out CV: Mean={scores.mean():.4f}, Std={scores.std():.4f}")
    print(f"Number of iterations: {len(scores)}")
    
    return scores

def leave_p_out_cv(model, X, y, p=2, max_samples=100):
    """Leave-P-Out Cross-Validation (limited to max_samples)"""
    lpo = LeavePOut(p=p)
    
    # Note: LeavePOut can be very expensive, so we limit iterations
    scores = []
    
    for i, (train_idx, test_idx) in enumerate(lpo.split(X)):
        if i >= max_samples:
            break
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    print(f"Leave-{p}-Out CV ({len(scores)} samples): Mean={np.mean(scores):.4f}")
    
    return scores

# ============================================================================
# SHUFFLE SPLIT
# ============================================================================

def shuffle_split_cv(model, X, y, n_splits=10, test_size=0.2, random_state=42):
    """ShuffleSplit - random train/test splits"""
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    scores = cross_val_score(model, X, y, cv=ss, scoring='accuracy')
    
    print(f"ShuffleSplit CV Scores: {scores}")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    return scores

def stratified_shuffle_split_cv(model, X, y, n_splits=10, test_size=0.2, random_state=42):
    """Stratified ShuffleSplit"""
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    scores = cross_val_score(model, X, y, cv=sss, scoring='accuracy')
    
    print(f"Stratified ShuffleSplit CV Scores: {scores}")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    return scores

# ============================================================================
# NESTED CROSS-VALIDATION
# ============================================================================

def nested_cross_validation(model, param_grid, X, y, outer_cv=5, inner_cv=3):
    """Nested CV for unbiased model selection and evaluation"""
    from sklearn.model_selection import GridSearchCV
    
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    outer_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=inner_cv, scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate best model on outer test set
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test, y_test)
        outer_scores.append(score)
        
        print(f"Outer Fold {fold}: Best params={grid_search.best_params_}, Score={score:.4f}")
    
    print(f"\nNested CV Score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    
    return outer_scores

# ============================================================================
# PYTORCH K-FOLD CROSS-VALIDATION
# ============================================================================

def pytorch_k_fold_cv(model_class, dataset, n_splits=5, batch_size=32, 
                      num_epochs=10, device='cuda', **model_kwargs):
    """K-Fold CV for PyTorch models"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
        print(f"\nFold {fold}/{n_splits}")
        print("=" * 50)
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        # Initialize model
        model = model_class(**model_kwargs).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        fold_results.append(accuracy)
        
        print(f"Fold {fold} Accuracy: {accuracy:.2f}%")
    
    print(f"\nAverage Accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%")
    
    return fold_results

# ============================================================================
# CROSS-VALIDATION WITH MULTIPLE METRICS
# ============================================================================

def cv_with_multiple_metrics(model, X, y, cv=5):
    """Cross-validation with multiple scoring metrics"""
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc_ovr_weighted'
    }
    
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                 return_train_score=True)
    
    print("Cross-Validation Results:")
    print("-" * 60)
    
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        print(f"{metric.upper()}:")
        print(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
        print(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
    
    return cv_results

# ============================================================================
# CUSTOM CROSS-VALIDATION STRATEGIES
# ============================================================================

def custom_cv_split(X, y, n_splits=5, custom_split_func=None):
    """Custom cross-validation splitting strategy"""
    if custom_split_func is None:
        # Default: standard k-fold
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Use custom splitting function
    return custom_split_func(X, y, n_splits)

def stratified_group_k_fold(X, y, groups, n_splits=5):
    """Custom stratified group k-fold (manual implementation)"""
    from collections import defaultdict
    
    # Group samples by group and class
    group_class_indices = defaultdict(lambda: defaultdict(list))
    
    for idx, (group, label) in enumerate(zip(groups, y)):
        group_class_indices[group][label].append(idx)
    
    # Create folds ensuring both stratification and group separation
    folds = [[] for _ in range(n_splits)]
    
    for group in group_class_indices:
        # Assign entire group to one fold
        fold_idx = len(group) % n_splits
        for label in group_class_indices[group]:
            folds[fold_idx].extend(group_class_indices[group][label])
    
    return folds

# ============================================================================
# REPEATED K-FOLD
# ============================================================================

def repeated_k_fold_cv(model, X, y, n_splits=5, n_repeats=10, random_state=42):
    """Repeated K-Fold Cross-Validation"""
    from sklearn.model_selection import RepeatedKFold
    
    rkfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    scores = cross_val_score(model, X, y, cv=rkfold, scoring='accuracy')
    
    print(f"Repeated K-Fold CV ({n_splits} folds, {n_repeats} repeats):")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    print(f"Total iterations: {len(scores)}")
    
    return scores

def repeated_stratified_k_fold_cv(model, X, y, n_splits=5, n_repeats=10, random_state=42):
    """Repeated Stratified K-Fold"""
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    rskfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, 
                                      random_state=random_state)
    
    scores = cross_val_score(model, X, y, cv=rskfold, scoring='accuracy')
    
    print(f"Repeated Stratified K-Fold CV ({n_splits} folds, {n_repeats} repeats):")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    return scores

# ============================================================================
# VALIDATION CURVE & LEARNING CURVE
# ============================================================================

def plot_validation_curve(model, X, y, param_name, param_range, cv=5):
    """Generate validation curve for hyperparameter tuning"""
    from sklearn.model_selection import validation_curve
    
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    print(f"Validation Curve for {param_name}:")
    for i, param_val in enumerate(param_range):
        print(f"{param_val}: Train={train_mean[i]:.4f}±{train_std[i]:.4f}, "
              f"Val={val_mean[i]:.4f}±{val_std[i]:.4f}")
    
    return train_scores, val_scores

def plot_learning_curve(model, X, y, train_sizes=None, cv=5):
    """Generate learning curve to diagnose bias/variance"""
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    print("Learning Curve:")
    for i, size in enumerate(train_sizes):
        print(f"Size={size}: Train={train_mean[i]:.4f}±{train_std[i]:.4f}, "
              f"Val={val_mean[i]:.4f}±{val_std[i]:.4f}")
    
    return train_sizes, train_scores, val_scores

# ============================================================================
# REGRESSION METRICS WITH CV
# ============================================================================

def regression_cv_metrics(model, X, y, cv=5):
    """Cross-validation for regression with multiple metrics"""
    from sklearn.model_selection import cross_val_predict
    
    # Get predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("Regression Cross-Validation Metrics:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

# ============================================================================
# HOLDOUT VALIDATION
# ============================================================================

def holdout_validation(model, X, y, test_size=0.2, random_state=42):
    """Simple holdout validation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    # Training performance
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Test performance
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print("Holdout Validation:")
    print(f"Training Accuracy:   {train_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Difference:          {abs(train_acc - test_acc):.4f}")
    
    if train_acc - test_acc > 0.1:
        print("Warning: Possible overfitting detected!")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'model': model
    }

# ============================================================================
# BOOTSTRAPPING
# ============================================================================

def bootstrap_validation(model, X, y, n_iterations=1000, sample_size=None):
    """Bootstrap validation for model performance"""
    if sample_size is None:
        sample_size = len(X)
    
    scores = []
    
    for _ in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=sample_size, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # Out-of-bag samples
        oob_indices = list(set(range(len(X))) - set(indices))
        if len(oob_indices) == 0:
            continue
        
        X_oob = X[oob_indices]
        y_oob = y[oob_indices]
        
        # Train and evaluate
        model.fit(X_sample, y_sample)
        score = model.score(X_oob, y_oob)
        scores.append(score)
    
    scores = np.array(scores)
    
    print("Bootstrap Validation:")
    print(f"Mean Score: {scores.mean():.4f}")
    print(f"Std Dev:    {scores.std():.4f}")
    print(f"95% CI:     [{np.percentile(scores, 2.5):.4f}, {np.percentile(scores, 97.5):.4f}]")
    
    return scores

# ============================================================================
# EARLY STOPPING VALIDATION
# ============================================================================

def train_with_early_stopping(model, X_train, y_train, X_val, y_val, 
                              max_epochs=100, patience=10, verbose=True):
    """Training with early stopping based on validation performance"""
    best_val_score = -np.inf
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Train for one epoch
        model.fit(X_train, y_train)
        
        # Validate
        val_score = model.score(X_val, y_val)
        
        if verbose and epoch % 10 == 0:
            train_score = model.score(X_train, y_train)
            print(f"Epoch {epoch}: Train={train_score:.4f}, Val={val_score:.4f}")
        
        # Check improvement
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            # Save best model (in practice, use model.get_params() or similar)
            best_model_state = model
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    return best_model_state, best_val_score

# ============================================================================
# EVALUATION SUMMARY
# ============================================================================

def comprehensive_evaluation_summary(y_true, y_pred, y_pred_proba=None):
    """Comprehensive evaluation metrics summary"""
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 60)
    
    # Basic metrics
    print("\nClassification Metrics:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            print(f"ROC-AUC:   {auc:.4f}")
        except:
            pass
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("=" * 60)
