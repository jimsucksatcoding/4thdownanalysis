import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import calibration_curve


# Load wpep/nflfastR dataset (has all needed features)
df2 = pd.read_csv("data/2015-2024wpep.csv")

# Select key columns from wpep dataset - include game_date and season if available
cols_to_select = ['play_id', 'game_id', 'game_date', 'ep', 'wp', 'down', 'qtr', 'ydstogo', 
                  'yardline_100', 'fourth_down_converted', 'fourth_down_failed',
                  'yards_gained', 'play_type', 'shotgun', 'no_huddle', 'qb_dropback',
                  'score_differential', 'quarter_seconds_remaining', 'half_seconds_remaining',
                  'game_seconds_remaining', 'goal_to_go',
                  'posteam_type', 'season_type', 'week',
                  'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
                  'vegas_wp', 'vegas_home_wp',
                  'no_score_prob', 'fg_prob', 'td_prob',
                  'opp_fg_prob', 'opp_td_prob']

# Add season if it exists
if 'season' in df2.columns:
    cols_to_select.append('season')

# Select columns (only those that exist)
data = df2[[c for c in cols_to_select if c in df2.columns]].copy()

# Filter to 4th downs
data = data[data['down'] == 4].copy()

# Convert game_date to datetime
if 'game_date' in data.columns:
    data['game_date'] = pd.to_datetime(data['game_date'], errors='coerce')

# Create success metric: only for actual go-for-it plays (not punts/FGs)
# Filter to plays where we have explicit conversion/failure flags
go_mask = (data['fourth_down_converted'] == 1) | (data['fourth_down_failed'] == 1)
data = data[go_mask].copy()

# Success is simply whether the conversion flag is set
data['success'] = (data['fourth_down_converted'] == 1).astype(int)

# Build features
# Core probability features: ep, wp (removed epa, wpa - these are post-play, label leakage)
# Field position: ydstogo, yardline_100
# Game context: score_differential, qtr, goal_to_go
# Time remaining: quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining
# Play indicators: shotgun, no_huddle, qb_dropback

feature_cols = ['ep', 'wp', 'ydstogo', 'yardline_100', 
                'score_differential', 'qtr', 'goal_to_go',
                'game_seconds_remaining',  # Removed redundant quarter/half time features
                'week', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
                'vegas_wp', 'vegas_home_wp',
                'no_score_prob', 'fg_prob', 'td_prob',
                'opp_fg_prob', 'opp_td_prob']

# Add binary features for formation/play type (with guards)
for col, new_col in [
    ('shotgun', 'shotgun_binary'),
    ('no_huddle', 'no_huddle_binary'),
    ('qb_dropback', 'qb_dropback_binary'),
]:
    if col in data.columns:
        data[new_col] = (data[col] == 1).astype(int)
        feature_cols.append(new_col)

# Add play type indicators from play_type column (skip punt/FG - always 0 after go-for-it filter)
if 'play_type' in data.columns:
    data['is_pass_play'] = (data['play_type'] == 'pass').astype(int)
    data['is_rush_play'] = (data['play_type'] == 'run').astype(int)
    feature_cols.extend(['is_pass_play', 'is_rush_play'])

# Add categorical indicators for posteam_type and season_type
if 'posteam_type' in data.columns:
    data['is_home'] = (data['posteam_type'] == 'home').astype(int)
    feature_cols.append('is_home')

if 'season_type' in data.columns:
    data['is_regular_season'] = (data['season_type'] == 'REG').astype(int)
    feature_cols.append('is_regular_season')

# Prepare X and y with proper missing data handling
X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
y = data['success']

# Drop rows with any missing or infinite feature values
mask = ~X.isna().any(axis=1)
data = data[mask].copy()
X = X[mask].copy()
y = y[mask].copy()

# Leave-one-season-out cross-validation (Yurko, Ventura, Horowitz methodology)
# Use season column if available, otherwise extract from game_date
if 'season' in data.columns:
    data['season'] = data['season'].astype(int)
else:
    data['season'] = pd.to_datetime(data['game_date']).dt.year

# Season is now aligned with X and y after filtering
season_aligned = data['season']

# Model complexity sanity check: events per feature
n_obs = len(data)
n_events = int(data['success'].sum())
n_features = len(feature_cols)
events_per_feature = n_events / n_features

print(f"Model Complexity Check:")
print(f"  N observations: {n_obs}")
print(f"  N conversions: {n_events}")
print(f"  N features: {n_features}")
print(f"  Events per feature: {events_per_feature:.1f}")
if events_per_feature < 10:
    print(f"  WARNING: Events per feature < 10, consider reducing features or stronger regularization")
print()

# Get unique seasons
seasons = sorted(season_aligned.unique())
print(f"Leave-One-Season-Out Cross-Validation")
print(f"Available seasons: {seasons}\n")

# Store results for each fold
fold_results = []
all_preds = []
all_y_test = []
all_probs = []

# For each season, train on all other seasons and test on that season
for test_season in seasons:
    train_mask = season_aligned != test_season
    test_mask = season_aligned == test_season
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    
    # Skip if test set is too small
    if len(X_test) < 10:
        print(f"Skipping {test_season}: only {len(X_test)} samples")
        continue
    
    # Fit logistic regression with scaling (helps convergence and stability)
    mod = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=5000)
    )
    mod.fit(X_train, y_train)
    
    # Predictions
    preds = mod.predict(X_test)
    probs = mod.predict_proba(X_test)[:, 1]  # Probability of success (class 1)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    logloss = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    
    success_rate_train = y_train.mean()
    success_rate_test = y_test.mean()
    
    fold_results.append({
        'season': test_season,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_success_rate': success_rate_train,
        'test_success_rate': success_rate_test,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'log_loss': logloss,
        'brier_score': brier,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    all_preds.extend(preds)
    all_y_test.extend(y_test)
    all_probs.extend(probs)
    
    print(f"Season {test_season}: Train={len(X_train)}, Test={len(X_test)}")
    print(f"  Train Success={success_rate_train:.3f}, Test Success={success_rate_test:.3f}")
    print(f"  Accuracy={accuracy:.3f}, ROC-AUC={roc_auc:.3f}, Log Loss={logloss:.3f}, Brier={brier:.3f}")
    print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"  Confusion Matrix:\n{cm}")

# Overall results across all folds
all_preds = np.array(all_preds)
all_y_test = np.array(all_y_test)
all_probs = np.array(all_probs)

overall_accuracy = accuracy_score(all_y_test, all_preds)
overall_roc_auc = roc_auc_score(all_y_test, all_probs)
overall_logloss = log_loss(all_y_test, all_probs)
overall_brier = brier_score_loss(all_y_test, all_probs)
overall_precision = precision_score(all_y_test, all_preds, zero_division=0)
overall_recall = recall_score(all_y_test, all_preds, zero_division=0)
overall_f1 = f1_score(all_y_test, all_preds, zero_division=0)
overall_cm = confusion_matrix(all_y_test, all_preds)

print(f"\n{'='*60}")
print(f"Overall Cross-Validation Results (All Folds Combined):")
print(f"{'='*60}")
print(f"  Total test samples: {len(all_y_test)}")
print(f"  Average test success rate: {all_y_test.mean():.3f}")
print(f"\n  Classification Metrics:")
print(f"    Accuracy:  {overall_accuracy:.4f}")
print(f"    ROC-AUC:   {overall_roc_auc:.4f}")
print(f"    Log Loss:  {overall_logloss:.4f}")
print(f"    Brier:     {overall_brier:.4f}")
print(f"    Precision: {overall_precision:.4f}")
print(f"    Recall:    {overall_recall:.4f}")
print(f"    F1 Score:  {overall_f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    {overall_cm}")
print(f"    (Rows: Actual, Cols: Predicted)")
print(f"    [Failed, Converted]")
print(f"\n  Per-Season Average Metrics:")
print(f"    Accuracy:  {np.mean([r['accuracy'] for r in fold_results]):.4f}")
print(f"    ROC-AUC:   {np.mean([r['roc_auc'] for r in fold_results]):.4f}")
print(f"    Log Loss:  {np.mean([r['log_loss'] for r in fold_results]):.4f}")
print(f"    Brier:     {np.mean([r['brier_score'] for r in fold_results]):.4f}")
print(f"    F1:        {np.mean([r['f1'] for r in fold_results]):.4f}")

# Fit final model on all data with calibration for probability accuracy
# Note: Coefficients are prediction weights, not causal/importance metrics
# Given collinearity (EP/WP vs yardline/time/score), interpret with caution
print(f"\nFinal Model (trained on all data):")

# Base model with scaling
base_clf = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    LogisticRegression(max_iter=5000)
)

# Calibrated model for better probability estimates
# Note: parameter name changed from 'base_estimator' to 'estimator' in newer sklearn versions
calibrated = CalibratedClassifierCV(
    estimator=base_clf,
    method='isotonic',  # isotonic works well with plenty of data
    cv=5  # 5-fold internal CV for calibration
)

calibrated.fit(X, y)

# Get coefficients from the base model (on standardized scale)
coef_model = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    LogisticRegression(max_iter=5000)
)
coef_model.fit(X, y)
logit = coef_model.named_steps['logisticregression']

print(f"Feature coefficients (on standardized scale, prediction weights not importance):")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {logit.coef_[0][i]:.4f}")

# Calibration check: reliability diagram
print(f"\nCalibration Check:")
prob_true, prob_pred = calibration_curve(all_y_test, all_probs, n_bins=10)
print(f"  Calibration curve computed (use for visualization)")
print(f"  Brier score: {overall_brier:.4f} (lower is better)")
print(f"  If Brier < 0.25 and curve near diagonal, probabilities are well-calibrated")

# 1) Conversion probability vs yards to go (empirical + fitted) with 0.95 threshold
feat = 'ydstogo'

# Empirical success rate in bins
bins = np.arange(int(X[feat].min()), int(X[feat].max()) + 2)
bin_idx = np.digitize(X[feat], bins) - 1
bin_centers = []
bin_empirical = []

for i in range(len(bins) - 1):
    mask_i = bin_idx == i
    if mask_i.sum() == 0:
        continue
    bin_centers.append(X[feat][mask_i].mean())
    bin_empirical.append(y[mask_i].mean())

bin_centers = np.array(bin_centers)
bin_empirical = np.array(bin_empirical)

# Fitted curve: vary ydstogo, hold other features at their mean
grid = np.linspace(X[feat].min(), X[feat].max(), 100)
X_mean = X.mean().to_frame().T
X_grid = pd.concat([X_mean] * len(grid), ignore_index=True)
X_grid[feat] = grid
probs_grid = calibrated.predict_proba(X_grid)[:, 1]

plt.figure()
plt.scatter(bin_centers, bin_empirical, alpha=0.7, label='Empirical success rate')
plt.plot(grid, probs_grid, linewidth=2, label='Fitted probability')
plt.axhline(0.95, color='red', linestyle='--', label='0.95 threshold')
plt.xlabel('Yards to go on 4th down')
plt.ylabel('Probability of conversion')
plt.title('4th Down Conversion vs Yards to Go')
plt.legend()
plt.tight_layout()
plt.show()

# 2) ROC curve (overall, from LOSO CV probabilities)
plt.figure()
RocCurveDisplay.from_predictions(all_y_test, all_probs)
plt.title('ROC Curve - 4th Down Conversion Model')
plt.tight_layout()
plt.show()

# 3) Calibration (reliability) curve
prob_true, prob_pred = calibration_curve(all_y_test, all_probs, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.title('Calibration Curve - 4th Down Conversion Model')
plt.legend()
plt.tight_layout()
plt.show()

# 4) Distribution of predicted probabilities by outcome
plt.figure()
plt.hist(all_probs[all_y_test == 0], bins=20, alpha=0.6, density=True, label='Failed')
plt.hist(all_probs[all_y_test == 1], bins=20, alpha=0.6, density=True, label='Converted')
plt.xlabel('Predicted probability of conversion')
plt.ylabel('Density')
plt.title('Predicted Probability Distribution by Outcome')
plt.legend()
plt.tight_layout()
plt.show()

# 5) Per-season ROC-AUC bar chart (from fold_results)
seasons_cv = [r['season'] for r in fold_results]
auc_cv = [r['roc_auc'] for r in fold_results]

plt.figure()
plt.bar(seasons_cv, auc_cv)
plt.xlabel('Season')
plt.ylabel('ROC-AUC')
plt.title('Per-Season ROC-AUC (Leave-One-Season-Out)')
plt.tight_layout()
plt.show()




