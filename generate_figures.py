"""
generate_figures.py
====================
Regenerates all publication-quality figures from the real CMU r4.2 dataset.
Saves PNGs to results/  — commit these to the repo.

Usage:
    python generate_figures.py

Requires: cmu-dataset/r4.2/  (email.csv, logon.csv, device.csv,
                               file.csv, http.csv, psychometric.csv)
          emfdscore installed  (pip install git+https://github.com/medianeuroscience/emfdscore.git)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             confusion_matrix, roc_auc_score, f1_score,
                             precision_score, recall_score)

sys.path.insert(0, os.path.dirname(__file__))
import real_pipeline as rp

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    'MFT':        '#2196F3',   # blue
    'TF-IDF':     '#9E9E9E',   # grey
    'Big Five':   '#FF9800',   # orange
    'Behavioral': '#4CAF50',   # green
    'MFT+Beh':    '#E91E63',   # pink/red
}
sns.set_theme(style='whitegrid', font_scale=1.1)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & prepare data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
CMU_N = 3000
scenario_users = rp.load_malicious_users(rp.CMU_ANSWERS_DIR)
cmu_df, psycho_df = rp.load_cmu_emails(
    rp.CMU_EMAIL_CSV, rp.CMU_PSYCHO_CSV, scenario_users, CMU_N)
cmu_df['content_clean'] = cmu_df['content'].apply(rp.preprocess_text)

train_df, test_df = rp.scenario_split(cmu_df, scenario_users)
if len(train_df) == 0 or len(test_df) == 0:
    train_df, test_df = train_test_split(
        cmu_df, test_size=0.2, stratify=cmu_df['label'], random_state=42)
y_train, y_test = train_df['label'].values, test_df['label'].values
print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

# ── Feature matrices ──────────────────────────────────────────────────────────
print("Building features...")
X_mft_tr, feat_names = rp.build_mft_features(train_df)
X_mft_te, _          = rp.build_mft_features(test_df)

vec = TfidfVectorizer(max_features=100, sublinear_tf=True,
                      min_df=2, max_df=0.95, ngram_range=(1, 2))
X_tfidf_tr = vec.fit_transform(train_df['content_clean'].fillna('')).toarray()
X_tfidf_te = vec.transform(test_df['content_clean'].fillna('')).toarray()

X_big5_tr, _ = rp.build_big5_features(train_df, psycho_df)
X_big5_te, _ = rp.build_big5_features(test_df,  psycho_df)

beh_df = rp.load_behavioral_logs('cmu-dataset/r4.2')
X_beh_tr, beh_cols = rp.build_behavioral_features(train_df, beh_df)
X_beh_te, _        = rp.build_behavioral_features(test_df,  beh_df)

X_combo_tr = np.hstack([X_mft_tr, X_beh_tr])
X_combo_te = np.hstack([X_mft_te, X_beh_te])
print("  Features built.")

# ── Quick CV helper ───────────────────────────────────────────────────────────
def cv_scores(X, y, n_est=30, folds=3):
    cv  = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    models = {
        'RF':  RandomForestClassifier(n_estimators=n_est, n_jobs=-1,
                                      class_weight='balanced', random_state=42),
        'GBT': GradientBoostingClassifier(n_estimators=n_est, max_depth=3,
                                          random_state=42),
        'SVM': Pipeline([('sc', StandardScaler()),
                         ('clf', SVC(kernel='rbf', C=1, probability=True,
                                     class_weight='balanced'))]),
    }
    out = {}
    for name, clf in models.items():
        s = cross_validate(clf, X, y, cv=cv,
                           scoring=['f1', 'roc_auc', 'precision', 'recall'])
        out[name] = {k.replace('test_', ''): v.mean() for k, v in s.items()}
    return out

print("Training models (this takes ~30s)...")
cv_mft   = cv_scores(X_mft_tr,   y_train)
cv_tfidf = cv_scores(X_tfidf_tr, y_train)
cv_big5  = cv_scores(X_big5_tr,  y_train)
cv_beh   = cv_scores(X_beh_tr,   y_train)
cv_combo = cv_scores(X_combo_tr, y_train)
print("  Training done.")

# ── Fit best model for ROC / PR / confusion ───────────────────────────────────
best_clf = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
best_clf.fit(X_combo_tr, y_train)
y_prob_combo = best_clf.predict_proba(X_combo_te)[:, 1]

# MFT-only GBT for comparison curves
gbt_mft = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
gbt_mft.fit(X_mft_tr, y_train)
y_prob_mft = gbt_mft.predict_proba(X_mft_te)[:, 1]

# Optimal threshold on combo model
thresh = 0.20
y_pred_combo = (y_prob_combo >= thresh).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Feature-set comparison bar chart  (F1 + AUC, best model per set)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1: feature-set comparison...")

sets   = ['MFT', 'TF-IDF', 'Big Five', 'Behavioral', 'MFT+Beh']
cvs    = [cv_mft, cv_tfidf, cv_big5, cv_beh, cv_combo]
colors = [PALETTE[s] for s in sets]

best_f1  = [max(cv[m]['f1']      for m in cv) for cv in cvs]
best_auc = [max(cv[m]['roc_auc'] for m in cv) for cv in cvs]

x = np.arange(len(sets))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, best_f1,  w, label='F1',    color=colors, alpha=0.85)
b2 = ax.bar(x + w/2, best_auc, w, label='AUC',   color=colors, alpha=0.5,
            edgecolor=[c for c in colors], linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=11)
ax.set_ylim(0, 1.12); ax.set_ylabel('Score'); ax.legend()
ax.set_title('Feature-Set Comparison — Best Model per Set (3-fold CV)',
             fontweight='bold')
for bar in b1: ax.text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+0.02, f'{bar.get_height():.2f}',
                       ha='center', va='bottom', fontsize=9)
for bar in b2: ax.text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+0.02, f'{bar.get_height():.2f}',
                       ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('results/fig1_feature_comparison.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Confusion matrix (MFT+Behavioral GBT, threshold=0.20)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: confusion matrix...")

cm = confusion_matrix(y_test, y_pred_combo)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'],
            annot_kws={'size': 14})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
f1  = f1_score(y_test, y_pred_combo, zero_division=0)
auc_s = roc_auc_score(y_test, y_prob_combo)
ax.set_title(f'MFT + Behavioral GBT  (threshold=0.20)\nF1={f1:.3f}  AUC={auc_s:.3f}',
             fontweight='bold')
plt.tight_layout()
plt.savefig('results/fig2_confusion_matrix.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — MFT feature importance (RF on MFT features)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: MFT feature importance...")

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                            class_weight='balanced', random_state=42)
rf.fit(X_mft_tr, y_train)
importances = rf.feature_importances_
order = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors = ['#E91E63' if 'vice' in feat_names[i] else '#2196F3'
              for i in order]
ax.bar(range(len(feat_names)), importances[order], color=bar_colors)
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels([feat_names[i].replace('.', '\n') for i in order],
                   fontsize=9)
ax.set_ylabel('Importance')
ax.set_title('MFT Feature Importances (Random Forest)\n'
             'Blue = virtue   Pink = vice', fontweight='bold')
plt.tight_layout()
plt.savefig('results/fig3_mft_importance.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — MFT score distributions: malicious vs benign
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4: MFT distributions...")

mft_scores_df = pd.DataFrame(X_mft_tr, columns=feat_names)
mft_scores_df['Class'] = pd.Series(y_train).map({0: 'Benign', 1: 'Malicious'})

top4 = [feat_names[i] for i in order[:4]]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, feat in zip(axes, top4):
    for cls, color in [('Benign', '#2196F3'), ('Malicious', '#E91E63')]:
        vals = mft_scores_df[mft_scores_df['Class'] == cls][feat]
        ax.hist(vals, bins=25, alpha=0.6, color=color, label=cls, density=True)
    ax.set_title(feat.replace('.', '\n'), fontsize=10)
    ax.set_xlabel('Score')
    ax.legend(fontsize=8)
axes[0].set_ylabel('Density')
fig.suptitle('MFT Score Distributions — Malicious vs Benign (top 4 features)',
             fontweight='bold')
plt.tight_layout()
plt.savefig('results/fig4_mft_distributions.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — ROC curves: all feature sets
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: ROC curves...")

fig, ax = plt.subplots(figsize=(7, 6))
roc_sets = [
    ('MFT+Beh (ours)', X_combo_tr, X_combo_te, PALETTE['MFT+Beh'], 2.5),
    ('Behavioral',     X_beh_tr,   X_beh_te,   PALETTE['Behavioral'], 1.5),
    ('Big Five',       X_big5_tr,  X_big5_te,  PALETTE['Big Five'], 1.5),
    ('MFT only',       X_mft_tr,   X_mft_te,   PALETTE['MFT'], 1.5),
    ('TF-IDF',         X_tfidf_tr, X_tfidf_te, PALETTE['TF-IDF'], 1.5),
]
for label, Xtr, Xte, color, lw in roc_sets:
    g = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    g.fit(Xtr, y_train)
    yp = g.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, yp)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{label}  (AUC={roc_auc:.3f})',
            color=color, linewidth=lw)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Scenario-based Hold-out Test', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('results/fig5_roc_curves.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Precision-Recall curves
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: PR curves...")

fig, ax = plt.subplots(figsize=(7, 6))
for label, Xtr, Xte, color, lw in roc_sets:
    g = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    g.fit(Xtr, y_train)
    yp = g.predict_proba(Xte)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, yp)
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, label=f'{label}  (AP={pr_auc:.3f})',
            color=color, linewidth=lw)

baseline = y_test.mean()
ax.axhline(baseline, color='k', linestyle='--', linewidth=0.8,
           label=f'Baseline (AP={baseline:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Scenario-based Hold-out Test',
             fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results/fig6_pr_curves.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Classifier comparison within MFT+Beh feature set
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 7: classifier comparison...")

metrics = ['f1', 'roc_auc', 'precision', 'recall']
metric_labels = ['F1', 'AUC-ROC', 'Precision', 'Recall']
clfs = ['RF', 'GBT', 'SVM']
clf_colors = ['#1976D2', '#E91E63', '#388E3C']

x = np.arange(len(metrics))
w = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
for i, (clf, color) in enumerate(zip(clfs, clf_colors)):
    vals = [cv_combo[clf].get(m, 0) for m in metrics]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=clf, color=color, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.18); ax.set_ylabel('Score')
ax.set_title('MFT + Behavioral — Classifier Comparison (3-fold CV)',
             fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('results/fig7_classifier_comparison.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Behavioural feature importances
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 8: behavioral feature importances...")

rf_beh = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                class_weight='balanced', random_state=42)
rf_beh.fit(X_beh_tr, y_train)
beh_imp = rf_beh.feature_importances_
beh_order = np.argsort(beh_imp)[::-1]

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(beh_cols)), beh_imp[beh_order], color=PALETTE['Behavioral'])
ax.set_xticks(range(len(beh_cols)))
ax.set_xticklabels([beh_cols[i].replace('_', '\n') for i in beh_order], fontsize=9)
ax.set_ylabel('Importance')
ax.set_title('Behavioral Feature Importances (Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig('results/fig8_behavioral_importance.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — F1 breakdown: all feature sets × all classifiers (heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 9: F1 heatmap...")

cv_all = {
    'MFT':        cv_mft,
    'TF-IDF':     cv_tfidf,
    'Big Five':   cv_big5,
    'Behavioral': cv_beh,
    'MFT+Beh':    cv_combo,
}
heat_data = pd.DataFrame({
    feat_set: {clf: scores[clf]['f1'] for clf in ['RF', 'GBT', 'SVM']}
    for feat_set, scores in cv_all.items()
})

fig, ax = plt.subplots(figsize=(8, 3.5))
sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={'size': 11})
ax.set_title('F1 Score — All Feature Sets × Classifiers (3-fold CV)',
             fontweight='bold')
ax.set_ylabel('Classifier')
plt.tight_layout()
plt.savefig('results/fig9_f1_heatmap.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Threshold sensitivity (MFT+Beh GBT)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 10: threshold sensitivity...")

thresholds = np.linspace(0.05, 0.60, 50)
f1s, precs, recs = [], [], []
for t in thresholds:
    yp = (y_prob_combo >= t).astype(int)
    f1s.append(f1_score(y_test, yp, zero_division=0))
    precs.append(precision_score(y_test, yp, zero_division=0))
    recs.append(recall_score(y_test, yp, zero_division=0))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, f1s,   color=PALETTE['MFT+Beh'], linewidth=2.5, label='F1')
ax.plot(thresholds, precs, color=PALETTE['Behavioral'], linewidth=1.5,
        linestyle='--', label='Precision')
ax.plot(thresholds, recs,  color=PALETTE['MFT'], linewidth=1.5,
        linestyle=':', label='Recall')
best_t = thresholds[np.argmax(f1s)]
ax.axvline(best_t, color='black', linestyle='--', linewidth=0.8,
           label=f'Best threshold = {best_t:.2f}')
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.set_title('Threshold Sensitivity — MFT + Behavioral GBT\n'
             '(scenario-based hold-out test)', fontweight='bold')
ax.legend(); ax.set_xlim(0.05, 0.60); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('results/fig10_threshold_sensitivity.png', dpi=150)
plt.close()

print("\nAll 10 figures saved to results/")
print("\nFinal metrics (MFT+Beh GBT, threshold=0.20):")
print(f"  F1        = {f1_score(y_test, y_pred_combo, zero_division=0):.3f}")
print(f"  Precision = {precision_score(y_test, y_pred_combo, zero_division=0):.3f}")
print(f"  Recall    = {recall_score(y_test, y_pred_combo, zero_division=0):.3f}")
print(f"  AUC-ROC   = {roc_auc_score(y_test, y_prob_combo):.3f}")
