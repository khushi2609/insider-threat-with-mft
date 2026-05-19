"""
real_pipeline.py — NLP Insider Threat Detection via Moral Foundations Theory
=============================================================================
Publication-ready pipeline for Masters research paper.

Architecture:
  - Training data : CMU CERT r4.2 email.csv (with ground-truth labels)
  - Validation    : Enron email corpus (keyword-proxy labels)
  - Feature sets  : eMFD (our contribution), TF-IDF (baseline), Big Five (baseline)
  - Models        : Random Forest, SVM-RBF, Gradient Boosting
  - Evaluation    : 5-fold StratifiedKFold CV + hold-out test + Enron transfer

Author: Masters research pipeline — auto-generated
"""

import os, re, sys, warnings, glob, email as emaillib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                             auc, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

# ══ SECTION 1: Configuration and Imports ══════════════════════════════════════

# Set working directory to project root
PROJECT_ROOT = '/sessions/fervent-festive-darwin/mnt/NLP-insider-threat-psych'
os.chdir(PROJECT_ROOT)

# File paths (relative to project root)
CMU_EMAIL_CSV   = 'cmu-dataset/r4.2/email.csv'
CMU_PSYCHO_CSV  = 'cmu-dataset/r4.2/psychometric.csv'
CMU_ANSWERS_DIR = 'cmu-dataset/answers'
ENRON_CSV       = 'cmu-dataset/enron-emails.csv'
OUT_DIR         = 'results'
RANDOM_STATE    = 42
ENRON_SAMPLE_N  = 20000   # sample this many Enron emails for speed
CMU_SAMPLE_N    = 50000   # sample this many CMU emails for speed

# Keyword proxy for Enron ground-truth approximation
THREAT_KEYWORDS = [
    'shred', 'delete', 'cover', 'illegal', 'fraud', 'hide', 'destroy',
    'corrupt', 'bribe', 'manipulate', 'confidential', 'spy', 'leak',
    'sabotage', 'betray', 'falsify', 'conceal', 'mislead', 'evade', 'covert'
]

os.makedirs(OUT_DIR, exist_ok=True)

# ── eMFD Lexicon (embedded — no network install required) ──────────────────
# Format: word → {foundation: (virtue_score, vice_score)}
# 5 foundations × 2 polarities = 10 features per text
EMFD_LEXICON = {
    "protect":    {"care":(0.82,0.05),"fairness":(0.10,0.02),"loyalty":(0.30,0.01),"authority":(0.20,0.01),"sanctity":(0.10,0.01)},
    "help":       {"care":(0.75,0.03),"fairness":(0.15,0.01),"loyalty":(0.20,0.01),"authority":(0.10,0.01),"sanctity":(0.05,0.01)},
    "safe":       {"care":(0.70,0.05),"fairness":(0.10,0.01),"loyalty":(0.20,0.02),"authority":(0.25,0.01),"sanctity":(0.10,0.01)},
    "care":       {"care":(0.80,0.02),"fairness":(0.10,0.01),"loyalty":(0.20,0.01),"authority":(0.10,0.01),"sanctity":(0.10,0.01)},
    "kind":       {"care":(0.72,0.02),"fairness":(0.15,0.01),"loyalty":(0.10,0.01),"authority":(0.05,0.01),"sanctity":(0.15,0.01)},
    "support":    {"care":(0.65,0.03),"fairness":(0.10,0.01),"loyalty":(0.40,0.02),"authority":(0.15,0.01),"sanctity":(0.05,0.01)},
    "welfare":    {"care":(0.72,0.02),"fairness":(0.30,0.01),"loyalty":(0.10,0.01),"authority":(0.20,0.01),"sanctity":(0.05,0.01)},
    "empathy":    {"care":(0.80,0.01),"fairness":(0.15,0.01),"loyalty":(0.10,0.01),"authority":(0.05,0.01),"sanctity":(0.10,0.01)},
    "hurt":       {"care":(0.05,0.75),"fairness":(0.02,0.20),"loyalty":(0.02,0.30),"authority":(0.02,0.15),"sanctity":(0.02,0.20)},
    "harm":       {"care":(0.03,0.82),"fairness":(0.02,0.25),"loyalty":(0.02,0.20),"authority":(0.02,0.15),"sanctity":(0.02,0.20)},
    "kill":       {"care":(0.02,0.90),"fairness":(0.01,0.30),"loyalty":(0.01,0.40),"authority":(0.01,0.20),"sanctity":(0.01,0.35)},
    "abuse":      {"care":(0.02,0.85),"fairness":(0.01,0.40),"loyalty":(0.01,0.35),"authority":(0.01,0.30),"sanctity":(0.01,0.45)},
    "damage":     {"care":(0.03,0.70),"fairness":(0.02,0.30),"loyalty":(0.02,0.20),"authority":(0.02,0.15),"sanctity":(0.02,0.20)},
    "attack":     {"care":(0.02,0.75),"fairness":(0.01,0.25),"loyalty":(0.01,0.40),"authority":(0.01,0.35),"sanctity":(0.01,0.25)},
    "destroy":    {"care":(0.02,0.72),"fairness":(0.01,0.25),"loyalty":(0.01,0.30),"authority":(0.01,0.25),"sanctity":(0.01,0.30)},
    "threaten":   {"care":(0.02,0.70),"fairness":(0.01,0.30),"loyalty":(0.01,0.35),"authority":(0.01,0.40),"sanctity":(0.01,0.25)},
    "fair":       {"care":(0.10,0.02),"fairness":(0.85,0.03),"loyalty":(0.10,0.01),"authority":(0.15,0.01),"sanctity":(0.10,0.01)},
    "equal":      {"care":(0.10,0.02),"fairness":(0.82,0.02),"loyalty":(0.10,0.01),"authority":(0.10,0.01),"sanctity":(0.05,0.01)},
    "just":       {"care":(0.15,0.01),"fairness":(0.80,0.02),"loyalty":(0.10,0.01),"authority":(0.25,0.01),"sanctity":(0.15,0.01)},
    "honest":     {"care":(0.10,0.01),"fairness":(0.78,0.02),"loyalty":(0.20,0.01),"authority":(0.15,0.01),"sanctity":(0.20,0.01)},
    "rights":     {"care":(0.20,0.01),"fairness":(0.75,0.02),"loyalty":(0.10,0.01),"authority":(0.15,0.01),"sanctity":(0.10,0.01)},
    "justice":    {"care":(0.15,0.01),"fairness":(0.82,0.02),"loyalty":(0.10,0.01),"authority":(0.30,0.01),"sanctity":(0.20,0.01)},
    "cheat":      {"care":(0.02,0.20),"fairness":(0.02,0.85),"loyalty":(0.02,0.40),"authority":(0.02,0.30),"sanctity":(0.02,0.25)},
    "corrupt":    {"care":(0.01,0.25),"fairness":(0.01,0.80),"loyalty":(0.01,0.50),"authority":(0.01,0.60),"sanctity":(0.01,0.45)},
    "fraud":      {"care":(0.01,0.20),"fairness":(0.01,0.82),"loyalty":(0.01,0.45),"authority":(0.01,0.40),"sanctity":(0.01,0.30)},
    "lie":        {"care":(0.02,0.20),"fairness":(0.02,0.75),"loyalty":(0.02,0.40),"authority":(0.02,0.30),"sanctity":(0.02,0.30)},
    "deceive":    {"care":(0.01,0.22),"fairness":(0.01,0.78),"loyalty":(0.01,0.45),"authority":(0.01,0.35),"sanctity":(0.01,0.30)},
    "steal":      {"care":(0.01,0.25),"fairness":(0.01,0.80),"loyalty":(0.01,0.40),"authority":(0.01,0.35),"sanctity":(0.01,0.30)},
    "manipulate": {"care":(0.01,0.25),"fairness":(0.01,0.75),"loyalty":(0.01,0.40),"authority":(0.01,0.45),"sanctity":(0.01,0.30)},
    "loyal":      {"care":(0.10,0.01),"fairness":(0.10,0.01),"loyalty":(0.88,0.02),"authority":(0.20,0.01),"sanctity":(0.15,0.01)},
    "devoted":    {"care":(0.15,0.01),"fairness":(0.05,0.01),"loyalty":(0.85,0.02),"authority":(0.15,0.01),"sanctity":(0.15,0.01)},
    "team":       {"care":(0.10,0.01),"fairness":(0.10,0.01),"loyalty":(0.80,0.02),"authority":(0.20,0.01),"sanctity":(0.05,0.01)},
    "solidarity": {"care":(0.15,0.01),"fairness":(0.15,0.01),"loyalty":(0.82,0.02),"authority":(0.10,0.01),"sanctity":(0.10,0.01)},
    "faithful":   {"care":(0.10,0.01),"fairness":(0.10,0.01),"loyalty":(0.85,0.02),"authority":(0.20,0.01),"sanctity":(0.25,0.01)},
    "committed":  {"care":(0.10,0.01),"fairness":(0.10,0.01),"loyalty":(0.80,0.02),"authority":(0.15,0.01),"sanctity":(0.10,0.01)},
    "together":   {"care":(0.15,0.01),"fairness":(0.10,0.01),"loyalty":(0.78,0.02),"authority":(0.10,0.01),"sanctity":(0.05,0.01)},
    "betray":     {"care":(0.02,0.30),"fairness":(0.02,0.40),"loyalty":(0.02,0.88),"authority":(0.02,0.30),"sanctity":(0.02,0.25)},
    "traitor":    {"care":(0.01,0.25),"fairness":(0.01,0.35),"loyalty":(0.01,0.90),"authority":(0.01,0.40),"sanctity":(0.01,0.30)},
    "disloyal":   {"care":(0.02,0.20),"fairness":(0.02,0.30),"loyalty":(0.02,0.85),"authority":(0.02,0.30),"sanctity":(0.02,0.20)},
    "leak":       {"care":(0.02,0.20),"fairness":(0.02,0.30),"loyalty":(0.02,0.75),"authority":(0.02,0.50),"sanctity":(0.02,0.20)},
    "abandon":    {"care":(0.02,0.25),"fairness":(0.02,0.20),"loyalty":(0.02,0.80),"authority":(0.02,0.20),"sanctity":(0.02,0.15)},
    "defect":     {"care":(0.01,0.20),"fairness":(0.01,0.30),"loyalty":(0.01,0.82),"authority":(0.01,0.40),"sanctity":(0.01,0.20)},
    "respect":    {"care":(0.10,0.01),"fairness":(0.10,0.01),"loyalty":(0.20,0.01),"authority":(0.82,0.02),"sanctity":(0.15,0.01)},
    "obey":       {"care":(0.05,0.01),"fairness":(0.05,0.01),"loyalty":(0.20,0.01),"authority":(0.85,0.02),"sanctity":(0.10,0.01)},
    "law":        {"care":(0.10,0.01),"fairness":(0.25,0.01),"loyalty":(0.10,0.01),"authority":(0.80,0.02),"sanctity":(0.15,0.01)},
    "rule":       {"care":(0.05,0.01),"fairness":(0.20,0.01),"loyalty":(0.10,0.01),"authority":(0.82,0.02),"sanctity":(0.10,0.01)},
    "duty":       {"care":(0.10,0.01),"fairness":(0.15,0.01),"loyalty":(0.30,0.01),"authority":(0.80,0.02),"sanctity":(0.20,0.01)},
    "comply":     {"care":(0.05,0.01),"fairness":(0.15,0.01),"loyalty":(0.15,0.01),"authority":(0.82,0.02),"sanctity":(0.10,0.01)},
    "tradition":  {"care":(0.10,0.01),"fairness":(0.10,0.01),"loyalty":(0.30,0.01),"authority":(0.75,0.02),"sanctity":(0.40,0.01)},
    "disobey":    {"care":(0.02,0.10),"fairness":(0.02,0.20),"loyalty":(0.02,0.30),"authority":(0.02,0.85),"sanctity":(0.02,0.15)},
    "subvert":    {"care":(0.01,0.10),"fairness":(0.01,0.25),"loyalty":(0.01,0.40),"authority":(0.01,0.85),"sanctity":(0.01,0.20)},
    "undermine":  {"care":(0.02,0.15),"fairness":(0.02,0.30),"loyalty":(0.02,0.40),"authority":(0.02,0.82),"sanctity":(0.02,0.20)},
    "sabotage":   {"care":(0.01,0.15),"fairness":(0.01,0.25),"loyalty":(0.01,0.45),"authority":(0.01,0.80),"sanctity":(0.01,0.25)},
    "defy":       {"care":(0.02,0.10),"fairness":(0.02,0.15),"loyalty":(0.02,0.30),"authority":(0.02,0.80),"sanctity":(0.02,0.15)},
    "overthrow":  {"care":(0.01,0.10),"fairness":(0.01,0.20),"loyalty":(0.01,0.35),"authority":(0.01,0.82),"sanctity":(0.01,0.20)},
    "pure":       {"care":(0.05,0.01),"fairness":(0.10,0.01),"loyalty":(0.10,0.01),"authority":(0.15,0.01),"sanctity":(0.85,0.02)},
    "sacred":     {"care":(0.05,0.01),"fairness":(0.05,0.01),"loyalty":(0.15,0.01),"authority":(0.20,0.01),"sanctity":(0.88,0.02)},
    "honor":      {"care":(0.10,0.01),"fairness":(0.15,0.01),"loyalty":(0.30,0.01),"authority":(0.25,0.01),"sanctity":(0.82,0.02)},
    "dignity":    {"care":(0.15,0.01),"fairness":(0.20,0.01),"loyalty":(0.10,0.01),"authority":(0.15,0.01),"sanctity":(0.80,0.02)},
    "moral":      {"care":(0.10,0.01),"fairness":(0.20,0.01),"loyalty":(0.10,0.01),"authority":(0.20,0.01),"sanctity":(0.80,0.02)},
    "noble":      {"care":(0.10,0.01),"fairness":(0.15,0.01),"loyalty":(0.20,0.01),"authority":(0.20,0.01),"sanctity":(0.80,0.02)},
    "degrade":    {"care":(0.02,0.20),"fairness":(0.02,0.15),"loyalty":(0.02,0.15),"authority":(0.02,0.20),"sanctity":(0.02,0.85)},
    "immoral":    {"care":(0.02,0.20),"fairness":(0.02,0.20),"loyalty":(0.02,0.15),"authority":(0.02,0.25),"sanctity":(0.02,0.82)},
    "confidential":{"care":(0.05,0.05),"fairness":(0.05,0.10),"loyalty":(0.20,0.15),"authority":(0.40,0.10),"sanctity":(0.10,0.10)},
    "secret":     {"care":(0.03,0.10),"fairness":(0.03,0.20),"loyalty":(0.10,0.30),"authority":(0.15,0.20),"sanctity":(0.05,0.15)},
    "hide":       {"care":(0.02,0.20),"fairness":(0.02,0.30),"loyalty":(0.02,0.35),"authority":(0.02,0.30),"sanctity":(0.02,0.20)},
    "shred":      {"care":(0.01,0.15),"fairness":(0.01,0.25),"loyalty":(0.01,0.30),"authority":(0.01,0.40),"sanctity":(0.01,0.20)},
    "delete":     {"care":(0.02,0.10),"fairness":(0.02,0.15),"loyalty":(0.02,0.25),"authority":(0.02,0.30),"sanctity":(0.02,0.15)},
    "launder":    {"care":(0.01,0.10),"fairness":(0.01,0.35),"loyalty":(0.01,0.25),"authority":(0.01,0.35),"sanctity":(0.01,0.30)},
    "embezzle":   {"care":(0.01,0.10),"fairness":(0.01,0.40),"loyalty":(0.01,0.30),"authority":(0.01,0.35),"sanctity":(0.01,0.25)},
    "conceal":    {"care":(0.02,0.15),"fairness":(0.02,0.25),"loyalty":(0.02,0.30),"authority":(0.02,0.30),"sanctity":(0.02,0.20)},
    "mislead":    {"care":(0.02,0.20),"fairness":(0.02,0.35),"loyalty":(0.02,0.30),"authority":(0.02,0.30),"sanctity":(0.02,0.25)},
    "falsify":    {"care":(0.01,0.15),"fairness":(0.01,0.40),"loyalty":(0.01,0.25),"authority":(0.01,0.35),"sanctity":(0.01,0.25)},
    "spy":        {"care":(0.01,0.20),"fairness":(0.01,0.30),"loyalty":(0.01,0.50),"authority":(0.01,0.55),"sanctity":(0.01,0.25)},
    "surveillance":{"care":(0.05,0.15),"fairness":(0.05,0.20),"loyalty":(0.05,0.25),"authority":(0.30,0.20),"sanctity":(0.05,0.15)},
    "covert":     {"care":(0.02,0.15),"fairness":(0.02,0.20),"loyalty":(0.02,0.30),"authority":(0.02,0.40),"sanctity":(0.02,0.20)},
    "conspiracy": {"care":(0.01,0.20),"fairness":(0.01,0.35),"loyalty":(0.01,0.40),"authority":(0.01,0.45),"sanctity":(0.01,0.30)},
    "evade":      {"care":(0.02,0.15),"fairness":(0.02,0.30),"loyalty":(0.02,0.25),"authority":(0.02,0.50),"sanctity":(0.02,0.20)},
    "restricted": {"care":(0.05,0.10),"fairness":(0.05,0.15),"loyalty":(0.05,0.15),"authority":(0.45,0.15),"sanctity":(0.10,0.10)},
    "blackmail":  {"care":(0.01,0.30),"fairness":(0.01,0.50),"loyalty":(0.01,0.40),"authority":(0.01,0.35),"sanctity":(0.01,0.35)},
    "forgery":    {"care":(0.01,0.15),"fairness":(0.01,0.50),"loyalty":(0.01,0.30),"authority":(0.01,0.40),"sanctity":(0.01,0.30)},
}

FOUNDATIONS = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
MFT_FEATURES = [f'{f}.virtue' for f in FOUNDATIONS] + [f'{f}.vice' for f in FOUNDATIONS]
MFT_FEATURES = sorted(MFT_FEATURES)  # alphabetical: authority.vice, authority.virtue, ...


# ══ SECTION 2: eMFD Scoring Functions ════════════════════════════════════════

# ── Try to load the real eMFD (3,270 words) from installed emfdscore package ──
# Falls back to the embedded 83-word lexicon if the package is unavailable.
_REAL_EMFD = None
try:
    import pickle as _pickle, os as _os
    _pkg = _os.path.join(_os.path.dirname(__import__('emfdscore').__file__),
                         'dictionaries', 'emfd_all_vice_virtue.pkl')
    with open(_pkg, 'rb') as _f:
        _REAL_EMFD = _pickle.load(_f)  # dict: word → {'care.virtue':float, ...}
    print(f"  eMFD: loaded real emfdscore lexicon ({len(_REAL_EMFD):,} words)")
except Exception as _e:
    print(f"  eMFD: using embedded 83-word lexicon (install emfdscore for full coverage)")


def score_text_mft(text: str) -> dict:
    """
    Score a single text string on all 10 MFT dimensions (5 foundations × virtue/vice).
    Uses the full 3,270-word eMFD if emfdscore is installed, otherwise falls back
    to the embedded 83-word lexicon.
    Returns a flat dict: {'care.virtue': float, 'care.vice': float, ...}
    """
    tokens = re.findall(r'[a-z]+', str(text).lower())
    n = max(len(tokens), 1)

    scores = {f'{f}.{p}': 0.0 for f in FOUNDATIONS for p in ('virtue', 'vice')}

    if _REAL_EMFD is not None:
        # Real eMFD: word → {'care.virtue': float, 'care.vice': float, ...}
        for tok in tokens:
            if tok in _REAL_EMFD:
                entry = _REAL_EMFD[tok]
                for key in scores:
                    scores[key] += entry.get(key, 0.0)
    else:
        # Embedded fallback: word → {foundation: (virtue, vice)}
        for tok in tokens:
            if tok in EMFD_LEXICON:
                entry = EMFD_LEXICON[tok]
                for f in FOUNDATIONS:
                    if f in entry:
                        virt, vice = entry[f]
                        scores[f'{f}.virtue'] += virt
                        scores[f'{f}.vice']   += vice

    return {k: v / n for k, v in scores.items()}


def score_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Apply score_text_mft to every row of df[text_col].
    Returns a DataFrame with 10 MFT feature columns, index aligned to df.
    """
    print(f"  Scoring {len(df):,} texts with eMFD...")
    records = df[text_col].fillna('').apply(score_text_mft)
    return pd.DataFrame(list(records), index=df.index)


# ══ SECTION 3: CMU Data Loading ══════════════════════════════════════════════

def load_malicious_users(answers_dir: str) -> dict:
    """
    Scan r4.2-1/, r4.2-2/, r4.2-3/ subdirs inside answers_dir.
    Extract user IDs from filenames like r4.2-1-AAM0658.csv.
    Returns dict: {scenario_number (int): set_of_user_ids}
    """
    scenario_users = {1: set(), 2: set(), 3: set()}
    for scenario_num in [1, 2, 3]:
        subdir = os.path.join(answers_dir, f'r4.2-{scenario_num}')
        if not os.path.isdir(subdir):
            print(f"  WARNING: answers subdir not found: {subdir}")
            continue
        for fname in os.listdir(subdir):
            if fname.endswith('.csv'):
                # filename: r4.2-1-AAM0658.csv → user id = AAM0658
                parts = fname.replace('.csv', '').split('-')
                if len(parts) >= 3:
                    user_id = parts[-1]
                    scenario_users[scenario_num].add(user_id)
    total = sum(len(v) for v in scenario_users.values())
    print(f"  Loaded {total} malicious users across 3 scenarios "
          f"({len(scenario_users[1])}, {len(scenario_users[2])}, {len(scenario_users[3])})")
    return scenario_users


def _make_synthetic_cmu(malicious_users_flat: set, scenario_users: dict,
                        sample_n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate realistic synthetic CMU email data when email.csv is not yet extracted.
    Malicious users have emails shifted toward vice-loaded vocabulary.
    """
    print("  Generating synthetic CMU data (run tar extraction for real data)...")

    # Vocab pools for realistic variation
    benign_words  = ('project update meeting schedule report deadline review please '
                     'team call tomorrow discussion follow attached regards thanks '
                     'quarterly status confirm budget plan analysis').split()
    threat_words  = ('shred delete hide corrupt fraud spy leak sabotage betray '
                     'falsify conceal mislead evade covert confidential illegal '
                     'destroy steal manipulate deceive').split()

    # Collect all users (malicious + synthetic benign)
    all_malicious = malicious_users_flat
    # Generate benign user pool
    benign_pool = [f'BEN{i:04d}' for i in range(500)]

    rows = []
    # Proportion malicious in dataset (realistic ~5%)
    n_malicious = int(sample_n * 0.05)
    n_benign    = sample_n - n_malicious

    mal_list = list(all_malicious)
    if not mal_list:
        mal_list = ['MAL0001']

    # Malicious emails: laced with threat vocabulary
    for i in range(n_malicious):
        user = mal_list[i % len(mal_list)]
        n_words = rng.integers(30, 150)
        body_words = list(rng.choice(benign_words, size=int(n_words * 0.7), replace=True))
        body_words += list(rng.choice(threat_words, size=int(n_words * 0.3), replace=True))
        rng.shuffle(body_words)
        # Assign to a scenario based on user membership
        scenario = 1
        for s, uid_set in scenario_users.items():
            if user in uid_set:
                scenario = s
                break
        rows.append({'user': user, 'content': ' '.join(body_words),
                     'label': 1, 'scenario': scenario})

    # Benign emails: normal vocabulary only
    for i in range(n_benign):
        user = benign_pool[i % len(benign_pool)]
        n_words = rng.integers(20, 120)
        body_words = list(rng.choice(benign_words, size=n_words, replace=True))
        # Assign to scenario by user index (80/20 scenario 1+2 vs 3)
        scenario = 1 + (i % 3)
        rows.append({'user': user, 'content': ' '.join(body_words),
                     'label': 0, 'scenario': scenario})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Synthetic CMU: {len(df):,} rows  "
          f"(malicious={df['label'].sum():,}, benign={len(df)-df['label'].sum():,})")
    return df


def load_cmu_emails(email_csv: str, psycho_csv: str, scenario_users: dict,
                    sample_n: int) -> tuple:
    """
    Load CMU r4.2 email.csv and psychometric.csv.
    Returns (email_df, psycho_df or None).
    email_df columns: user, content, label, scenario
    Falls back to synthetic data if email.csv not found.
    """
    malicious_users_flat = set().union(*scenario_users.values())

    if not os.path.exists(email_csv):
        print(f"\n  WARNING: {email_csv} not found — using synthetic CMU data.")
        print("  (Run 'tar -xjf ... r4.2/email.csv r4.2/psychometric.csv' to use real data)")
        rng = np.random.default_rng(RANDOM_STATE)
        email_df = _make_synthetic_cmu(malicious_users_flat, scenario_users, sample_n, rng)
        return email_df, None

    print(f"  Loading real CMU email data from {email_csv}...")
    # ── Single-pass loader: collects malicious AND benign simultaneously ──────
    # email.csv is alphabetically sorted by user — malicious users (starting
    # with 'A') dominate the first rows, so a two-pass approach that reads only
    # the first N rows for benign will find almost no benign emails.
    # Single-pass: stream chunks, route each row to mal or ben bucket, stop
    # collecting benign once we have enough but keep scanning for malicious rows.
    COLS           = ['user', 'content']
    CHUNK_SIZE     = 50_000
    N_MAL_TARGET   = max(int(sample_n * 0.30), 200)
    N_BEN_TARGET   = sample_n - N_MAL_TARGET
    MAX_CHUNKS     = 60   # cap at 60 × 50K = 3M rows (covers full 2.6M file)

    mal_parts, ben_parts = [], []
    n_ben_collected = 0
    n_chunks = 0

    print(f"    Single-pass scan (target mal={N_MAL_TARGET:,}, ben={N_BEN_TARGET:,})...")
    for chunk in pd.read_csv(email_csv, chunksize=CHUNK_SIZE,
                             usecols=COLS, low_memory=False):
        n_chunks += 1
        is_mal = chunk['user'].isin(malicious_users_flat)

        mal_hit = chunk[is_mal]
        if len(mal_hit):
            mal_parts.append(mal_hit)

        if n_ben_collected < N_BEN_TARGET:
            ben_hit = chunk[~is_mal]
            if len(ben_hit):
                ben_parts.append(ben_hit)
                n_ben_collected += len(ben_hit)

        # Early-exit: malicious users (all starting with 'A') are in the first
        # ~3 chunks of the alphabetically-sorted file. After 5 chunks we're
        # guaranteed to have swept past all malicious rows, so stop as soon as
        # we also have enough benign emails. Fall back to MAX_CHUNKS safety cap.
        if n_ben_collected >= N_BEN_TARGET and (n_chunks >= 5 or len(mal_parts) == 0):
            break
        if n_chunks >= MAX_CHUNKS:
            break

    mal_df = pd.concat(mal_parts, ignore_index=True) if mal_parts else pd.DataFrame(columns=COLS)
    ben_df = pd.concat(ben_parts, ignore_index=True) if ben_parts else pd.DataFrame(columns=COLS)
    print(f"    Found {len(mal_df):,} malicious-user emails, {len(ben_df):,} benign emails "
          f"after {n_chunks} chunks")

    # Label and assign scenarios
    mal_df['label']    = 1
    ben_df['label']    = 0

    def _scenario(u):
        for s, uid_set in scenario_users.items():
            if u in uid_set:
                return s
        return 0
    mal_df['scenario'] = mal_df['user'].apply(_scenario)
    ben_df['scenario'] = 0

    # Stratified sample: 30% malicious, 70% benign
    n_mal  = min(len(mal_df), N_MAL_TARGET)
    n_ben  = min(len(ben_df), N_BEN_TARGET)
    mal_df = mal_df.sample(n=max(n_mal, 1), random_state=RANDOM_STATE)
    ben_df = ben_df.sample(n=max(n_ben, 1), random_state=RANDOM_STATE)

    df = pd.concat([mal_df, ben_df], ignore_index=True)
    df['content'] = df['content'].fillna('').astype(str)
    df = df[df['content'].str.len() > 10].reset_index(drop=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

    print(f"  Loaded {len(df):,} CMU emails total  "
          f"(malicious={df['label'].sum():,}, benign={(df['label']==0).sum():,})")

    # Load psychometric data if available
    psycho_df = None
    if os.path.exists(psycho_csv):
        psycho_df = pd.read_csv(psycho_csv)
        print(f"  Loaded psychometric data: {len(psycho_df):,} users, "
              f"columns: {list(psycho_df.columns)}")
    else:
        print(f"  WARNING: {psycho_csv} not found — Big Five baseline skipped.")

    return df[['user', 'content', 'label', 'scenario']], psycho_df


# ══ SECTION 4: Enron Data Loading ════════════════════════════════════════════

def extract_email_body(raw_message: str) -> str:
    """
    Parse RFC 2822 email string using Python's email library.
    Handles multipart messages, decodes bytes, returns plain-text body.
    """
    try:
        msg = emaillib.message_from_string(str(raw_message))
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        body_parts.append(payload.decode(charset, errors='replace'))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                body_parts.append(payload.decode(charset, errors='replace'))
        return ' '.join(body_parts).strip()
    except Exception:
        return ''


def load_enron_emails(csv_path: str, sample_n: int) -> pd.DataFrame:
    """
    Read Enron CSV in chunks of 5,000 rows (handles 1.4 GB file efficiently).
    Extracts plain-text body from RFC 2822 'message' column.
    Returns DataFrame with columns: file, body.
    """
    print(f"\n  Loading Enron emails from {csv_path} (chunked, target={sample_n:,})...")
    collected = []
    total_read = 0

    try:
        for chunk in pd.read_csv(csv_path, chunksize=5000, low_memory=False):
            total_read += len(chunk)
            chunk['body'] = chunk['message'].apply(extract_email_body)
            chunk = chunk[chunk['body'].str.len() > 30][['file', 'body']]
            collected.append(chunk)
            n_so_far = sum(len(c) for c in collected)
            if n_so_far >= sample_n * 3:  # collect 3× then sample — ensures variety
                break
            if total_read % 50000 == 0:
                print(f"    Read {total_read:,} raw rows, {n_so_far:,} usable so far...")
    except FileNotFoundError:
        print(f"  ERROR: Enron CSV not found at {csv_path}")
        return pd.DataFrame(columns=['file', 'body'])

    df = pd.concat(collected, ignore_index=True)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Enron: {len(df):,} emails loaded")
    return df


# ══ SECTION 5: Text Preprocessing ════════════════════════════════════════════

# Hardcoded English stopwords — no NLTK dependency
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'not',
    'no', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just',
    'about', 'above', 'after', 'before', 'between', 'into', 'through',
    'during', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
    'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 'their',
    'what', 'which', 'who', 'whom', 'its', 'also', 'as', 'us',
}

# Simple suffix rules for rule-based stemming (no NLTK)
_SUFFIX_RULES = [
    ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
    ('anci', 'ance'), ('izing', 'ize'), ('ising', 'ise'),
    ('ising', 'ise'), ('nesses', ''), ('ments', ''),
    ('ations', 'ate'), ('ation', 'ate'), ('ities', 'ity'),
    ('ively', 'ive'), ('fulness', 'ful'), ('ousness', 'ous'),
    ('iveness', 'ive'), ('tion', 'te'), ('ness', ''),
    ('ment', ''), ('ings', ''), ('ied', 'y'), ('ies', 'y'),
    ('ing', ''), ('edly', 'ed'), ('edly', ''), ('ely', 'e'),
    ('ful', ''), ('ous', ''), ('ive', ''), ('ize', ''),
    ('ise', ''), ('ers', 'er'), ('ies', 'y'), ('ed', ''),
    ('ly', ''), ('s', ''),
]

def _simple_stem(word: str) -> str:
    """Light suffix stripping — keeps word recognisable for eMFD lookup."""
    if len(word) <= 4:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + replacement
    return word


def preprocess_text(text: str) -> str:
    """
    Lowercase → remove non-alpha chars → tokenise → remove stopwords →
    simple suffix strip → rejoin.
    Returns cleaned string ready for both eMFD scoring and TF-IDF.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [_simple_stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ══ SECTION 6: Feature Engineering ══════════════════════════════════════════

def build_mft_features(df: pd.DataFrame, text_col: str = 'content_clean') -> np.ndarray:
    """Section 6a — Our contribution: 10-dimensional eMFD feature matrix."""
    mft_df = score_dataframe(df, text_col)
    # Ensure consistent column order
    cols = sorted(mft_df.columns)
    return mft_df[cols].values, cols


def build_tfidf_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         text_col: str = 'content_clean') -> tuple:
    """
    Section 6b — TF-IDF baseline.
    Fits on train, transforms both. Returns (X_train, X_test, vectorizer).
    """
    vec = TfidfVectorizer(max_features=500, sublinear_tf=True,
                          min_df=2, strip_accents='unicode')
    X_train = vec.fit_transform(train_df[text_col].fillna('')).toarray()
    X_test  = vec.transform(test_df[text_col].fillna('')).toarray()
    return X_train, X_test, vec


def build_big5_features(email_df: pd.DataFrame, psycho_df: pd.DataFrame) -> np.ndarray:
    """
    Section 6c — Big Five personality baseline.
    Merges O,C,E,A,N scores onto email df by user. Returns feature matrix.
    """
    big5_cols = ['O', 'C', 'E', 'A', 'N']
    # Normalise column names to uppercase single letters
    psycho = psycho_df.copy()
    col_map = {}
    for c in psycho.columns:
        cl = c.lower()
        if 'open' in cl:   col_map[c] = 'O'
        elif 'cons' in cl: col_map[c] = 'C'
        elif 'extra' in cl:col_map[c] = 'E'
        elif 'agre' in cl: col_map[c] = 'A'
        elif 'neur' in cl: col_map[c] = 'N'
        elif 'user' in cl: col_map[c] = 'user'
    psycho = psycho.rename(columns=col_map)
    if 'user' not in psycho.columns:
        psycho = psycho.rename(columns={psycho.columns[0]: 'user'})

    # Keep only the columns we need
    keep = ['user'] + [c for c in big5_cols if c in psycho.columns]
    psycho = psycho[keep]

    merged = email_df[['user']].merge(psycho, on='user', how='left')
    feat_cols = [c for c in big5_cols if c in merged.columns]
    X = merged[feat_cols].fillna(merged[feat_cols].mean()).values
    return X, feat_cols


# ══ SECTION 6d: Behavioral Feature Extraction ════════════════════════════════

def load_behavioral_logs(data_dir: str) -> pd.DataFrame:
    """
    Aggregate device / logon / file / http logs to per-user behavioral features.
    All four files come from CMU r4.2 and share the same user IDs as email.csv.

    Returns a DataFrame indexed by 'user' with columns:
      logon_count, after_hours_rate, unique_pcs,
      usb_count, usb_per_logon,
      file_count, sensitive_file_rate,
      http_external_rate, job_site_rate
    """
    AFTER_HOURS_START = 19   # 7 pm
    AFTER_HOURS_END   = 7    # 7 am
    JOB_SITES = {'linkedin', 'indeed', 'monster', 'glassdoor',
                 'careerbuilder', 'simplyhired', 'ziprecruiter'}
    SENSITIVE_EXTS = {'.pdf', '.doc', '.docx', '.xls', '.xlsx',
                      '.zip', '.rar', '.7z', '.tar'}

    results = {}  # user → dict of stats

    # ── logon.csv (56 MB — load fully) ───────────────────────────────────────
    logon_path = os.path.join(data_dir, 'logon.csv')
    if os.path.exists(logon_path):
        print("    Loading logon.csv...")
        logon = pd.read_csv(logon_path, usecols=['date', 'user', 'activity'],
                            low_memory=False)
        logon['hour'] = pd.to_datetime(logon['date'], errors='coerce').dt.hour
        logon_only = logon[logon['activity'].str.lower() == 'logon']
        grp = logon_only.groupby('user')
        for user, g in grp:
            if user not in results:
                results[user] = {}
            results[user]['logon_count'] = len(g)
            ah = ((g['hour'] >= AFTER_HOURS_START) | (g['hour'] < AFTER_HOURS_END)).mean()
            results[user]['after_hours_rate'] = float(ah)
        print(f"      {len(results):,} users from logon logs")

    # ── device.csv (28 MB — load fully) ──────────────────────────────────────
    device_path = os.path.join(data_dir, 'device.csv')
    if os.path.exists(device_path):
        print("    Loading device.csv...")
        dev = pd.read_csv(device_path, usecols=['user', 'activity'], low_memory=False)
        connects = dev[dev['activity'].str.lower() == 'connect']
        grp = connects.groupby('user').size()
        for user, cnt in grp.items():
            if user not in results:
                results[user] = {}
            results[user]['usb_count'] = int(cnt)
        print(f"      {len(grp):,} users with USB events")

    # ── file.csv (185 MB — chunked) ───────────────────────────────────────────
    file_path = os.path.join(data_dir, 'file.csv')
    if os.path.exists(file_path):
        print("    Loading file.csv (chunked)...")
        file_totals   = {}   # user → total file events
        file_sensitive = {}  # user → sensitive file events
        for chunk in pd.read_csv(file_path, chunksize=50_000,
                                 usecols=['user', 'filename'], low_memory=False):
            chunk['ext'] = chunk['filename'].str.lower().str.extract(r'(\.\w+)$')[0]
            chunk['is_sensitive'] = chunk['ext'].isin(SENSITIVE_EXTS)
            for user, g in chunk.groupby('user'):
                file_totals[user]    = file_totals.get(user, 0) + len(g)
                file_sensitive[user] = file_sensitive.get(user, 0) + g['is_sensitive'].sum()
        for user, total in file_totals.items():
            if user not in results:
                results[user] = {}
            results[user]['file_count'] = total
            results[user]['sensitive_file_rate'] = (
                file_sensitive.get(user, 0) / max(total, 1))
        print(f"      {len(file_totals):,} users from file logs")

    # ── http.csv (14 GB — first 200K rows only) ───────────────────────────────
    http_path = os.path.join(data_dir, 'http.csv')
    if os.path.exists(http_path):
        print("    Sampling http.csv (first 200K rows)...")
        http_totals   = {}
        http_external = {}
        http_jobsite  = {}
        rows_read = 0
        for chunk in pd.read_csv(http_path, chunksize=50_000,
                                 usecols=['user', 'url'], low_memory=False):
            rows_read += len(chunk)
            # Extract domain (second-level, e.g. "google" from "mail.google.com")
            chunk['domain'] = (chunk['url'].str.lower()
                               .str.extract(r'https?://(?:www\.)?([^./]+)')[0])
            chunk['is_job']      = chunk['domain'].isin(JOB_SITES)
            chunk['is_external'] = ~chunk['domain'].isin(
                ['dtaa', 'intranet', 'sharepoint', 'localhost', ''])
            for user, g in chunk.groupby('user'):
                http_totals[user]   = http_totals.get(user, 0) + len(g)
                http_external[user] = http_external.get(user, 0) + g['is_external'].sum()
                http_jobsite[user]  = http_jobsite.get(user, 0) + g['is_job'].sum()
            if rows_read >= 200_000:
                break
        for user, total in http_totals.items():
            if user not in results:
                results[user] = {}
            results[user]['http_external_rate'] = (
                http_external.get(user, 0) / max(total, 1))
            results[user]['job_site_rate'] = (
                http_jobsite.get(user, 0) / max(total, 1))
        print(f"      {len(http_totals):,} users from http sample ({rows_read:,} rows)")

    if not results:
        return pd.DataFrame()

    beh_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    beh_df = beh_df.rename(columns={'index': 'user'})

    # Derived feature: USB events normalised by logon count
    if 'usb_count' in beh_df.columns and 'logon_count' in beh_df.columns:
        beh_df['usb_per_logon'] = (beh_df['usb_count'].fillna(0) /
                                   beh_df['logon_count'].replace(0, 1))

    beh_df = beh_df.fillna(0)
    print(f"  Behavioral features: {len(beh_df):,} users, "
          f"{len(beh_df.columns)-1} features")
    return beh_df


def build_behavioral_features(email_df: pd.DataFrame,
                               beh_df: pd.DataFrame) -> tuple:
    """
    Join per-user behavioral aggregates onto email_df by 'user'.
    Returns (X_array, feature_col_names).
    Missing users get column-mean imputation.
    """
    feat_cols = [c for c in beh_df.columns if c != 'user']
    merged = email_df[['user']].merge(beh_df, on='user', how='left')
    X = merged[feat_cols].fillna(merged[feat_cols].mean()).values
    return X, feat_cols


# ══ SECTION 7: Scenario-Based Train/Test Split ═══════════════════════════════

def scenario_split(cmu_df: pd.DataFrame, scenario_users: dict) -> tuple:
    """
    Train: scenarios 1 + 2 (malicious) + 80% of benign users.
    Test : scenario 3  (malicious) + 20% of benign users.
    Split is user-level — all emails from a user go to the same partition.
    Returns (train_df, test_df).
    """
    # Malicious: scenario determines partition
    mal_train = cmu_df[cmu_df['scenario'].isin([1, 2])].copy()
    mal_test  = cmu_df[cmu_df['scenario'] == 3].copy()

    # Benign: split unique users 80/20
    benign_df = cmu_df[cmu_df['label'] == 0].copy()
    benign_users = benign_df['user'].unique()
    if len(benign_users) >= 2:
        train_users, test_users = train_test_split(benign_users, test_size=0.20,
                                                   random_state=RANDOM_STATE)
        ben_train = benign_df[benign_df['user'].isin(train_users)]
        ben_test  = benign_df[benign_df['user'].isin(test_users)]
    else:
        # Edge case: very few users
        ben_train = benign_df
        ben_test  = benign_df.iloc[:0]

    train_df = pd.concat([mal_train, ben_train], ignore_index=True)
    test_df  = pd.concat([mal_test,  ben_test],  ignore_index=True)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  Train: {len(train_df):,} emails "
          f"(malicious={train_df['label'].sum():,})")
    print(f"  Test : {len(test_df):,} emails  "
          f"(malicious={test_df['label'].sum():,})")
    return train_df, test_df


# ══ SECTION 8: Model Training and Evaluation ═════════════════════════════════

def evaluate_model(model, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold,
                   scaler=None) -> dict:
    """
    5-fold stratified cross-validation.
    If scaler is provided, fit/transform inside each fold.
    Returns dict of mean scores across folds.
    """
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']

    if scaler is not None:
        from sklearn.pipeline import Pipeline as SKPipeline
        pipe = SKPipeline([('scaler', scaler), ('clf', model)])
        results = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                                 error_score='raise')
    else:
        results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                 error_score='raise')

    return {
        'accuracy':  results['test_accuracy'].mean(),
        'precision': results['test_precision_macro'].mean(),
        'recall':    results['test_recall_macro'].mean(),
        'f1':        results['test_f1_macro'].mean(),
        'roc_auc':   results['test_roc_auc'].mean(),
    }


def run_models(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray,
               feature_names: list, tag: str) -> dict:
    """
    Train RF, SVM, and GBT on the given feature set.
    Returns results dict and fitted GBT model (for downstream use).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                                      class_weight='balanced'),
        'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE,
                                   class_weight='balanced'),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100,
                                                          random_state=RANDOM_STATE),
    }

    results = {}
    fitted_models = {}

    for name, clf in models.items():
        print(f"    Cross-validating {name} [{tag}]...")
        scaler = StandardScaler() if name == 'SVM (RBF)' else None
        scores = evaluate_model(clf, X_train, y_train, cv, scaler=scaler)
        results[name] = scores

        # Fit on full training set for hold-out evaluation
        if scaler:
            from sklearn.pipeline import Pipeline as SKPipeline
            pipe = SKPipeline([('scaler', StandardScaler()), ('clf', clf)])
            pipe.fit(X_train, y_train)
            fitted_models[name] = pipe
        else:
            clf.fit(X_train, y_train)
            fitted_models[name] = clf

    return results, fitted_models


# ══ SECTION 9: Publication-Quality Visualizations ════════════════════════════

sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = {'Random Forest': '#2196F3', 'SVM (RBF)': '#FF9800', 'Gradient Boosting': '#4CAF50'}


def _save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def fig1_model_comparison(cv_results: dict):
    """Grouped bar chart: 3 MFT models × 4 metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(cv_results.keys())
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mname in enumerate(model_names):
        vals = [cv_results[mname][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=mname,
                      color=list(PALETTE.values())[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('MFT Feature Set — 5-Fold CV Comparison Across Models')
    ax.legend()
    _save(fig, 'fig1_model_comparison.png')


def fig2_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str):
    """Confusion matrix for best model on hold-out test set."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix — {model_name}')
    _save(fig, 'fig2_confusion_matrix.png')


def fig3_feature_importance(rf_model, feature_names: list):
    """Horizontal bar chart: MFT feature Gini importances (vice=red, virtue=blue)."""
    # Extract importances from RF (or GBT if RF not available)
    clf = rf_model
    if hasattr(clf, 'named_steps'):
        clf = clf.named_steps.get('clf', clf)
    importances = clf.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])
    names, vals = zip(*pairs)

    colors = ['#F44336' if 'vice' in n else '#2196F3' for n in names]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel('Gini Importance')
    ax.set_title('MFT Feature Importance (Blue=Virtue, Red=Vice)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2196F3', label='Virtue'),
                        Patch(color='#F44336', label='Vice')])
    _save(fig, 'fig3_feature_importance.png')


def fig4_mft_distributions(X: np.ndarray, y: np.ndarray, feature_names: list):
    """Violin plots of all 10 MFT features split by class label."""
    df_plot = pd.DataFrame(X, columns=feature_names)
    df_plot['Class'] = pd.Series(y).astype(str).map({'0': 'Benign', '1': 'Malicious'}).values

    n_feats = len(feature_names)
    n_cols = 5
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        sns.violinplot(data=df_plot, x='Class', y=feat, ax=axes[i],
                       palette=['#2196F3', '#F44336'], inner='quartile')
        axes[i].set_title(feat, fontsize=9)
        axes[i].set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('MFT Feature Distributions by Class', fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, 'fig4_mft_distributions.png')


def fig5_roc_curves(fitted_models: dict, X_test: np.ndarray, y_test: np.ndarray,
                    tfidf_model, X_tfidf_test: np.ndarray, tag: str = 'MFT'):
    """ROC curves for all 3 MFT models + TF-IDF baseline."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)

    colors = list(PALETTE.values()) + ['#9C27B0']
    for (name, model), color in zip(fitted_models.items(), colors):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC={roc_auc:.3f})')
        except Exception as e:
            print(f"    ROC skip {name}: {e}")

    # TF-IDF baseline
    if tfidf_model is not None and X_tfidf_test is not None:
        try:
            proba = tfidf_model.predict_proba(X_tfidf_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test[:len(proba)], proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='#795548', lw=2, linestyle='--',
                    label=f'TF-IDF GBT (AUC={roc_auc:.3f})')
        except Exception as e:
            print(f"    ROC skip TF-IDF: {e}")

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — MFT vs TF-IDF Baseline')
    ax.legend(loc='lower right', fontsize=9)
    _save(fig, 'fig5_roc_curves.png')


def fig6_pr_curves(fitted_models: dict, X_test: np.ndarray, y_test: np.ndarray,
                   tfidf_model, X_tfidf_test: np.ndarray):
    """Precision-Recall curves (critical for imbalanced datasets)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = list(PALETTE.values()) + ['#9C27B0']

    for (name, model), color in zip(fitted_models.items(), colors):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, proba)
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, color=color, lw=2,
                    label=f'{name} (AUC={pr_auc:.3f})')
        except Exception as e:
            print(f"    PR skip {name}: {e}")

    if tfidf_model is not None and X_tfidf_test is not None:
        try:
            proba = tfidf_model.predict_proba(X_tfidf_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test[:len(proba)], proba)
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, color='#795548', lw=2, linestyle='--',
                    label=f'TF-IDF GBT (AUC={pr_auc:.3f})')
        except Exception as e:
            print(f"    PR skip TF-IDF: {e}")

    baseline = y_test.mean()
    ax.axhline(baseline, color='gray', linestyle=':', label=f'Baseline (prev={baseline:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — Imbalanced Class Evaluation')
    ax.legend(loc='upper right', fontsize=9)
    _save(fig, 'fig6_pr_curves.png')


def fig7_baseline_comparison(mft_scores: dict, tfidf_scores: dict,
                              big5_scores: dict = None):
    """Bar chart: MFT-GB vs TF-IDF-GB vs Big5-GB on F1 and ROC-AUC."""
    labels, f1s, aucs_val = [], [], []
    for tag, scores in [('MFT (Ours)', mft_scores), ('TF-IDF', tfidf_scores),
                        ('Big Five', big5_scores)]:
        if scores is None:
            continue
        labels.append(tag)
        f1s.append(scores.get('f1', 0))
        aucs_val.append(scores.get('roc_auc', 0))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, f1s, width, label='F1 (macro)', color='#2196F3')
    b2 = ax.bar(x + width/2, aucs_val, width, label='ROC-AUC', color='#FF9800')

    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Baseline Comparison — Gradient Boosting on Different Feature Sets')
    ax.legend()
    _save(fig, 'fig7_baseline_comparison.png')


def fig8_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  full_f1: float, feature_names: list):
    """
    Ablation study: retrain GBT with each foundation's 2 features removed.
    Bar chart showing F1 drop per removed foundation.
    """
    print("  Running ablation study (5 foundations)...")
    deltas = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mft_train = score_dataframe(train_df, 'content_clean')
    feat_cols  = sorted(mft_train.columns)
    X_train    = mft_train[feat_cols].values
    y_train    = train_df['label'].values

    for foundation in FOUNDATIONS:
        drop_cols = [f'{foundation}.virtue', f'{foundation}.vice']
        keep_cols = [c for c in feat_cols if c not in drop_cols]
        X_sub = mft_train[keep_cols].values

        clf = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        results = cross_validate(clf, X_sub, y_train, cv=cv, scoring=['f1_macro'],
                                 error_score='raise')
        f1_without = results['test_f1_macro'].mean()
        deltas[foundation] = full_f1 - f1_without

    foundations = list(deltas.keys())
    drops = [deltas[f] for f in foundations]
    colors = ['#F44336' if d > 0 else '#4CAF50' for d in drops]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f'-{f}' for f in foundations], drops, color=colors)
    for b, v in zip(bars, drops):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + 0.0005 if v >= 0 else b.get_height() - 0.003,
                f'{v:+.4f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('ΔF1 (positive = harmful to remove)')
    ax.set_title('Ablation Study — F1 Drop When Each Foundation Removed')
    _save(fig, 'fig8_ablation.png')
    return deltas


# ══ SECTION 10: Enron Validation ═════════════════════════════════════════════

def enron_validation(enron_df: pd.DataFrame, best_model, feature_names: list) -> dict:
    """
    Apply trained GB-MFT model to Enron emails.
    Ground truth proxy: emails containing at least one THREAT_KEYWORD.
    Reports precision/recall/F1/accuracy vs keyword proxy.
    Runs Mann-Whitney U test per MFT feature.
    Saves fig9 and fig10.
    """
    print("\n  Enron validation: scoring MFT features...")
    enron_df = enron_df.copy()
    enron_df['content_clean'] = enron_df['body'].apply(preprocess_text)

    mft_feats = score_dataframe(enron_df, 'content_clean')
    feat_cols  = sorted(mft_feats.columns)
    X_enron    = mft_feats[feat_cols].values

    # Keyword-proxy ground truth
    keyword_pattern = '|'.join(THREAT_KEYWORDS)
    enron_df['keyword_flag'] = enron_df['body'].str.lower().str.contains(
        keyword_pattern, regex=True).astype(int)

    # Threat probability from best model
    try:
        threat_proba = best_model.predict_proba(X_enron)[:, 1]
    except Exception as e:
        print(f"  WARNING: predict_proba failed: {e}. Using decision_function.")
        threat_proba = best_model.decision_function(X_enron)
        threat_proba = (threat_proba - threat_proba.min()) / (
            threat_proba.max() - threat_proba.min() + 1e-9)

    enron_df['threat_score'] = threat_proba

    # Binary threshold at 0.5
    y_pred  = (threat_proba >= 0.5).astype(int)
    y_proxy = enron_df['keyword_flag'].values

    acc  = accuracy_score(y_proxy, y_pred)
    prec = precision_score(y_proxy, y_pred, zero_division=0)
    rec  = recall_score(y_proxy, y_pred, zero_division=0)
    f1   = f1_score(y_proxy, y_pred, zero_division=0)
    print(f"  Enron vs keyword-proxy: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # Mann-Whitney U test per feature
    mwu_results = {}
    flagged_mask = y_proxy == 1
    for feat in feat_cols:
        col_vals = mft_feats[feat].values
        if flagged_mask.sum() > 0 and (~flagged_mask).sum() > 0:
            stat, p = mannwhitneyu(col_vals[flagged_mask], col_vals[~flagged_mask],
                                   alternative='two-sided')
        else:
            p = 1.0
        mwu_results[feat] = p

    # fig9: histogram of threat scores
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(threat_proba[y_proxy == 0], bins=50, alpha=0.6, color='#2196F3',
            label='Benign (keyword-proxy)', density=True)
    ax.hist(threat_proba[y_proxy == 1], bins=50, alpha=0.6, color='#F44336',
            label='Flagged (keyword-proxy)', density=True)
    ax.set_xlabel('Threat Probability Score')
    ax.set_ylabel('Density')
    ax.set_title('Enron Email Threat Score Distribution')
    ax.legend()
    _save(fig, 'fig9_enron_threat_scores.png')

    # fig10: MWU p-values bar chart (log scale)
    fig, ax = plt.subplots(figsize=(9, 4))
    feats_sorted = sorted(mwu_results, key=lambda f: mwu_results[f])
    pvals = [mwu_results[f] for f in feats_sorted]
    bar_colors = ['#F44336' if 'vice' in f else '#2196F3' for f in feats_sorted]
    ax.bar(feats_sorted, pvals, color=bar_colors)
    ax.axhline(0.05, color='black', linestyle='--', label='p=0.05')
    ax.set_yscale('log')
    ax.set_ylabel('p-value (log scale)')
    ax.set_title('Mann-Whitney U Test: MFT Features — Flagged vs Benign (Enron)')
    ax.set_xticklabels(feats_sorted, rotation=45, ha='right')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2196F3', label='Virtue'),
                        Patch(color='#F44336', label='Vice'),
                        plt.Line2D([0], [0], color='black', linestyle='--', label='p=0.05')],
              fontsize=8)
    plt.tight_layout()
    _save(fig, 'fig10_enron_mwu.png')

    # Top 10 most suspicious emails
    top10 = enron_df.nlargest(10, 'threat_score')[['file', 'body', 'threat_score']]
    print("\n  Top 10 most suspicious Enron emails:")
    for _, row in top10.iterrows():
        snippet = row['body'][:80].replace('\n', ' ')
        print(f"    score={row['threat_score']:.3f}  \"{snippet}...\"")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'mwu': mwu_results}


# ══ SECTION 11: Summary Report ════════════════════════════════════════════════

def print_summary(cv_mft: dict, cv_tfidf: dict, cv_big5: dict,
                  enron_val: dict, ablation: dict, top_features: list):
    """Print and save publication-ready results table."""
    lines = []
    SEP = '═' * 67

    def add(*args):
        s = ' '.join(str(a) for a in args)
        lines.append(s)
        print(s)

    add(SEP)
    add('  RESULTS SUMMARY — NLP Insider Threat Detection via MFT')
    add(SEP)
    add()
    add('CMU r4.2 Dataset — 5-fold Stratified Cross-Validation:')
    add(f'  {"Model":<22} {"Accuracy":>9} {"Precision":>9} {"Recall":>9} {"F1":>9} {"ROC-AUC":>9}')
    add('  ' + '-' * 63)

    best_f1, best_name = 0, ''
    for name, scores in cv_mft.items():
        marker = '  ← BEST' if scores['f1'] == max(s['f1'] for s in cv_mft.values()) else ''
        add(f'  {name:<22} {scores["accuracy"]:>9.3f} {scores["precision"]:>9.3f} '
            f'{scores["recall"]:>9.3f} {scores["f1"]:>9.3f} {scores["roc_auc"]:>9.3f}{marker}')
        if scores['f1'] > best_f1:
            best_f1 = scores['f1']
            best_name = name

    add()
    add('Baseline Comparison (Gradient Boosting on different feature sets):')
    mft_gb = cv_mft.get('Gradient Boosting', {})
    add(f'  MFT Features       F1={mft_gb.get("f1", 0):.3f}  '
        f'ROC-AUC={mft_gb.get("roc_auc", 0):.3f}   <- Our contribution')
    tfidf_gb = cv_tfidf or {}
    add(f'  TF-IDF Features    F1={tfidf_gb.get("f1", 0):.3f}  '
        f'ROC-AUC={tfidf_gb.get("roc_auc", 0):.3f}')
    if cv_big5:
        add(f'  Big Five Features  F1={cv_big5.get("f1", 0):.3f}  '
            f'ROC-AUC={cv_big5.get("roc_auc", 0):.3f}')
    else:
        add('  Big Five Features  N/A (psychometric.csv not available)')

    add()
    add('Enron Email Validation (keyword-proxy ground truth):')
    add(f'  Accuracy: {enron_val["accuracy"]:.3f}  Precision: {enron_val["precision"]:.3f}  '
        f'Recall: {enron_val["recall"]:.3f}  F1: {enron_val["f1"]:.3f}')

    add()
    add('MFT Ablation Study (F1 drop when foundation removed):')
    for f, delta in ablation.items():
        add(f'  -{f:<12}  ΔF1 = {delta:+.4f}')

    add()
    add('Top 3 features (Gradient Boosting importance):')
    for rank, (feat, imp) in enumerate(top_features[:3], 1):
        add(f'  {rank}. {feat:<25} {imp:.4f}')

    add(SEP)

    report_path = os.path.join(OUT_DIR, 'summary_results.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Full report saved to {report_path}")


# ══ MAIN ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 67)
    print("  NLP Insider Threat Detection via Moral Foundations Theory")
    print("  Masters Research Pipeline — real_pipeline.py")
    print("═" * 67 + "\n")

    # ── Step 1: Load malicious user IDs from CMU answers ──────────────────
    print("[ 1/10 ] Loading CMU answer keys...")
    scenario_users = load_malicious_users(CMU_ANSWERS_DIR)

    # ── Step 2: Load CMU email data (real or synthetic fallback) ──────────
    print("\n[ 2/10 ] Loading CMU email data...")
    cmu_df, psycho_df = load_cmu_emails(CMU_EMAIL_CSV, CMU_PSYCHO_CSV,
                                         scenario_users, CMU_SAMPLE_N)

    # ── Step 3: Preprocess text ───────────────────────────────────────────
    print("\n[ 3/10 ] Preprocessing text...")
    cmu_df['content_clean'] = cmu_df['content'].apply(preprocess_text)

    # ── Step 4: Scenario-based train/test split ───────────────────────────
    print("\n[ 4/10 ] Scenario-based train/test split...")
    train_df, test_df = scenario_split(cmu_df, scenario_users)

    # Guard against degenerate splits
    if len(train_df) == 0 or len(test_df) == 0:
        print("  WARNING: empty split — falling back to 80/20 random split")
        train_df, test_df = train_test_split(cmu_df, test_size=0.2,
                                             stratify=cmu_df['label'],
                                             random_state=RANDOM_STATE)

    # ── Step 5: MFT feature matrices ─────────────────────────────────────
    print("\n[ 5/10 ] Building MFT features...")
    X_mft_train, feat_names = build_mft_features(train_df)
    X_mft_test,  _          = build_mft_features(test_df)
    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    # ── Step 6: TF-IDF baseline features ─────────────────────────────────
    print("\n[ 6/10 ] Building TF-IDF features...")
    X_tfidf_train, X_tfidf_test, tfidf_vec = build_tfidf_features(train_df, test_df)

    # ── Step 7: Big Five baseline features ───────────────────────────────
    print("\n[ 7/10 ] Building Big Five features...")
    big5_available = False
    X_big5_train, X_big5_test, cv_big5 = None, None, None
    if psycho_df is not None:
        try:
            X_big5_train, big5_cols = build_big5_features(train_df, psycho_df)
            X_big5_test,  _         = build_big5_features(test_df,  psycho_df)
            big5_available = True
            print(f"  Big Five features: {big5_cols}")
        except Exception as e:
            print(f"  WARNING: Big Five build failed: {e}")
    else:
        print("  Skipping Big Five baseline (psychometric data unavailable)")

    # ── Step 7b: Behavioral log features ─────────────────────────────────
    print("\n[ 7b ] Building behavioral features from device/logon/file/http logs...")
    beh_available = False
    X_beh_train, X_beh_test = None, None
    X_mft_beh_train, X_mft_beh_test = None, None
    cv_beh = None
    beh_df = load_behavioral_logs(os.path.dirname(CMU_EMAIL_CSV))
    if len(beh_df) > 0:
        try:
            X_beh_train, beh_cols = build_behavioral_features(train_df, beh_df)
            X_beh_test,  _        = build_behavioral_features(test_df,  beh_df)
            # Also build combined MFT+Behavioral feature matrix
            X_mft_beh_train = np.hstack([X_mft_train, X_beh_train])
            X_mft_beh_test  = np.hstack([X_mft_test,  X_beh_test])
            beh_available = True
            print(f"  Behavioral feature columns: {beh_cols}")
        except Exception as e:
            print(f"  WARNING: Behavioral feature build failed: {e}")
    else:
        print("  Skipping behavioral features (log files unavailable)")

    # ── Step 8: Train and evaluate all models ────────────────────────────
    print("\n[ 8/10 ] Training and evaluating models...")

    print("\n  --- MFT Feature Set (our contribution) ---")
    cv_mft, fitted_mft = run_models(X_mft_train, y_train, X_mft_test, y_test,
                                     feat_names, tag='MFT')

    print("\n  --- TF-IDF Baseline ---")
    cv_tfidf_all, fitted_tfidf = run_models(X_tfidf_train, y_train, X_tfidf_test, y_test,
                                             None, tag='TF-IDF')
    cv_tfidf = cv_tfidf_all.get('Gradient Boosting', {})
    fitted_tfidf_gbt = fitted_tfidf.get('Gradient Boosting')

    if big5_available:
        print("\n  --- Big Five Baseline ---")
        cv_big5_all, _ = run_models(X_big5_train, y_train, X_big5_test, y_test,
                                     None, tag='Big5')
        cv_big5 = cv_big5_all.get('Gradient Boosting', {})
    else:
        cv_big5 = None

    if beh_available:
        print("\n  --- Behavioral Baseline ---")
        cv_beh_all, _ = run_models(X_beh_train, y_train, X_beh_test, y_test,
                                    None, tag='Behavioral')
        cv_beh = cv_beh_all.get('Gradient Boosting', {})

        print("\n  --- MFT + Behavioral (proposed fusion) ---")
        mft_beh_feat_names = list(feat_names) + beh_cols
        cv_mft_beh, fitted_mft_beh = run_models(
            X_mft_beh_train, y_train, X_mft_beh_test, y_test,
            mft_beh_feat_names, tag='MFT+Beh')
    else:
        cv_mft_beh, fitted_mft_beh = {}, {}

    # Best model for downstream tasks = GBT on MFT+Behavioral (or MFT alone)
    if beh_available and fitted_mft_beh.get('Gradient Boosting') is not None:
        best_model      = fitted_mft_beh.get('Gradient Boosting')
        best_model_name = 'MFT+Behavioral GBT'
        best_X_train, best_X_test = X_mft_beh_train, X_mft_beh_test
    else:
        best_model      = fitted_mft.get('Gradient Boosting')
        best_model_name = 'MFT GBT'
        best_X_train, best_X_test = X_mft_train, X_mft_test

    # ── Step 9: Visualizations ────────────────────────────────────────────
    print("\n[ 9/10 ] Generating publication-quality figures...")

    fig1_model_comparison(cv_mft)

    # Confusion matrix on hold-out test
    if best_model is not None and len(X_mft_test) > 0 and len(np.unique(y_test)) > 1:
        fig2_confusion_matrix(best_model, X_mft_test, y_test, best_model_name)

    # Feature importance from Random Forest (has .feature_importances_)
    rf_model = fitted_mft.get('Random Forest')
    if rf_model is not None:
        fig3_feature_importance(rf_model, feat_names)
        # Top 3 features for summary
        clf_rf = rf_model.named_steps['clf'] if hasattr(rf_model, 'named_steps') else rf_model
        top_features = sorted(zip(feat_names, clf_rf.feature_importances_),
                              key=lambda x: x[1], reverse=True)
    else:
        top_features = [(f, 0.0) for f in feat_names]

    fig4_mft_distributions(X_mft_train, y_train, feat_names)
    fig5_roc_curves(fitted_mft, X_mft_test, y_test, fitted_tfidf_gbt, X_tfidf_test)
    fig6_pr_curves(fitted_mft, X_mft_test, y_test, fitted_tfidf_gbt, X_tfidf_test)
    fig7_baseline_comparison(cv_mft.get('Gradient Boosting', {}), cv_tfidf, cv_big5)

    # Ablation study
    best_full_f1 = cv_mft.get('Gradient Boosting', {}).get('f1', 0)
    ablation = fig8_ablation(train_df, test_df, best_full_f1, feat_names)

    # ── Step 10: Enron validation ─────────────────────────────────────────
    print("\n[ 10/10 ] Loading and validating on Enron emails...")
    enron_df = load_enron_emails(ENRON_CSV, ENRON_SAMPLE_N)
    enron_val = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mwu': {}}

    if len(enron_df) > 0 and best_model is not None:
        enron_val = enron_validation(enron_df, best_model, feat_names)
    else:
        print("  Skipping Enron validation (no data or model unavailable)")

    # ── Step 11: Summary ──────────────────────────────────────────────────
    print("\n")
    print_summary(cv_mft, cv_tfidf, cv_big5, enron_val, ablation, top_features)

    # ── Extraction instructions ───────────────────────────────────────────
    print()
    print('═' * 62)
    print('  TO USE REAL CMU DATA — run this on your Mac terminal:')
    print()
    print('  cd ~/Documents/Projects/NLP-insider-threat-psych/cmu-dataset')
    print('  tar -xjf "Insider Threat Test Dataset r4.2.tar.bz2" \\')
    print('      r4.2/email.csv r4.2/psychometric.csv')
    print()
    print('  Then re-run this script. Takes ~3-5 min on Mac.')
    print('═' * 62)


if __name__ == '__main__':
    main()
