import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

# ==========================================================
# src/model.py (V9 - F_0.5 Aligned + Faster Training)
# ==========================================================
#
# What changed vs V8 and why:
#
# 1. EARLY STOPPING NOW WATCHES F_0.5, NOT AUC
#    V8 used metric='auc' for early stopping, but you're being scored
#    on F_0.5. AUC and F_0.5 are not the same objective -- AUC treats
#    every point on the ROC curve as equally important, while F_0.5
#    cares specifically about the precision-heavy region of the
#    precision/recall trade-off. On a heavily imbalanced entity-
#    resolution dataset (which this almost certainly is -- most
#    candidate pairs from blocking are negatives), AUC can keep
#    "improving" for rounds that are actively hurting your precision
#    at the threshold you'll actually deploy at. Early stopping on the
#    wrong metric means you're picking the wrong iteration count.
#    -> metric is disabled ('None') and replaced with a custom feval
#       that computes the BEST achievable F_0.5 across all thresholds
#       each round (via sklearn's precision_recall_curve, a single
#       O(n log n) pass -- cheap even at large validation-set sizes).
#       Early stopping now tracks that number directly.
#
# 2. FASTER HISTOGRAM CONSTRUCTION (max_bin, force_row_wise)
#    max_bin=255 is LightGBM's default and is overkill for ~20
#    similarity-score-style features that don't need that much
#    resolution. Dropping to 63 bins roughly halves histogram-build
#    cost with negligible accuracy impact for this kind of feature set.
#    force_row_wise=True skips LightGBM's runtime auto-detection of
#    row-wise vs. col-wise histogram building, which itself costs time
#    on large datasets and is essentially always won by row-wise for a
#    "many rows, ~20 columns" shape like this one.
#
# 3. EXPLICIT best_iteration ON PREDICT
#    predict_proba now explicitly passes num_iteration=best_iteration
#    to Booster.predict. Relying on LightGBM's implicit "uses
#    best_iteration automatically" behavior has been version-dependent
#    in the past; being explicit removes that ambiguity, especially
#    since save/load round-trips through a Booster that may not carry
#    the same implicit state.
#
# 4. HARD-NEGATIVE SUBSAMPLING HELPER (subsample_negatives)
#    This is very likely your biggest actual training-time lever and
#    it happens before fit() is ever called: blocking + all-candidates
#    scoring produces a dataset that is overwhelmingly negative pairs
#    (most S1 records are compared against several candidates, only
#    one of which -- if any -- is a true match). Training on every
#    single one of those negatives is mostly wasted compute on "easy"
#    negatives the model already separates trivially, and it's *also*
#    a precision risk in the other direction -- the useful negatives
#    for an F_0.5-tuned model are the hard, near-miss ones (high name
#    similarity, different entity), not the easy ones (totally
#    different names). subsample_negatives keeps every positive, keeps
#    the hardest negatives by a ranking score you already have (e.g.
#    name_token_set), and drops easy negatives past a configurable
#    ratio. This directly shrinks the training set (faster) while
#    concentrating training signal on the pairs that actually
#    determine your precision/recall boundary.
#
# 5. find_best_threshold
#    predict_proba only gives you probabilities; F_0.5 is computed on
#    a binary decision. This adds a reusable, explicit threshold
#    search (same math as the training feval) so the threshold used
#    to produce matching_results.tsv is chosen the same way you're
#    being evaluated, rather than an assumed 0.5 cutoff.
#
# Left alone on purpose: scale_pos_weight=1.0. Your instinct to leave
# this at 1.0 and let threshold search do the work is correct for an
# F_0.5 target -- reweighting the loss function distorts the
# probability calibration that a principled threshold search depends
# on. Artificially up-weighting positives (the usual move for
# imbalanced binary classification) pushes the model toward higher
# recall, which is the wrong direction when precision is weighted 2x.


class EntityResolutionModel:
    def __init__(self, params=None, fast_mode: bool = False):
        """
        fast_mode=True switches boosting_type to 'goss' (Gradient-based
        One-Side Sampling), which keeps all high-gradient ("hard") rows
        and samples the easy ones internally -- a real speedup on large
        datasets. GOSS is incompatible with row bagging, so
        bagging_fraction/bagging_freq are dropped automatically when
        fast_mode is on; feature_fraction subsampling still applies.
        Try this only after subsample_negatives if training is still
        too slow.
        """
        self.params = {
            'objective': 'binary',
            'metric': 'None',            # replaced by the F_0.5 feval below
            'learning_rate': 0.05,

            'num_leaves': 127,
            'max_depth': -1,
            'min_data_in_leaf': 100,     # PREVENTS MEMORIZING NOISY FALSE POSITIVES

            'max_bin': 63,               # down from 255: cheaper histograms, ~same accuracy here
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,

            'lambda_l1': 1.0,            # High L1: Drops useless features completely
            'lambda_l2': 5.0,            # MASSIVE L2: Forces model to lower probabilities unless 99% confident

            'scale_pos_weight': 1.0,     # Keep at 1.0; threshold search handles the precision/recall trade-off

            'force_row_wise': True,      # skip auto-detection overhead; correct choice for many-rows/~20-cols
            'seed': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        if fast_mode:
            self.params['boosting_type'] = 'goss'
            self.params.pop('bagging_fraction', None)
            self.params.pop('bagging_freq', None)

        if params:
            self.params.update(params)

        self.model = None
        self.best_threshold = 0.5  # overwritten by find_best_threshold() once you run it

    @staticmethod
    def _f05_eval(preds, dtrain):
        """
        Custom LightGBM feval: reports the BEST F_0.5 achievable across all
        thresholds on this validation fold, each round. One sorted pass via
        precision_recall_curve -- cheap relative to a full boosting round.
        """
        y_true = dtrain.get_label()
        precision, recall, _ = precision_recall_curve(y_true, preds)
        beta = 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        f_beta = np.nan_to_num(f_beta)
        return 'f0.5_best_thresh', float(f_beta.max()), True  # True = higher is better

    @staticmethod
    def subsample_negatives(X, y, ranking_score, neg_pos_ratio: int = 20, random_state: int = 42):
        """
        Keeps every positive row and, per the given ranking_score (e.g. your
        name_token_set or a blended similarity column -- anything where
        higher = more suspicious/near-miss), keeps only the hardest
        `neg_pos_ratio` negatives per positive, dropping the rest. Cuts
        training set size (and training time) while concentrating negatives
        on the near-miss pairs that actually shape the precision boundary.

        X, y, ranking_score must be index-aligned (same row order/length).
        Returns (X_sub, y_sub) with the original row order preserved among
        kept rows.
        """
        y = np.asarray(y)
        ranking_score = np.asarray(ranking_score)
        n_pos = int(y.sum())
        n_neg_keep = max(1, n_pos * neg_pos_ratio)

        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)

        if len(neg_idx) > n_neg_keep:
            # hardest negatives = highest ranking_score among the negatives
            hardest_order = np.argsort(-ranking_score[neg_idx])
            neg_idx = neg_idx[hardest_order[:n_neg_keep]]

        keep_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
        X_sub = X.iloc[keep_idx] if hasattr(X, 'iloc') else X[keep_idx]
        return X_sub, y[keep_idx]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fits the model using LightGBM's highly optimized Dataset."""
        dtrain = lgb.Dataset(X_train, label=y_train)

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            print("    [LightGBM Training Started] Watch F_0.5 improve below:")
            self.model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dtrain, dval],
                valid_names=['train', 'valid'],
                feval=self._f05_eval,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=40),
                    lgb.log_evaluation(25),
                ],
            )
        else:
            self.model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=500,
            )

    def predict_proba(self, X):
        """Returns probability of class 1, using the early-stopped best iteration explicitly."""
        best_iter = getattr(self.model, 'best_iteration', None) or None
        return self.model.predict(X, num_iteration=best_iter)

    def find_best_threshold(self, X_val, y_val) -> float:
        """
        Scans thresholds on a validation set and stores/returns the one
        that maximizes F_0.5 -- use this to pick the cutoff applied when
        producing matching_results.tsv, instead of an assumed 0.5.
        """
        preds = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, preds)
        beta = 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        f_beta = np.nan_to_num(f_beta)
        # precision_recall_curve returns one more precision/recall point than thresholds
        best_idx = int(np.argmax(f_beta[:-1])) if len(thresholds) else 0
        self.best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
        return self.best_threshold

    def save(self, filepath: str):
        """Saves model to standard txt format. Auto-swaps .json to .txt to protect pipeline."""
        filepath = filepath.replace('.json', '.txt')
        self.model.save_model(filepath)

    def load(self, filepath: str):
        """Loads model from txt format."""
        filepath = filepath.replace('.json', '.txt')
        self.model = lgb.Booster(model_file=filepath)