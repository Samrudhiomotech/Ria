import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

try:
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# Keras compatibility layer for models saved with a newer Keras serializer.
# Some saved Dense-layer configs contain quantization_config=None, which older
# Dense deserializers reject even though it has no effect on inference.
if TF_AVAILABLE:
    class CompatibleDense(keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            config = config.copy()
            config.pop("quantization_config", None)
            return super().from_config(config)


# ==================================================================================
# CONFIG
# ==================================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "model": os.path.join(MODEL_DIR, "fusion_model.keras"),
    "genomic_scaler": os.path.join(MODEL_DIR, "genomic_scaler.joblib"),
    "mutation_scaler": os.path.join(MODEL_DIR, "mutation_scaler.joblib"),
    "clinical_preprocessor": os.path.join(MODEL_DIR, "clinical_preprocessor.joblib"),
}

FALLBACK_GENOMIC_FEATURES = ["gene_1", "gene_2", "gene_3", "gene_4", "gene_5"]
FALLBACK_CLINICAL_FEATURES = [
    "age_at_diagnosis", "tumor_grade", "tumor_size_mm",
    "er_status", "pr_status", "her2_status",
    "lymph_nodes_positive", "menopausal_status",
]
FALLBACK_MUTATION_FEATURES = ["mut_feature_1", "mut_feature_2", "mut_feature_3"]

THRESHOLD_DEFAULT = 0.5

# Dropdown categories for clinical fields recognized by keyword in their name.
CLINICAL_SELECT_OPTIONS = {
    "status": ["Positive", "Negative", "Equivocal"],
    "grade": ["1", "2", "3"],
    "menopaus": ["Pre-menopausal", "Post-menopausal"],
}

# Generic Low / Normal / High labels shown in the UI for genomic and mutation
# features. The *raw numeric value* behind each label is derived per-feature
# from the fitted scaler's own statistics wherever possible (see
# derive_bucket_values below) rather than a single flat -1/0/1 pair used for
# every feature regardless of its real scale -- that mismatch between a
# placeholder -1..1 range and a scaler fit on real expression magnitudes is
# the most common reason this kind of form produces meaningless predictions.
BUCKET_OPTIONS = ["Low", "Normal", "High"]
DEFAULT_BUCKET_VALUES = {"Low": -1.0, "Normal": 0.0, "High": 1.0}

# Reasonable real-world ranges for common numeric clinical fields, keyed by
# a substring match against the (lowercased) feature name. Used so fields
# like age or tumor size get an actual number entry with a sane default
# instead of being forced through the -1/0/1 Low/Normal/High bucket, which
# is only appropriate for genuinely categorical or expression-style data.
CLINICAL_NUMERIC_RANGES = {
    "age": (18.0, 100.0, 55.0),
    "size": (1.0, 200.0, 25.0),
    "lymph_node": (0.0, 50.0, 0.0),
    "mutation_count": (0.0, 60.0, 4.0),
    "nottingham": (2.0, 8.0, 4.5),
}
CLINICAL_NUMERIC_DEFAULT_RANGE = (0.0, 10.0, 5.0)

# Fields that represent whole-number counts, shown as plain integers
# (e.g. "4" not "4.00") rather than decimals, which read oddly for a count
# of lymph nodes or mutations.
CLINICAL_INTEGER_KEYWORDS = ["lymph_node", "mutation_count", "nodes_positive"]

# ----------------------------------------------------------------------------------
# Known category options for clinical fields that are genuinely categorical
# (text labels), not numbers -- matched against a fitted preprocessor's real
# encoder first (most accurate), and used as a labeled fallback only when no
# fitted encoder is available to read the true training categories from.
# Field names match the common METABRIC breast-cancer clinical dataset.
# ----------------------------------------------------------------------------------
CLINICAL_CATEGORY_OPTIONS = {
    "cancer_type_detailed": [
        "Breast Invasive Ductal Carcinoma", "Breast Invasive Lobular Carcinoma",
        "Breast Mixed Ductal and Lobular Carcinoma",
        "Breast Invasive Mixed Mucinous Carcinoma", "Metaplastic Breast Cancer",
        "Invasive Breast Carcinoma",
    ],
    "laterality": ["Left", "Right"],
    "histologic": ["Ductal/NST", "Lobular", "Mixed", "Tubular/Cribriform",
                   "Mucinous", "Medullary", "Metaplastic", "Other"],
    "pam50": ["LumA", "LumB", "Her2", "Basal", "Normal", "claudin-low", "NC"],
    "integrative_cluster": ["1", "2", "3", "4ER+", "4ER-", "5", "6", "7", "8", "9", "10"],
    "breast_surgery": ["Mastectomy", "Breast Conserving"],
    "cellularity": ["Low", "Moderate", "High"],
}

# ----------------------------------------------------------------------------------
# FALLBACK NUMERIC ENCODING for clinical categorical dropdowns.
# Only used when clinical_preprocessor.joblib is missing/unavailable, so the
# raw DataFrame passed to the model is guaranteed numeric (no "Positive"
# strings reaching model.predict, which is what caused "Invalid dtype: object").
# NOTE: this is an approximation. It will NOT match the exact encoding the
# model was actually trained on (e.g. one-hot vs ordinal, category order).
# Predictions made with this fallback are for exercising the UI only -- get
# the real fitted clinical_preprocessor.joblib for trustworthy predictions.
# ----------------------------------------------------------------------------------
FALLBACK_STATUS_ENCODE = {"Positive": 1.0, "Negative": 0.0, "Equivocal": 0.5}
FALLBACK_MENOPAUSE_ENCODE = {"Pre-menopausal": 0.0, "Post-menopausal": 1.0}

# ----------------------------------------------------------------------------------
# LABEL FORMATTING
# ----------------------------------------------------------------------------------
_ACRONYMS = {"er": "ER", "pr": "PR", "her2": "HER2", "pam50": "PAM50", "id": "ID"}


def prettify_label(feat):
    tokens = feat.replace("-", " ").split("_")
    words = []
    for t in tokens:
        if not t:
            continue
        low = t.lower()
        if low in _ACRONYMS:
            words.append(_ACRONYMS[low])
        elif t == "+":
            words.append("+")
        else:
            words.append(t.capitalize())
    return " ".join(words) if words else feat


def prettify_gene(feat):
    return feat.upper()


def prettify_mutation(feat):
    if feat.lower().endswith("_mut"):
        return f"{feat[:-4].upper()} (mutation)"
    return prettify_label(feat)


CLINICAL_GROUPS = [
    ("Demographics & History", ["age", "menopaus"]),
    ("Tumor Characteristics", ["tumor", "grade", "size", "nottingham",
                                "cellularity", "laterality", "histologic",
                                "cancer_type"]),
    ("Receptor & Molecular Subtype", ["er_status", "pr_status", "her2",
                                       "pam50", "integrative_cluster",
                                       "subtype"]),
    ("Nodes, Surgery & Treatment History", ["lymph_node", "surgery",
                                             "mutation_count"]),
]


def assign_clinical_group(feat):
    key_lower = feat.lower()
    for group_name, keywords in CLINICAL_GROUPS:
        if any(kw in key_lower for kw in keywords):
            return group_name
    return "Other Clinical Factors"


def get_feature_names(transformer, fallback):
    if transformer is None:
        return list(fallback), False
    names = getattr(transformer, "feature_names_in_", None)
    if names is not None:
        return list(names), True
    return list(fallback), False


def patch_sklearn_compat(obj, _seen=None):
    """Best-effort fix for objects pickled with a different scikit-learn
    version than what's installed now. The real fix is matching
    scikit-learn versions between training and deployment; this patch
    keeps the app usable in the meantime."""
    if not SKLEARN_AVAILABLE or obj is None:
        return obj
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return obj
    _seen.add(id(obj))

    if isinstance(obj, SimpleImputer) and not hasattr(obj, "_fill_dtype"):
        stats = getattr(obj, "statistics_", None)
        try:
            obj._fill_dtype = stats.dtype if stats is not None else np.float64
        except Exception:
            obj._fill_dtype = np.float64

    for attr_name in ("steps", "transformers", "transformers_", "transformer_list"):
        container = getattr(obj, attr_name, None)
        if container:
            for item in container:
                if isinstance(item, tuple):
                    for sub in item:
                        if hasattr(sub, "__dict__"):
                            patch_sklearn_compat(sub, _seen)
                elif hasattr(item, "__dict__"):
                    patch_sklearn_compat(item, _seen)

    for attr_name in ("named_steps", "named_transformers_"):
        mapping = getattr(obj, attr_name, None)
        if mapping:
            try:
                for sub in mapping.values():
                    patch_sklearn_compat(sub, _seen)
            except Exception:
                pass

    return obj


def ensure_sklearn_unpickle_compat():
    """Register a shim for sklearn._column_transformer._RemainderColsList
    so a ColumnTransformer pickled with one scikit-learn version can still
    be unpickled with another (see scikit-learn issue #32090). Best-effort
    only -- matching versions between training and deployment is the
    reliable fix."""
    if not SKLEARN_AVAILABLE:
        return
    try:
        from sklearn.compose import _column_transformer as _ct_mod
        if not hasattr(_ct_mod, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            _ct_mod._RemainderColsList = _RemainderColsList
    except Exception:
        pass


def match_category_keyword(feat):
    """Return the CLINICAL_CATEGORY_OPTIONS entry whose keyword matches this
    feature name, or None."""
    key_lower = feat.lower()
    for keyword, options in CLINICAL_CATEGORY_OPTIONS.items():
        if keyword in key_lower:
            return options
    return None


def encode_clinical_fallback(clinical_input):
    encoded = {}
    for feat, val in clinical_input.items():
        if isinstance(val, str):
            key_lower = feat.lower()
            if "status" in key_lower and val in FALLBACK_STATUS_ENCODE:
                encoded[feat] = FALLBACK_STATUS_ENCODE[val]
            elif "menopaus" in key_lower and val in FALLBACK_MENOPAUSE_ENCODE:
                encoded[feat] = FALLBACK_MENOPAUSE_ENCODE[val]
            else:
                cat_options = match_category_keyword(feat)
                if cat_options and val in cat_options:
                    # Simple ordinal index -- an approximation only. The real
                    # trained encoding (e.g. one-hot) is used automatically
                    # whenever clinical_preprocessor.joblib loads correctly.
                    encoded[feat] = float(cat_options.index(val))
                else:
                    try:
                        encoded[feat] = float(val)
                    except (TypeError, ValueError):
                        encoded[feat] = 0.0
        else:
            encoded[feat] = float(val)
    return encoded


# ==================================================================================
# CORE FIX #1 -- per-feature bucket values instead of a flat -1/0/1
# A gene's real expression values might range from say 4 to 14 (log scale);
# tumor size in mm might range 5 to 150. Feeding a flat -1/0/1 into a
# scaler.transform() that was fit on those real magnitudes produces z-scores
# far outside anything the model saw in training, which is a common cause
# of predictions that look arbitrary or barely move with the inputs. This
# derives Low/Normal/High raw values from the fitted scaler's own mean_/
# scale_ (StandardScaler) or data_min_/data_max_ (MinMaxScaler) so the
# bucket the user picks lands in a range the model actually recognizes.
# ==================================================================================
def derive_bucket_values(features, scaler):
    result = {}
    raw_names = getattr(scaler, "feature_names_in_", None) if scaler is not None else None
    names = list(raw_names) if raw_names is not None else []
    for feat in features:
        vals = None
        if scaler is not None and feat in names:
            idx = names.index(feat)
            try:
                if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    mean = float(scaler.mean_[idx])
                    scale = float(scaler.scale_[idx]) or 1.0
                    vals = {"Low": mean - scale, "Normal": mean, "High": mean + scale}
                elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                    lo = float(scaler.data_min_[idx])
                    hi = float(scaler.data_max_[idx])
                    vals = {"Low": lo, "Normal": (lo + hi) / 2.0, "High": hi}
                elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                    center = float(scaler.center_[idx])
                    scale = float(scaler.scale_[idx]) or 1.0
                    vals = {"Low": center - scale, "Normal": center, "High": center + scale}
            except Exception:
                vals = None
        result[feat] = vals if vals is not None else dict(DEFAULT_BUCKET_VALUES)
    return result


# ==================================================================================
# CORE FIX #2 -- numeric clinical fields get real number inputs, not buckets
# ==================================================================================
def guess_numeric_range(feat, preprocessor):
    """Try to pull a real min/max/default for a numeric clinical field from
    the fitted preprocessor's numeric-scaling step; fall back to a
    keyword-matched sane range, then a generic 0-10 range."""
    key_lower = feat.lower()
    if preprocessor is not None:
        for attr in ("named_transformers_",):
            mapping = getattr(preprocessor, attr, None)
            if not mapping:
                continue
            for sub in mapping.values():
                raw_names = getattr(sub, "feature_names_in_", None)
                names = list(raw_names) if raw_names is not None else []
                if feat not in names:
                    continue
                idx = names.index(feat)
                try:
                    if hasattr(sub, "data_min_") and hasattr(sub, "data_max_"):
                        return float(sub.data_min_[idx]), float(sub.data_max_[idx]), \
                            float((sub.data_min_[idx] + sub.data_max_[idx]) / 2)
                    if hasattr(sub, "mean_") and hasattr(sub, "scale_"):
                        mean, scale = float(sub.mean_[idx]), float(sub.scale_[idx])
                        return mean - 3 * scale, mean + 3 * scale, mean
                except Exception:
                    pass
    for kw, rng in CLINICAL_NUMERIC_RANGES.items():
        if kw in key_lower:
            return rng
    return CLINICAL_NUMERIC_DEFAULT_RANGE


def get_fitted_categories(feat, preprocessor):
    """Best-effort lookup of the exact category labels a fitted
    OneHotEncoder/OrdinalEncoder learned for `feat`, by walking a
    ColumnTransformer's sub-transformers (and any Pipeline steps inside
    them). Returns None if nothing usable is found, in which case callers
    fall back to a keyword-matched category list or a numeric input."""
    if preprocessor is None:
        return None
    try:
        candidates = []
        mapping = getattr(preprocessor, "named_transformers_", None)
        if mapping:
            candidates.extend(mapping.values())
        else:
            candidates.append(preprocessor)

        for sub in candidates:
            named_steps = getattr(sub, "named_steps", None)
            objs = list(named_steps.values()) if named_steps else [sub]
            for obj in objs:
                names = getattr(obj, "feature_names_in_", None)
                cats = getattr(obj, "categories_", None)
                if names is not None and cats is not None:
                    names_list = list(names)
                    if feat in names_list:
                        idx = names_list.index(feat)
                        return [str(c) for c in cats[idx]]
    except Exception:
        pass
    return None


def render_clinical_select(feat, preprocessor):
    label = prettify_label(feat)
    key_lower = feat.lower()

    # 1) Read the real trained categories straight off the fitted encoder
    #    when possible -- this is guaranteed to match what the model was
    #    actually trained on.
    fitted_categories = get_fitted_categories(feat, preprocessor)
    if fitted_categories:
        return st.selectbox(label, fitted_categories, key=f"clin_{feat}")

    # 2) Common binary/tri-state fields recognized by keyword.
    for keyword, options in CLINICAL_SELECT_OPTIONS.items():
        if keyword in key_lower:
            choice = st.selectbox(label, options, key=f"clin_{feat}")
            if keyword == "grade":
                return int(choice)
            return choice

    # 3) Known categorical clinical fields (e.g. tumor subtype, laterality,
    #    surgery type, cellularity) that aren't numbers, shown with real
    #    labels even when no fitted encoder is available to confirm the
    #    exact training categories.
    cat_options = match_category_keyword(feat)
    if cat_options:
        return st.selectbox(label, cat_options, key=f"clin_{feat}")

    # 4) Genuinely numeric fields (age, tumor size, node count, etc.)
    lo, hi, default = guess_numeric_range(feat, preprocessor)
    input_mode = st.session_state.get("clinical_input_mode", "Exact number")

    if input_mode == "Multiple choice (Low / Normal / High)":
        choice = st.selectbox(label, BUCKET_OPTIONS, index=1, key=f"clin_{feat}")
        bucket_vals = {"Low": lo, "Normal": default, "High": hi}
        return bucket_vals[choice]

    if any(kw in key_lower for kw in CLINICAL_INTEGER_KEYWORDS):
        return float(st.number_input(label, min_value=int(round(lo)), max_value=int(round(hi)),
                                      value=int(round(default)), step=1, key=f"clin_{feat}"))
    step = 1.0 if float(default).is_integer() and (hi - lo) > 5 else 0.1
    return st.number_input(label, min_value=float(lo), max_value=float(hi),
                            value=float(default), step=step, format="%.2f", key=f"clin_{feat}")


def render_editable_table(features_subset, values, bucket_map, widget_key, prettify_fn,
                           show_search, search_placeholder=""):
    if show_search:
        search = st.text_input(
            "Search", key=f"search_{widget_key}",
            placeholder=search_placeholder, label_visibility="collapsed",
        )
        filtered = [f for f in features_subset if search.lower() in f.lower()] if search else features_subset
    else:
        search = ""
        filtered = features_subset

    if not filtered:
        st.caption("No matching features.")
        return

    bcol1, bcol2, bcol3, bcol4 = st.columns([1, 1, 1, 2])
    with bcol1:
        if st.button("Set shown: Low", key=f"low_{widget_key}", use_container_width=True):
            for f in filtered:
                values[f] = "Low"
    with bcol2:
        if st.button("Set shown: Normal", key=f"norm_{widget_key}", use_container_width=True):
            for f in filtered:
                values[f] = "Normal"
    with bcol3:
        if st.button("Set shown: High", key=f"high_{widget_key}", use_container_width=True):
            for f in filtered:
                values[f] = "High"

    df = pd.DataFrame({
        "Feature": [prettify_fn(f) for f in filtered],
        "Level": [values[f] for f in filtered],
    }, index=filtered)

    editor_kwargs = dict(
        hide_index=True,
        use_container_width=True,
        height=min(480, 46 + 36 * max(len(filtered), 1)),
        disabled=["Feature"],
        key=f"editor_{widget_key}_{search}",
    )
    if hasattr(st, "column_config") and hasattr(st.column_config, "SelectboxColumn"):
        editor_kwargs["column_config"] = {
            "Level": st.column_config.SelectboxColumn("Level", options=BUCKET_OPTIONS, required=True),
        }

    edited = st.data_editor(df, **editor_kwargs)
    for raw in edited.index:
        val = edited.loc[raw, "Level"]
        if val in BUCKET_OPTIONS:
            values[raw] = val


KEY_GENE_SYMBOLS = {
    "ESR1", "PGR", "ERBB2", "MKI67", "TP53", "PIK3CA", "BRCA1", "BRCA2",
    "PTEN", "AKT1", "GATA3", "FOXA1", "CCND1", "MYC", "RB1", "CDH1", "ATM",
    "CHEK2", "PALB2", "STK11", "NF1", "MAP3K1", "EGFR", "KRAS", "NOTCH1",
    "AURKA", "MDM2", "CDKN2A", "SMAD4", "APC", "VEGFA", "MTOR", "TSC1",
    "TSC2", "ERBB3", "FGFR1", "FGFR2", "MET", "JAK2", "STAT3", "BCL2",
    "CASP8", "XBP1", "TBX3", "RUNX1", "ARID1A", "KMT2C", "MAP2K4", "NCOR1",
}


def render_feature_section(all_features, session_key, prettify_fn, strip_suffix,
                            search_placeholder, bucket_map):
    if session_key not in st.session_state:
        st.session_state[session_key] = {f: "Normal" for f in all_features}
    values = st.session_state[session_key]

    def base_symbol(f):
        s = f[:-len(strip_suffix)] if strip_suffix and f.lower().endswith(strip_suffix) else f
        return s.upper()

    key_feats = sorted(
        [f for f in all_features if base_symbol(f) in KEY_GENE_SYMBOLS],
        key=prettify_fn,
    )
    other_feats = sorted(
        [f for f in all_features if base_symbol(f) not in KEY_GENE_SYMBOLS],
        key=prettify_fn,
    )

    non_normal_total = sum(1 for f in all_features if values[f] != "Normal")
    st.markdown(
        f"<span class='count-chip'>{len(key_feats)} key features</span>"
        f"<span class='count-chip'>{len(other_feats)} additional features</span>"
        f"<span class='count-chip'>{non_normal_total} set away from Normal overall</span>",
        unsafe_allow_html=True,
    )

    st.markdown("**Key features** (most clinically relevant -- shown by default)")
    if key_feats:
        render_editable_table(key_feats, values, bucket_map, f"{session_key}_key", prettify_fn, show_search=False)
    else:
        st.caption("None of this file's features matched the curated key-gene list. "
                   "Use the search box below to browse all of them instead.")

    with st.expander(f"All other features ({len(other_feats)})", expanded=False):
        render_editable_table(
            other_feats, values, bucket_map, f"{session_key}_other", prettify_fn,
            show_search=True, search_placeholder=search_placeholder,
        )

    return {f: bucket_map.get(f, DEFAULT_BUCKET_VALUES)[values[f]] for f in all_features}


# ==================================================================================
# CORE FIX #3 -- resolve which model input is which, instead of guessing
# A late-fusion Keras model with three named branches must receive its
# arrays in the exact order model.predict expects. Hard-coding
# [genomic, clinical, mutation] silently produces plausible-looking but
# wrong predictions whenever the model was actually compiled in a
# different order. This inspects model.input_names / model.inputs to
# resolve the mapping automatically, first by name, then by matching each
# input's expected feature width against the genomic/clinical/mutation
# feature counts.
# ==================================================================================
def determine_input_mapping(model, gen_dim, clin_dim, mut_dim):
    try:
        names = list(model.input_names)
        shapes = [tuple(t.shape) for t in model.inputs]
    except Exception:
        return None, "Could not introspect model inputs."
    if len(names) != 3:
        return None, f"Model has {len(names)} input(s), expected 3 (genomic, clinical, mutation)."

    NAME_HINTS = {
        "genomic": ["gene", "genom", "expr"],
        "clinical": ["clin"],
        "mutation": ["mut", "treat"],
    }
    mapping = {}
    used = set()
    for role, hints in NAME_HINTS.items():
        for i, name in enumerate(names):
            if i in used:
                continue
            if any(h in name.lower() for h in hints):
                mapping[role] = i
                used.add(i)
                break

    if len(mapping) == 3:
        return mapping, "Matched by input layer name."

    dims = {"genomic": gen_dim, "clinical": clin_dim, "mutation": mut_dim}
    remaining_roles = [r for r in dims if r not in mapping]
    remaining_idx = [i for i in range(3) if i not in used]
    if len(remaining_roles) != len(remaining_idx):
        return None, "Ambiguous input mapping -- name and shape matching both inconclusive."

    def feature_width(shape):
        vals = [d for d in shape[1:] if d not in (None, 1)]
        return vals[0] if vals else (shape[1] if len(shape) > 1 else None)

    unresolved = list(remaining_roles)
    for i in remaining_idx:
        width = feature_width(shapes[i])
        match = None
        for role in unresolved:
            if width == dims[role]:
                match = role
                break
        if match is None:
            return None, "Ambiguous input mapping -- could not match feature counts to input shapes."
        mapping[match] = i
        unresolved.remove(match)

    if len(mapping) == 3:
        return mapping, "Matched by expected feature width."
    return None, "Could not resolve input mapping."


def needs_channel_dim(model, idx):
    try:
        return len(model.inputs[idx].shape) == 3
    except Exception:
        return True


def validate_shapes(model, ordered_inputs):
    problems = []
    try:
        expected_shapes = [tuple(t.shape) for t in model.inputs]
    except Exception:
        return problems
    for i, (arr, exp) in enumerate(zip(ordered_inputs, expected_shapes)):
        actual = tuple(arr.shape)
        exp_nonbatch = [d for d in exp[1:] if d is not None]
        act_nonbatch = [d for d in actual[1:] if d != 1 or len(exp[1:]) == len(actual[1:])]
        # Compare the feature-count dimension specifically (ignore trailing 1s added for CNN channels).
        exp_feat = next((d for d in exp[1:] if d not in (None, 1)), None)
        act_feat = next((d for d in actual[1:] if d != 1), actual[1] if len(actual) > 1 else None)
        if exp_feat is not None and act_feat is not None and exp_feat != act_feat:
            problems.append(f"Input {i}: model expects feature width {exp_feat}, got {act_feat} "
                             f"(full expected shape {exp}, got {actual}).")
    return problems


# ==================================================================================
# CORE FIX #4 -- lightweight, no-API sensitivity analysis for interpretability
# No paid/external API is used or required. This ranks which inputs moved
# the prediction the most by perturbing one feature at a time (Low <-> High)
# and re-running the already-loaded local model, batched for speed.
# ==================================================================================
def compute_sensitivity(model, ordered_inputs, mapping, role_features, role_bucket_maps,
                         role_current_values, max_features_per_role=25):
    role_of_idx = {v: k for k, v in mapping.items()}
    baseline = float(np.ravel(model.predict(ordered_inputs, verbose=0))[0])
    results = []

    for role in ("genomic", "clinical", "mutation"):
        idx = mapping[role]
        base_arr = ordered_inputs[idx]
        feats = role_features[role][:max_features_per_role]
        if not feats or role == "clinical":
            # Clinical fields mix categories and continuous numbers with very
            # different scales -- a uniform perturbation isn't meaningful, so
            # sensitivity is reported for genomic/mutation inputs only.
            continue
        bucket_map = role_bucket_maps[role]
        col_names = role_features[role]
        for feat in feats:
            col = col_names.index(feat)
            low_val = bucket_map.get(feat, DEFAULT_BUCKET_VALUES)["Low"]
            high_val = bucket_map.get(feat, DEFAULT_BUCKET_VALUES)["High"]

            perturbed_batch = []
            for test_val in (low_val, high_val):
                arr = base_arr.copy()
                if arr.ndim == 3:
                    arr[0, col, 0] = test_val
                else:
                    arr[0, col] = test_val
                perturbed_batch.append(arr)
            batch = np.concatenate(perturbed_batch, axis=0)

            call_inputs = list(ordered_inputs)
            other_idxs = [i for i in range(3) if i != idx]
            tiled = [call_inputs[i].copy() for i in other_idxs]
            tiled = [np.repeat(t, 2, axis=0) for t in tiled]
            full_call = [None, None, None]
            full_call[idx] = batch
            for i, t in zip(other_idxs, tiled):
                full_call[i] = t

            preds = np.ravel(model.predict(full_call, verbose=0))
            spread = float(abs(preds[1] - preds[0]))
            results.append({"feature": prettify_gene(feat) if role == "genomic" else prettify_mutation(feat),
                             "role": role, "impact": spread})

    results.sort(key=lambda r: r["impact"], reverse=True)
    return baseline, results[:15]


# ==================================================================================
# PAGE CONFIG + THEME
# ==================================================================================
st.set_page_config(
    page_title="pCR Prediction Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#F5F3FA",
    "bg_alt": "#ECE7F7",
    "card": "#FFFFFF",
    "primary": "#5B3FA0",
    "primary_dark": "#402C74",
    "primary_light": "#B6A6DE",
    "accent": "#1E9E8E",
    "text": "#25203A",
    "text_soft": "#635C7C",
    "success": "#1E9E8E",
    "danger": "#C0392B",
    "border": "#DCD4EF",
}

st.markdown(f"""
<style>
    .stApp {{
        background: {PALETTE['bg']};
    }}
    h1, h2, h3, h4 {{
        color: {PALETTE['text']} !important;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        letter-spacing: -0.01em;
    }}
    p, span, label, div {{
        color: {PALETTE['text']};
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }}
    .hero {{
        background: linear-gradient(135deg, {PALETTE['primary_dark']} 0%, {PALETTE['primary']} 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 14px;
        color: white !important;
        margin-bottom: 1.4rem;
    }}
    .hero, .hero h1, .hero p, .hero span, .hero div {{ color: #FFFFFF !important; }}
    .hero .subtitle {{ color: #E4DDF5 !important; font-size: 0.95rem; }}
    .metric-card {{
        background: {PALETTE['card']};
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        border: 1px solid {PALETTE['border']};
    }}
    .status-ok {{ color: {PALETTE['success']}; font-weight: 700; }}
    .status-missing {{ color: {PALETTE['danger']}; font-weight: 700; }}
    .stButton>button {{
        background: {PALETTE['primary']};
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
    }}
    .stButton>button:hover {{ background: {PALETTE['primary_dark']}; color: #FFFFFF !important; }}
    .stButton>button p, .stButton>button span, .stButton>button div {{ color: #FFFFFF !important; }}
    div[data-testid="stSlider"] div[role="slider"] {{
        background-color: {PALETTE['primary']} !important;
        border-color: {PALETTE['primary']} !important;
    }}
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
        background: {PALETTE['primary']} !important;
    }}
    div[data-testid="stSliderTickBar"] {{ color: {PALETTE['text_soft']} !important; }}
    .stTabs [data-baseweb="tab"] {{ color: {PALETTE['text_soft']}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{
        color: {PALETTE['primary_dark']} !important;
        border-bottom-color: {PALETTE['primary']} !important;
    }}
    div[data-baseweb="select"] > div {{
        border-radius: 8px !important;
        border-color: {PALETTE['border']} !important;
    }}
    div[data-baseweb="input"] {{
        border-radius: 8px !important;
        border-color: {PALETTE['border']} !important;
    }}
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {{
        border-color: {PALETTE['primary']} !important;
        box-shadow: 0 0 0 1px {PALETTE['primary']} !important;
    }}
    .stNumberInput input:focus, .stTextInput input:focus {{
        border-color: {PALETTE['primary']} !important;
        box-shadow: 0 0 0 1px {PALETTE['primary']} !important;
    }}
    .section-card {{
        background: {PALETTE['card']};
        border-radius: 10px;
        padding: 0.4rem 1rem 0.6rem 1rem;
        border: 1px solid {PALETTE['border']};
        margin-bottom: 0.6rem;
    }}
    .count-chip {{
        display: inline-block;
        background: {PALETTE['bg_alt']};
        color: {PALETTE['primary_dark']};
        border: 1px solid {PALETTE['border']};
        border-radius: 999px;
        padding: 0.15rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }}
    .footnote {{ color: {PALETTE['text_soft']}; font-size: 0.82rem; }}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ==================================================================================
# CACHED LOADERS
# ==================================================================================
@st.cache_resource(show_spinner="Loading fusion model...")
def load_model():
    if not TF_AVAILABLE:
        return None, "TensorFlow/Keras is not installed."
    if not os.path.exists(PATHS["model"]):
        return None, f"Model file not found at {PATHS['model']}"
    try:
        model = keras.models.load_model(
            PATHS["model"],
            compile=False,
            custom_objects={"Dense": CompatibleDense},
        )
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner="Loading preprocessors...")
def load_preprocessors():
    out, errors = {}, {}
    for key in ["genomic_scaler", "mutation_scaler", "clinical_preprocessor"]:
        path = PATHS[key]
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                out[key] = patch_sklearn_compat(obj)
            except Exception as e:
                errors[key] = str(e)
                out[key] = None
        else:
            errors[key] = f"File not found at {path}"
            out[key] = None
    return out, errors


ensure_sklearn_unpickle_compat()
_model, _model_err = load_model()
_preprocs, _prep_errs = load_preprocessors()

GENOMIC_FEATURES, genomic_detected = get_feature_names(
    _preprocs.get("genomic_scaler"), FALLBACK_GENOMIC_FEATURES)
CLINICAL_FEATURES, clinical_detected = get_feature_names(
    _preprocs.get("clinical_preprocessor"), FALLBACK_CLINICAL_FEATURES)
MUTATION_FEATURES, mutation_detected = get_feature_names(
    _preprocs.get("mutation_scaler"), FALLBACK_MUTATION_FEATURES)

CLINICAL_PREPROCESSOR_MISSING = _preprocs.get("clinical_preprocessor") is None

GENOMIC_BUCKETS = derive_bucket_values(GENOMIC_FEATURES, _preprocs.get("genomic_scaler"))
MUTATION_BUCKETS = derive_bucket_values(MUTATION_FEATURES, _preprocs.get("mutation_scaler"))

INPUT_MAPPING, MAPPING_NOTE = (None, "Model not loaded.")
if _model is not None:
    INPUT_MAPPING, MAPPING_NOTE = determine_input_mapping(
        _model, len(GENOMIC_FEATURES), len(CLINICAL_FEATURES), len(MUTATION_FEATURES))


# ==================================================================================
# SIDEBAR
# ==================================================================================
with st.sidebar:
    st.markdown("### About this dashboard")
    st.markdown(
        "This tool runs a multimodal late-fusion model that combines "
        "genomic expression, clinical variables, and mutation/treatment "
        "history to estimate pathological complete response (pCR) to "
        "neoadjuvant chemotherapy in breast cancer."
    )
    st.markdown("---")
    st.markdown("### If predictions look wrong")
    st.markdown(
        "- Open **Model Input Mapping** below and confirm the order matches "
        "how `fusion_model.keras` was compiled.\n"
        "- Open **Model file status** and confirm all three preprocessor "
        "files loaded (a missing `clinical_preprocessor.joblib` forces an "
        "approximate fallback encoding).\n"
        "- Use **Prediction Insights** after running a prediction to see "
        "which inputs actually move the output -- if nothing moves it, the "
        "wrong tensors are likely reaching the wrong branch."
    )
    st.markdown("---")
    st.caption("No external API is used. All predictions and the sensitivity "
               "analysis run locally against the loaded model file.")


# ==================================================================================
# HERO
# ==================================================================================
st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:0.3rem; color:#FFFFFF !important;">Breast Cancer pCR Prediction</h1>
    <p class="subtitle" style="color:#E4DDF5 !important;">Multimodal late-fusion model &middot; genomic expression, clinical variables, mutation and treatment history</p>
    <p style="font-size:1.0rem; max-width: 760px; margin-top:0.8rem; color:#FFFFFF !important;">
    Set a patient's profile below to estimate pathological complete response
    (pCR) to neoadjuvant chemotherapy. Intended to support, not replace,
    clinical judgment.
    </p>
</div>
""", unsafe_allow_html=True)

if _model is None:
    st.warning(f"Model not loaded: {_model_err}. You can still explore the input form below.")

if CLINICAL_PREPROCESSOR_MISSING:
    st.warning(
        "clinical_preprocessor.joblib is missing or failed to load. Clinical category "
        "fields (ER/PR/HER2 status, menopausal status) will be converted using a "
        "simple approximate numeric mapping instead of the real trained encoding, so "
        "predictions below are for testing the UI only. Add the correct "
        "clinical_preprocessor.joblib next to this script to fix this."
    )

status_col1, status_col2 = st.columns(2)
with status_col1:
    with st.expander("Model file status", expanded=False):
        status_rows = [
            ("fusion_model.keras", _model is not None, _model_err),
            ("genomic_scaler.joblib", _preprocs.get("genomic_scaler") is not None,
             _prep_errs.get("genomic_scaler")),
            ("mutation_scaler.joblib", _preprocs.get("mutation_scaler") is not None,
             _prep_errs.get("mutation_scaler")),
            ("clinical_preprocessor.joblib", _preprocs.get("clinical_preprocessor") is not None,
             _prep_errs.get("clinical_preprocessor")),
        ]
        cols = st.columns(len(status_rows))
        for col, (name, ok, err) in zip(cols, status_rows):
            with col:
                css_class = "status-ok" if ok else "status-missing"
                label = "FOUND" if ok else "MISSING"
                st.markdown(f"**{name}**<br><span class='{css_class}'>{label}</span>", unsafe_allow_html=True)
                if not ok and err:
                    st.caption(err)
        st.caption(f"Genomic features detected: {len(GENOMIC_FEATURES)} "
                   f"({'from scaler' if genomic_detected else 'fallback'})")
        st.caption(f"Clinical features detected: {len(CLINICAL_FEATURES)} "
                   f"({'from preprocessor' if clinical_detected else 'fallback'})")
        st.caption(f"Mutation features detected: {len(MUTATION_FEATURES)} "
                   f"({'from scaler' if mutation_detected else 'fallback'})")

with status_col2:
    with st.expander("Model Input Mapping", expanded=False):
        if _model is None:
            st.caption("Load the model to inspect its input order.")
        elif INPUT_MAPPING is None:
            st.markdown(f"<span class='status-missing'>Could not auto-detect</span> -- {MAPPING_NOTE}",
                        unsafe_allow_html=True)
            st.caption("Falling back to the order [genomic, clinical, mutation]. "
                       "If predictions look wrong, this is the first thing to check -- "
                       "confirm this order against how the model was compiled and, "
                       "ideally, give the Keras Input layers matching names "
                       "(e.g. 'genomic_input') so this can be detected automatically.")
        else:
            st.markdown(f"<span class='status-ok'>Auto-detected</span> -- {MAPPING_NOTE}",
                        unsafe_allow_html=True)
            order_display = sorted(INPUT_MAPPING.items(), key=lambda kv: kv[1])
            st.caption("Model input order: " + " -> ".join(f"[{i}] {role}" for role, i in order_display))


# ==================================================================================
# INPUT TABS
# ==================================================================================
tab1, tab2, tab3 = st.tabs(["Clinical Variables", "Genomic Expression", "Mutation / Treatment History"])

clinical_input = {}
with tab1:
    st.caption(f"{len(CLINICAL_FEATURES)} clinical feature(s) detected, grouped below. "
               "Categorical fields always show a dropdown; numeric fields follow the mode below.")
    st.radio(
        "Numeric field input mode",
        ["Exact number", "Multiple choice (Low / Normal / High)"],
        key="clinical_input_mode",
        horizontal=True,
    )

    grouped = {}
    for feat in CLINICAL_FEATURES:
        grouped.setdefault(assign_clinical_group(feat), []).append(feat)

    ordered_groups = [g for g, _ in CLINICAL_GROUPS if g in grouped]
    if "Other Clinical Factors" in grouped:
        ordered_groups.append("Other Clinical Factors")

    for group_name in ordered_groups:
        feats_in_group = sorted(grouped[group_name], key=prettify_label)
        with st.expander(f"{group_name} ({len(feats_in_group)})", expanded=True):
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            ccols = st.columns(3)
            for i, feat in enumerate(feats_in_group):
                with ccols[i % 3]:
                    clinical_input[feat] = render_clinical_select(feat, _preprocs.get("clinical_preprocessor"))
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.caption(f"{len(GENOMIC_FEATURES)} genomic expression feature(s) detected. Low/Normal/High "
               "values are calibrated per-gene from the fitted scaler where available.")
    genomic_input = render_feature_section(
        GENOMIC_FEATURES, "genomic_values", prettify_gene, strip_suffix=None,
        search_placeholder="Search genes, e.g. BRCA1", bucket_map=GENOMIC_BUCKETS,
    )

with tab3:
    st.caption(f"{len(MUTATION_FEATURES)} mutation / treatment-history feature(s) detected. Low/Normal/High "
               "values are calibrated per-feature from the fitted scaler where available.")
    mutation_input = render_feature_section(
        MUTATION_FEATURES, "mutation_values", prettify_mutation, strip_suffix="_mut",
        search_placeholder="Search mutations, e.g. TP53", bucket_map=MUTATION_BUCKETS,
    )


# ==================================================================================
# PREDICTION
# ==================================================================================
st.markdown("---")
st.markdown("### Prediction")
threshold = st.slider(
    "Decision threshold for pCR classification",
    min_value=0.05, max_value=0.95, value=THRESHOLD_DEFAULT, step=0.01,
    help=(
        "The model outputs a probability of pCR, not a yes/no answer. "
        "This threshold sets the cutoff used to turn that probability into "
        "a label: predictions at or above this value are shown as "
        "'Likely pCR (Responder)', and predictions below it are shown as "
        "'Unlikely pCR (Non-responder)'. Raising the threshold requires "
        "more confidence before calling a patient a responder; lowering it "
        "does the opposite. 0.50 is the standard neutral cutoff."
    ),
)
st.caption(
    f"At the current setting, a predicted probability of {threshold:.0%} or higher "
    "will be labeled a likely responder; anything below that will be labeled unlikely."
)
run = st.button("Run Prediction", use_container_width=True)

if run:
    if _model is None:
        st.error("Cannot run prediction -- the fusion model isn't loaded. Check Model file status above.")
    else:
        try:
            if _preprocs.get("clinical_preprocessor") is not None:
                clin_raw = pd.DataFrame([clinical_input])[CLINICAL_FEATURES]
                clin_proc = _preprocs["clinical_preprocessor"].transform(clin_raw)
            else:
                clin_encoded = encode_clinical_fallback(clinical_input)
                clin_raw = pd.DataFrame([clin_encoded])[CLINICAL_FEATURES]
                clin_proc = clin_raw.values.astype(np.float32)
            clin_proc = np.asarray(clin_proc, dtype=np.float32)

            gen_raw = pd.DataFrame([genomic_input])[GENOMIC_FEATURES]
            gen_proc = _preprocs["genomic_scaler"].transform(gen_raw) \
                if _preprocs.get("genomic_scaler") is not None else gen_raw.values.astype(np.float32)
            gen_proc = np.asarray(gen_proc, dtype=np.float32)

            mut_raw = pd.DataFrame([mutation_input])[MUTATION_FEATURES]
            mut_proc = _preprocs["mutation_scaler"].transform(mut_raw) \
                if _preprocs.get("mutation_scaler") is not None else mut_raw.values.astype(np.float32)
            mut_proc = np.asarray(mut_proc, dtype=np.float32)

            processed = {"genomic": gen_proc, "clinical": clin_proc, "mutation": mut_proc}
            mapping = INPUT_MAPPING or {"genomic": 0, "clinical": 1, "mutation": 2}

            ordered_inputs = [None, None, None]
            for role, idx in mapping.items():
                arr = processed[role]
                if needs_channel_dim(_model, idx) and arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=-1)
                ordered_inputs[idx] = arr

            shape_problems = validate_shapes(_model, ordered_inputs)
            if shape_problems:
                st.error("Input shapes don't match what the model expects -- prediction was not run.")
                for p in shape_problems:
                    st.caption(p)
                st.info("This usually means a preprocessor was fit on a different feature set than "
                        "the model's corresponding input branch. Re-check that fusion_model.keras, "
                        "genomic_scaler.joblib, mutation_scaler.joblib, and clinical_preprocessor.joblib "
                        "all come from the same training run.")
                st.stop()

            pred = _model.predict(ordered_inputs, verbose=0)
            prob = float(np.ravel(pred)[0])

            if CLINICAL_PREPROCESSOR_MISSING:
                st.info(
                    "This prediction used the approximate fallback clinical encoding "
                    "(see warning above), not the real trained clinical_preprocessor. "
                    "Treat the result as a UI smoke test, not a clinical estimate."
                )
            if INPUT_MAPPING is None:
                st.info(
                    "Model input order could not be auto-detected, so the default order "
                    "[genomic, clinical, mutation] was used. Verify this against how the "
                    "model was compiled -- see Model Input Mapping above."
                )

            st.markdown("### Result")
            r1, r2 = st.columns([1, 1.4])
            with r1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "pCR Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": PALETTE["primary"]},
                        "steps": [
                            {"range": [0, 50], "color": PALETTE["bg_alt"]},
                            {"range": [50, 100], "color": PALETTE["primary_light"]},
                        ],
                        "threshold": {
                            "line": {"color": PALETTE["danger"], "width": 4},
                            "value": threshold * 100,
                        },
                    },
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with r2:
                label = "Likely pCR (Responder)" if prob >= threshold else "Unlikely pCR (Non-responder)"
                color = PALETTE["success"] if prob >= threshold else PALETTE["danger"]
                st.markdown(f"""
                <div class="metric-card">
                <h3 style="color:{color} !important;">{label}</h3>
                <p>Predicted probability: <b>{prob:.1%}</b></p>
                <p>Decision threshold: <b>{threshold:.0%}</b></p>
                <p class="footnote">
                This estimate integrates genomic expression, clinical, and
                treatment-history signals via the late-fusion model. Use
                alongside clinical judgment -- not a standalone diagnostic tool.
                </p>
                </div>
                """, unsafe_allow_html=True)

            report_lines = [
                "pCR Prediction Report",
                f"Predicted probability: {prob:.1%}",
                f"Decision threshold: {threshold:.0%}",
                f"Result: {label}",
                "",
                "Clinical inputs:",
            ] + [f"  {prettify_label(k)}: {v}" for k, v in clinical_input.items()] + [
                "",
                "Genomic inputs (non-Normal only):",
            ] + [f"  {prettify_gene(k)}: {st.session_state['genomic_values'][k]}"
                 for k in GENOMIC_FEATURES if st.session_state["genomic_values"][k] != "Normal"] + [
                "",
                "Mutation/treatment inputs (non-Normal only):",
            ] + [f"  {prettify_mutation(k)}: {st.session_state['mutation_values'][k]}"
                 for k in MUTATION_FEATURES if st.session_state["mutation_values"][k] != "Normal"]
            report_text = "\n".join(report_lines)
            st.download_button("Download report (.txt)", data=report_text,
                                file_name="pcr_prediction_report.txt", mime="text/plain")

            with st.expander("Prediction Insights (local sensitivity analysis, no external API)", expanded=False):
                st.caption(
                    "Each listed feature is temporarily swapped between its Low and High "
                    "value while every other input stays fixed, and the model is re-run "
                    "locally to measure how much the predicted probability moves. Larger "
                    "bars mean the model is more sensitive to that input. This is computed "
                    "entirely from the already-loaded model -- no external service is called."
                )
                role_features = {"genomic": GENOMIC_FEATURES, "clinical": CLINICAL_FEATURES,
                                  "mutation": MUTATION_FEATURES}
                role_bucket_maps = {"genomic": GENOMIC_BUCKETS, "clinical": {}, "mutation": MUTATION_BUCKETS}
                role_current_values = {"genomic": genomic_input, "clinical": clinical_input,
                                        "mutation": mutation_input}
                with st.spinner("Running local sensitivity analysis..."):
                    try:
                        _, sensitivity = compute_sensitivity(
                            _model, ordered_inputs, mapping, role_features,
                            role_bucket_maps, role_current_values,
                        )
                    except Exception as sens_err:
                        sensitivity = None
                        st.caption(f"Sensitivity analysis unavailable: {sens_err}")

                if sensitivity:
                    sens_df = pd.DataFrame(sensitivity)
                    fig2 = go.Figure(go.Bar(
                        x=sens_df["impact"], y=sens_df["feature"], orientation="h",
                        marker_color=PALETTE["primary"],
                    ))
                    fig2.update_layout(
                        height=max(280, 28 * len(sens_df)),
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Change in predicted probability (Low to High)",
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                elif sensitivity == []:
                    st.caption("No genomic or mutation features were available to analyze.")

        except AttributeError as e:
            st.error(f"Prediction failed: {e}")
            st.info("This is a scikit-learn version mismatch: the .joblib file was saved with a "
                    "different scikit-learn version than what's installed here, so an internal "
                    "attribute the current version expects is missing from the old pickle. The app "
                    "already applies a best-effort patch for this on SimpleImputer objects, but it "
                    "may not cover every affected step. The reliable fix is to install the exact "
                    "scikit-learn version used when the preprocessor was fit and saved (check with "
                    "`pip show scikit-learn` in your training environment), or re-fit and re-save "
                    "the preprocessor using the scikit-learn version installed here.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("If this is a feature-name mismatch, one of the preprocessors was likely fit on "
                    "a plain array rather than a named DataFrame, so this app fell back to placeholder "
                    "names for it. Re-save that preprocessor from a fitted DataFrame. If the shapes "
                    "otherwise look right, check Model Input Mapping above to confirm the input order.")
