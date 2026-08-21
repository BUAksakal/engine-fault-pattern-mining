"""
Runs Apriori association rule mining over the one-hot sensor-bucket basket
and isolates the rules whose consequent is one of the three engine
condition labels (Normal / Warning / Critical).
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_rules(
    onehot: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.5,
    max_len: int = 3,
) -> pd.DataFrame:
    frequent_itemsets = apriori(
        onehot, min_support=min_support, use_colnames=True, max_len=max_len
    )
    if frequent_itemsets.empty:
        raise RuntimeError("No frequent itemsets found -- try lowering min_support.")

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    return rules


def condition_specific_rules(rules: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keeps only rules whose consequent is exactly {label}
    (label in {'Normal', 'Warning', 'Critical'}), sorted by lift."""
    mask = rules["consequents"].apply(lambda s: set(s) == {label})
    out = rules[mask].sort_values("lift", ascending=False).reset_index(drop=True)
    return out[["antecedents_str", "consequents_str", "support", "confidence", "lift"]]
