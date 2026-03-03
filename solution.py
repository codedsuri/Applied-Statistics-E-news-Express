#!/usr/bin/env python3
"""Statistical analysis for the E-news Express A/B test problem.

Usage:
    python solution.py --data abtest.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import pandas as pd
from scipy import stats

ALPHA = 0.05


@dataclass
class TestResult:
    question: str
    test_name: str
    statistic_label: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    interpretation: str


def _validate_columns(df: pd.DataFrame) -> None:
    required = {
        "user_id",
        "group",
        "landing_page",
        "time_spent_on_the_page",
        "converted",
        "language_preferred",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _to_binary(series: pd.Series) -> pd.Series:
    """Convert common conversion labels to binary 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)

    cleaned = series.astype(str).str.strip().str.lower()
    positive = {"yes", "y", "true", "1", "converted"}
    return cleaned.isin(positive).astype(int)


def question_1_time_new_gt_old(df: pd.DataFrame, alpha: float = ALPHA) -> TestResult:
    old = df.loc[df["landing_page"].str.lower() == "old", "time_spent_on_the_page"]
    new = df.loc[df["landing_page"].str.lower() == "new", "time_spent_on_the_page"]

    # H0: mean_new <= mean_old ; H1: mean_new > mean_old
    t_stat, p_two_sided = stats.ttest_ind(new, old, equal_var=False, nan_policy="omit")
    if math.isnan(t_stat) or math.isnan(p_two_sided):
        raise ValueError("Unable to compute t-test for Question 1.")

    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
    reject = p_one_sided < alpha

    return TestResult(
        question="Q1: Do users spend more time on the new page than the old page?",
        test_name="Welch two-sample t-test (one-sided)",
        statistic_label="t",
        statistic=float(t_stat),
        p_value=float(p_one_sided),
        alpha=alpha,
        reject_null=reject,
        interpretation=(
            "Reject H0: evidence that mean time on the new page is greater than old page."
            if reject
            else "Fail to reject H0: insufficient evidence that new page has higher mean time."
        ),
    )


def question_2_conversion_new_gt_old(df: pd.DataFrame, alpha: float = ALPHA) -> TestResult:
    data = df.copy()
    data["converted_bin"] = _to_binary(data["converted"])

    new = data.loc[data["landing_page"].str.lower() == "new", "converted_bin"]
    old = data.loc[data["landing_page"].str.lower() == "old", "converted_bin"]

    x1, n1 = int(new.sum()), int(new.count())
    x2, n2 = int(old.sum()), int(old.count())
    if n1 == 0 or n2 == 0:
        raise ValueError("Both landing_page groups must have at least one observation.")

    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        raise ValueError("Standard error is zero in two-proportion z-test.")

    p1_hat = x1 / n1
    p2_hat = x2 / n2
    z_stat = (p1_hat - p2_hat) / se
    p_one_sided = 1 - stats.norm.cdf(z_stat)
    reject = p_one_sided < alpha

    return TestResult(
        question="Q2: Is conversion rate for new page greater than old page?",
        test_name="Two-proportion z-test (one-sided)",
        statistic_label="z",
        statistic=float(z_stat),
        p_value=float(p_one_sided),
        alpha=alpha,
        reject_null=reject,
        interpretation=(
            "Reject H0: evidence that new page conversion rate is higher than old page."
            if reject
            else "Fail to reject H0: insufficient evidence that new page conversion is higher."
        ),
    )


def question_3_conversion_depends_on_language(df: pd.DataFrame, alpha: float = ALPHA) -> TestResult:
    data = df.copy()
    data["converted_bin"] = _to_binary(data["converted"])

    contingency = pd.crosstab(data["language_preferred"], data["converted_bin"])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    reject = p_value < alpha

    return TestResult(
        question="Q3: Does converted status depend on preferred language?",
        test_name="Chi-square test of independence",
        statistic_label="chi2",
        statistic=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
        reject_null=reject,
        interpretation=(
            "Reject H0: converted status depends on preferred language."
            if reject
            else "Fail to reject H0: no evidence of dependence between conversion and language."
        ),
    )


def question_4_new_page_time_same_by_language(df: pd.DataFrame, alpha: float = ALPHA) -> TestResult:
    new_page = df.loc[df["landing_page"].str.lower() == "new"].copy()
    groups = [
        grp["time_spent_on_the_page"].dropna().values
        for _, grp in new_page.groupby("language_preferred")
    ]
    if len(groups) < 2:
        raise ValueError("Need at least two language groups for ANOVA.")

    f_stat, p_value = stats.f_oneway(*groups)
    reject = p_value < alpha

    return TestResult(
        question="Q4: Is time spent on new page same across language users?",
        test_name="One-way ANOVA",
        statistic_label="F",
        statistic=float(f_stat),
        p_value=float(p_value),
        alpha=alpha,
        reject_null=reject,
        interpretation=(
            "Reject H0: mean time on new page differs for at least one language."
            if reject
            else "Fail to reject H0: no evidence of mean time differences by language on new page."
        ),
    )


def run_analysis(data_path: str, alpha: float = ALPHA) -> list[TestResult]:
    df = pd.read_csv(data_path)
    _validate_columns(df)

    results = [
        question_1_time_new_gt_old(df, alpha),
        question_2_conversion_new_gt_old(df, alpha),
        question_3_conversion_depends_on_language(df, alpha),
        question_4_new_page_time_same_by_language(df, alpha),
    ]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve E-news Express statistical hypothesis tests.")
    parser.add_argument("--data", required=True, help="Path to CSV data file.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Significance level (default: 0.05).")
    args = parser.parse_args()

    results = run_analysis(args.data, args.alpha)

    print(f"Significance level (alpha): {args.alpha}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.question}")
        print(f"   Test: {r.test_name}")
        print(f"   {r.statistic_label}-statistic: {r.statistic:.6f}")
        print(f"   p-value: {r.p_value:.6f}")
        print(f"   Decision: {'Reject H0' if r.reject_null else 'Fail to reject H0'}")
        print(f"   Interpretation: {r.interpretation}\n")


if __name__ == "__main__":
    main()
