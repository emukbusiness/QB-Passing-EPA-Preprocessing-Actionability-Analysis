# Eamon Mukhopadhyay - Data Preprocessing Techniques for Applicability/Actionability in Analytics

#1 Imports and setup
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BLOCK_PATTERNS = ["epa", "rating", "qbr", "wpa", "success", "passer"]
BLOCK_EXACT = {"passing_epa"}

#2 Core helpers
def fraction_safe(top_series, bottom_series):
    top_series = pd.to_numeric(top_series, errors="coerce")
    bottom_series = pd.to_numeric(bottom_series, errors="coerce")
    out_series = top_series / bottom_series.replace(0, np.nan)
    return out_series.replace([np.inf, -np.inf], np.nan)


def existing_only(frame_obj, requested_cols):
    return [field_name for field_name in requested_cols if field_name in frame_obj.columns]


def list_suspicious_cols(frame_obj, target_name):
    flagged = {}
    for token in BLOCK_PATTERNS:
        flagged[token] = [
            field_name
            for field_name in frame_obj.columns
            if token in field_name.lower() and field_name != target_name
        ]
    return flagged


def leakage_filter(field_list, target_name):
    kept_fields = []
    removed_fields = []

    for field_name in field_list:
        field_low = field_name.lower()

        if field_name == target_name:
            removed_fields.append(field_name)
            continue

        if field_low in BLOCK_EXACT:
            removed_fields.append(field_name)
            continue

        if any(token in field_low for token in BLOCK_PATTERNS):
            removed_fields.append(field_name)
            continue

        kept_fields.append(field_name)

    return kept_fields, removed_fields


def add_model_fields(frame_obj):
    work_df = frame_obj.copy()

    if "completion_pct" not in work_df.columns and {"completions", "attempts"}.issubset(work_df.columns):
        work_df["completion_pct"] = fraction_safe(work_df["completions"], work_df["attempts"])

    if "yards_per_attempt" not in work_df.columns and {"passing_yards", "attempts"}.issubset(work_df.columns):
        work_df["yards_per_attempt"] = fraction_safe(work_df["passing_yards"], work_df["attempts"])

    if "yards_per_completion" not in work_df.columns and {"passing_yards", "completions"}.issubset(work_df.columns):
        work_df["yards_per_completion"] = fraction_safe(work_df["passing_yards"], work_df["completions"])

    if "td_rate" not in work_df.columns and {"passing_tds", "attempts"}.issubset(work_df.columns):
        work_df["td_rate"] = fraction_safe(work_df["passing_tds"], work_df["attempts"])

    if "int_rate" not in work_df.columns and {"passing_interceptions", "attempts"}.issubset(work_df.columns):
        work_df["int_rate"] = fraction_safe(work_df["passing_interceptions"], work_df["attempts"])

    if "pacr" not in work_df.columns and {"passing_yards", "passing_air_yards"}.issubset(work_df.columns):
        work_df["pacr"] = fraction_safe(work_df["passing_yards"], work_df["passing_air_yards"])

    if "total_tds" not in work_df.columns and {"passing_tds", "rushing_tds"}.issubset(work_df.columns):
        work_df["total_tds"] = work_df["passing_tds"].fillna(0) + work_df["rushing_tds"].fillna(0)

    if "total_yards" not in work_df.columns and {"passing_yards", "rushing_yards"}.issubset(work_df.columns):
        work_df["total_yards"] = work_df["passing_yards"].fillna(0) + work_df["rushing_yards"].fillna(0)

    if "total_first_downs" not in work_df.columns and {"passing_first_downs", "rushing_first_downs"}.issubset(work_df.columns):
        work_df["total_first_downs"] = work_df["passing_first_downs"].fillna(0) + work_df["rushing_first_downs"].fillna(0)

    if "total_turnovers_lost" not in work_df.columns:
        turnover_base = pd.Series(0, index=work_df.index, dtype="float64")
        if "passing_interceptions" in work_df.columns:
            turnover_base = turnover_base + work_df["passing_interceptions"].fillna(0)
        if "sack_fumbles_lost" in work_df.columns:
            turnover_base = turnover_base + work_df["sack_fumbles_lost"].fillna(0)
        if "rushing_fumbles_lost" in work_df.columns:
            turnover_base = turnover_base + work_df["rushing_fumbles_lost"].fillna(0)
        if turnover_base.sum() != 0:
            work_df["total_turnovers_lost"] = turnover_base

    if "qb_usage" not in work_df.columns:
        usage_base = pd.Series(0, index=work_df.index, dtype="float64")
        seen_any = False
        for field_name in ["attempts", "carries", "sacks_suffered"]:
            if field_name in work_df.columns:
                usage_base = usage_base + work_df[field_name].fillna(0)
                seen_any = True
        if seen_any:
            work_df["qb_usage"] = usage_base

    return work_df


def build_ridge_pipe(alpha_value=1.0):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha_value)),
        ]
    )


def ridge_bundle(x_data, y_data, alpha_value=1.0, seed_value=42):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.20, random_state=seed_value
    )

    ridge_pipe = build_ridge_pipe(alpha_value=alpha_value)
    ridge_pipe.fit(x_train, y_train)
    y_hat = ridge_pipe.predict(x_test)

    return {
        "pipe": ridge_pipe,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_hat": y_hat,
        "r2": r2_score(y_test, y_hat),
        "mae": mean_absolute_error(y_test, y_hat),
    }

#3 Plot and table writers
def save_bar_view(label_list, value_list, chart_title, y_axis_title, export_path):
    plt.figure(figsize=(9, 5))
    bar_obj = plt.bar(label_list, value_list)
    plt.title(chart_title)
    plt.ylabel(y_axis_title)
    plt.ylim(min(0, min(value_list) - 0.05), max(value_list) + 0.15)

    for bar_piece, val_now in zip(bar_obj, value_list):
        plt.text(
            bar_piece.get_x() + bar_piece.get_width() / 2,
            val_now + 0.02,
            f"{val_now:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(export_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_corr_view(frame_obj, corr_fields, png_path, csv_path):
    corr_frame = frame_obj[corr_fields].corr(numeric_only=True)
    corr_frame.to_csv(csv_path, index=True)

    plt.figure(figsize=(10, 8))
    heat_obj = plt.imshow(corr_frame, aspect="auto")
    plt.colorbar(heat_obj)
    plt.xticks(range(len(corr_fields)), corr_fields, rotation=45, ha="right")
    plt.yticks(range(len(corr_fields)), corr_fields)
    plt.title("Corr mat part 2: QB passing variables")

    for row_idx in range(len(corr_fields)):
        for col_idx in range(len(corr_fields)):
            plt.text(
                col_idx,
                row_idx,
                f"{corr_frame.iloc[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_perm_view(model_pipe, x_holdout, y_holdout, chosen_fields, png_path, csv_path):
    val_imp = permutation_importance(
        model_pipe,
        x_holdout,
        y_holdout,
        n_repeats=30,
        random_state=42,
        scoring="r2",
    )

    perm_frame = pd.DataFrame(
        {
            "feature": chosen_fields,
            "perm_drop_r2": val_imp.importances_mean,
        }
    ).sort_values("perm_drop_r2", ascending=True)

    perm_frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 7))
    plt.barh(perm_frame["feature"], perm_frame["perm_drop_r2"])
    plt.xlabel("Permutation importance (drop in R² when shuffled)")
    plt.title("Actionable feature importance")
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()


def median_flag(frame_obj, field_name):
    mid_point = frame_obj[field_name].median()
    high_low = np.where(frame_obj[field_name] >= mid_point, "High", "Low")
    return pd.Series(high_low, index=frame_obj.index), mid_point


def two_way_bin_table(frame_obj, field_left, field_right, target_name):
    left_flag, left_cut = median_flag(frame_obj, field_left)
    right_flag, right_cut = median_flag(frame_obj, field_right)

    scratch_df = frame_obj.copy()
    scratch_df[field_left + "_bin"] = left_flag
    scratch_df[field_right + "_bin"] = right_flag

    pair_order = [("High", "High"), ("High", "Low"), ("Low", "High"), ("Low", "Low")]
    record_list = []

    for left_state, right_state in pair_order:
        row_mask = (
            (scratch_df[field_left + "_bin"] == left_state)
            & (scratch_df[field_right + "_bin"] == right_state)
        )
        record_list.append(
            {
                field_left + "_bin": left_state,
                field_right + "_bin": right_state,
                "label": f"{left_state} {field_left} / {right_state} {field_right}",
                "count": int(row_mask.sum()),
                "avg_target": scratch_df.loc[row_mask, target_name].mean(),
                f"{field_left}_median": left_cut,
                f"{field_right}_median": right_cut,
            }
        )

    return pd.DataFrame.from_records(record_list)


def save_two_way_view(bin_frame, display_labels, chart_title, y_axis_title, png_path, csv_path):
    export_frame = bin_frame.copy()
    export_frame["chart_label"] = display_labels
    export_frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    bars_now = plt.bar(export_frame["chart_label"], export_frame["avg_target"])
    plt.ylabel(y_axis_title)
    plt.title(chart_title)
    plt.xticks(rotation=20, ha="right")

    for piece_now, val_now in zip(bars_now, export_frame["avg_target"]):
        bump = 0.25 if pd.notna(val_now) and val_now >= 0 else -0.75
        plt.text(
            piece_now.get_x() + piece_now.get_width() / 2,
            val_now + bump,
            f"{val_now:.2f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()

#4 Main workflow
def main():
    arg_tool = argparse.ArgumentParser(description="QB passing EPA analysis build")
    arg_tool.add_argument("excel_file", help="Path to the Excel workbook")
    arg_tool.add_argument("--sheet", default="Model_Data", help="Sheet name to load")
    arg_tool.add_argument("--target", default="passing_epa", help="Target field name")
    arg_tool.add_argument("--output_dir", default="qb_model_outputs", help="Folder for PNG and CSV outputs")
    arg_tool.add_argument("--alpha", default=1.0, type=float, help="Ridge alpha")
    run_args = arg_tool.parse_args()

    book_path = Path(run_args.excel_file)
    out_dir = Path(run_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not book_path.exists():
        raise FileNotFoundError(f"Could not find file: {book_path}")

    base_df = pd.read_excel(book_path, sheet_name=run_args.sheet)
    print(f"Loaded {len(base_df):,} rows and {len(base_df.columns):,} columns from {book_path.name}")

    if "include_in_model" in base_df.columns:
        base_df = base_df[base_df["include_in_model"] == 1].copy()
        print(f"Filtered include_in_model ==1 -> {len(base_df):,} rows")

    if run_args.target not in base_df.columns:
        raise ValueError(f"Target field '{run_args.target}' was not found")

    hit_map = list_suspicious_cols(base_df, run_args.target)
    print("\nSuspicious column scan:")
    for token_key, token_hits in hit_map.items():
        if len(token_hits) > 0:
            print(f" {token_key}: {token_hits}")

    ready_df = add_model_fields(base_df)
    ready_df = ready_df[ready_df[run_args.target].notna()].copy()
    print(f"Rows with non-missing target '{run_args.target}': {len(ready_df):,}")

    broad_candidates = [
        "season",
        "week",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "passing_cpoe",
        "passing_2pt_conversions",
        "sacks_suffered",
        "sack_yards_lost",
        "sack_fumbles",
        "sack_fumbles_lost",
        "sack_rate",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles",
        "rushing_fumbles_lost",
        "rushing_first_downs",
        "completion_pct",
        "yards_per_attempt",
        "yards_per_completion",
        "td_rate",
        "int_rate",
        "pacr",
        "total_tds",
        "total_yards",
        "total_first_downs",
        "total_turnovers_lost",
        "qb_usage",
    ]

    actionable_candidates = [
        "completions",
        "attempts",
        "passing_cpoe",
        "sack_rate",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "carries",
        "int_rate",
        "td_rate",
        "qb_usage",
        "total_turnovers_lost",
        "pacr",
    ]

    efficiency_candidates = [
        "passing_cpoe",
        "completion_pct",
        "sack_rate",
        "passing_air_yards",
        "passing_yards_after_catch",
        "pacr",
        "qb_usage",
        "carries",
        "total_turnovers_lost",
    ]

    broad_fields = existing_only(ready_df, broad_candidates)
    act_fields = existing_only(ready_df, actionable_candidates)
    eff_fields = existing_only(ready_df, efficiency_candidates)

    broad_fields, broad_removed = leakage_filter(broad_fields, run_args.target)
    act_fields, act_removed = leakage_filter(act_fields, run_args.target)
    eff_fields, eff_removed = leakage_filter(eff_fields, run_args.target)

    print("\nRemoved by leakage filter:")
    print(" Broad:", broad_removed)
    print(" Actoinable:", act_removed)
    print(" Efficiency only:", eff_removed)

    if len(broad_fields) < 5:
        raise ValueError(f"Broad model field list too small: {broad_fields}")
    if len(act_fields) < 5:
        raise ValueError(f"Actionable field list too small: {act_fields}")
    if len(eff_fields) < 4:
        raise ValueError(f"Effeiciency only field list too small: {eff_fields}")

    y_target = ready_df[run_args.target]

    broad_pack = ridge_bundle(ready_df[broad_fields], y_target, alpha_value=run_args.alpha)
    act_pack = ridge_bundle(ready_df[act_fields], y_target, alpha_value=run_args.alpha)
    eff_pack = ridge_bundle(ready_df[eff_fields], y_target, alpha_value=run_args.alpha)

    score_board = pd.DataFrame(
        [
            {
                "setup": "Naive cleaned broad model",
                "n_features": len(broad_fields),
                "r2": broad_pack["r2"],
                "mae": broad_pack["mae"],
            },
            {
                "setup": "Actionable subset",
                "n_features": len(act_fields),
                "r2": act_pack["r2"],
                "mae": act_pack["mae"],
            },
            {
                "setup": "Efficiency only subset",
                "n_features": len(eff_fields),
                "r2": eff_pack["r2"],
                "mae": eff_pack["mae"],
            },
        ]
    )
    score_board.to_csv(out_dir / "model_compare_story.csv", index=False)

    save_bar_view(
        [
            "Naive cleaned\nbroad model",
            "Actionable\nsubset",
            "Efficiency only\nsubset",
        ],
        score_board["r2"].tolist(),
        "How field filtering changed model fit",
        "Test R²",
        out_dir / "model_compare_story.png",
    )

    corr_fields = existing_only(
        ready_df,
        [
            run_args.target,
            "passing_cpoe",
            "completion_pct",
            "attempts",
            "qb_usage",
            "sack_rate",
            "passing_air_yards",
            "passing_yards_after_catch",
            "pacr",
            "carries",
            "total_turnovers_lost",
            "passing_first_downs",
        ],
    )

    save_corr_view(
        ready_df,
        corr_fields,
        out_dir / "corr_mat_part2.png",
        out_dir / "corr_mat_part2.csv",
    )

    save_perm_view(
        act_pack["pipe"],
        act_pack["x_test"],
        act_pack["y_test"],
        act_fields,
        out_dir / "feat_rank_actionable.png",
        out_dir / "feat_rank_actionable.csv",
    )

    if {"passing_air_yards", "completion_pct", run_args.target}.issubset(ready_df.columns):
        air_comp_tbl = two_way_bin_table(ready_df, "passing_air_yards", "completion_pct", run_args.target)
        save_two_way_view(
            air_comp_tbl,
            [
                "High air /\nHigh comp",
                "High air /\nLow comp",
                "Low air /\nHigh comp",
                "Low air /\nLow comp",
            ],
            "Selective aggression vs blind aggression",
            f"Average {run_args.target}",
            out_dir / "bin_air_comp_view.png",
            out_dir / "bin_air_comp_view.csv",
        )

    if {"qb_usage", "sack_rate", run_args.target}.issubset(ready_df.columns):
        usage_sack_tbl = two_way_bin_table(ready_df, "qb_usage", "sack_rate", run_args.target)
        save_two_way_view(
            usage_sack_tbl,
            [
                "High usage /\nHigh sack",
                "High usage /\nLow sack",
                "Low usage /\nHigh sack",
                "Low usage /\nLow sack",
            ],
            "Volume only pays off when sack rate stays low",
            f"Average {run_args.target}",
            out_dir / "bin_usage_sack_view.png",
            out_dir / "bin_usage_sack_view.csv",
        )

    if {"passing_yards_after_catch", "total_turnovers_lost", run_args.target}.issubset(ready_df.columns):
        yac_turn_tbl = two_way_bin_table(ready_df, "passing_yards_after_catch", "total_turnovers_lost", run_args.target)
        save_two_way_view(
            yac_turn_tbl,
            [
                "High YAC /\nHigh TO",
                "High YAC /\nLow TO",
                "Low YAC /\nHigh TO",
                "Low YAC /\nLow TO",
            ],
            "YAC is elite when the offense stays clean",
            f"Average {run_args.target}",
            out_dir / "bin_yac_turn_view.png",
            out_dir / "bin_yac_turn_view.csv",
        )

    field_book = pd.DataFrame(
        {
            "broad_fields": pd.Series(broad_fields, dtype="object"),
            "actionable_fields": pd.Series(act_fields, dtype="object"),
            "efficiency_fields": pd.Series(eff_fields, dtype="object"),
        }
    )
    field_book.to_csv(out_dir / "field_sets_used.csv", index=False)

    print("\nSaved outputs to:")
    print(out_dir.resolve())

    print("\nModel comparison:")
    print(score_board.to_string(index=False))

    print("\nBroad model fields:")
    print(broad_fields)

    print("\nActionable model fields:")
    print(act_fields)

    print("\nEfficiency only fields:")
    print(eff_fields)

    print("\nDone.")


#5 main call
if __name__ == "__main__":
    main()