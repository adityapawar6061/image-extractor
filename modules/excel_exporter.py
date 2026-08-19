"""Excel and CSV export functionality."""

import io
from datetime import datetime
from typing import Any

import pandas as pd


def build_excel(
    all_rows: list[dict[str, Any]],
    needs_review: list[dict[str, Any]],
    failed_images: list[dict[str, Any]],
    summary: dict[str, Any],
    columns: list[str],
) -> bytes:
    """Build an Excel workbook with multiple sheets.

    Returns the workbook as bytes for download.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Extracted Data
        if all_rows:
            df_data = pd.DataFrame(all_rows)
            # Ensure column order: source_image, source_row, then user columns
            ordered_cols = []
            for c in ["source_image", "source_row"] + columns:
                if c in df_data.columns:
                    ordered_cols.append(c)
            # Add any remaining columns
            for c in df_data.columns:
                if c not in ordered_cols:
                    ordered_cols.append(c)
            # Drop internal cols
            for drop_col in ("needs_review", "review_reasons"):
                if drop_col in ordered_cols:
                    ordered_cols.remove(drop_col)
            df_data = df_data[ordered_cols]
        else:
            df_data = pd.DataFrame(columns=["source_image", "source_row"] + columns)
        df_data.to_excel(writer, sheet_name="Extracted Data", index=False)

        # Sheet 2: Needs Review
        if needs_review:
            df_review = pd.DataFrame(needs_review)
            for drop_col in ("needs_review",):
                if drop_col in df_review.columns:
                    df_review = df_review.drop(columns=[drop_col])
            # Convert review_reasons list to string
            if "review_reasons" in df_review.columns:
                df_review["review_reasons"] = df_review["review_reasons"].apply(
                    lambda x: "; ".join(x) if isinstance(x, list) else str(x)
                )
        else:
            df_review = pd.DataFrame(columns=["source_image", "source_row"])
        df_review.to_excel(writer, sheet_name="Needs Review", index=False)

        # Sheet 3: Failed Images
        if failed_images:
            df_failed = pd.DataFrame(failed_images)
        else:
            df_failed = pd.DataFrame(
                columns=["filename", "error", "retry_count", "timestamp"]
            )
        df_failed.to_excel(writer, sheet_name="Failed Images", index=False)

        # Sheet 4: Processing Summary
        summary_data = [
            {"Metric": "Total Images", "Value": summary.get("total_images", 0)},
            {"Metric": "Processed", "Value": summary.get("processed", 0)},
            {"Metric": "Failed", "Value": summary.get("failed", 0)},
            {"Metric": "Needs Review", "Value": summary.get("needs_review_count", 0)},
            {"Metric": "Total Rows Extracted", "Value": summary.get("total_rows", 0)},
            {"Metric": "Estimated Cost (USD)", "Value": f"${summary.get('cost_usd', 0):.4f}"},
            {"Metric": "Estimated Cost (INR)", "Value": f"₹{summary.get('cost_inr', 0):.2f}"},
            {"Metric": "Generated At", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        ]
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Processing Summary", index=False)

    return output.getvalue()


def build_csv(all_rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Build a CSV string from all extracted rows."""
    if all_rows:
        df = pd.DataFrame(all_rows)
        ordered_cols = []
        for c in ["source_image", "source_row"] + columns:
            if c in df.columns:
                ordered_cols.append(c)
        for c in df.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)
        for drop_col in ("needs_review", "review_reasons"):
            if drop_col in ordered_cols:
                ordered_cols.remove(drop_col)
        df = df[ordered_cols]
    else:
        df = pd.DataFrame(columns=["source_image", "source_row"] + columns)
    return df.to_csv(index=False)
