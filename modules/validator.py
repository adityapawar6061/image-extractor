"""Data validation for extracted rows."""

import re
from datetime import datetime
from typing import Any


# Common date formats to validate
DATE_PATTERNS = [
    r"\d{2}-\d{2}-\d{4}",     # DD-MM-YYYY
    r"\d{2}/\d{2}/\d{4}",     # DD/MM/YYYY
    r"\d{4}-\d{2}-\d{2}",     # YYYY-MM-DD
    r"\d{2}\.\d{2}\.\d{4}",   # DD.MM.YYYY
    r"\d{2}-\d{2}-\d{2}",     # DD-MM-YY
    r"\d{2}/\d{2}/\d{2}",     # DD/MM/YY
]

# Phone number patterns (Indian format, 10 digits)
PHONE_PATTERN = re.compile(r"^[\d\s\-\+\(\)]{7,15}$")
# Indian mobile numbers should be exactly 10 digits
INDIAN_MOBILE_LEN = 10


def is_null_value(value: Any) -> bool:
    """Check if a value is considered null/missing."""
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in ("", "null", "none", "n/a", "na", "-", "--", "---"):
            return True
    return False


def is_valid_date_format(value: str) -> bool:
    """Check if a value matches any common date pattern."""
    if not isinstance(value, str):
        return False
    return any(re.match(pat, value.strip()) for pat in DATE_PATTERNS)


def is_suspicious_phone(value: Any) -> bool:
    """Check if a phone number looks suspicious."""
    if is_null_value(value):
        return False
    val = str(value).strip()
    digits_only = re.sub(r"\D", "", val)
    # Indian mobile numbers must be exactly 10 digits
    if len(digits_only) != INDIAN_MOBILE_LEN:
        return True
    return False


def is_suspicious_amount(value: Any) -> bool:
    """Check if an amount value looks suspicious."""
    if is_null_value(value):
        return False
    val = str(value).strip()
    # Remove common currency symbols and commas
    cleaned = re.sub(r"[₹$,€£\s]", "", val)
    try:
        num = float(cleaned)
        if num < 0:
            return True  # Negative amounts might be suspicious
    except ValueError:
        # Non-numeric amounts could be suspicious
        if cleaned and not cleaned.replace(".", "").replace("-", "").isdigit():
            return True
    return False


def fix_ocr_phone(value: Any) -> str | None:
    """Try to fix common OCR errors in phone numbers.

    Rules:
    - Must be exactly 10 digits
    - If not 10 digits, return as-is for review (don't guess)
    - We flag rather than auto-fix, since wrong digit removal corrupts data
    """
    if is_null_value(value):
        return None
    val = str(value).strip()
    digits_only = re.sub(r"\D", "", val)
    return digits_only if digits_only else None


def fix_ocr_amount(value: Any) -> str | None:
    """Try to fix common OCR errors in amounts.

    Rules:
    - Must be numeric only (no letters)
    - Remove currency symbols, text prefixes, and commas
    """
    if is_null_value(value):
        return None
    val = str(value).strip()
    # Remove common currency prefixes: Rs, INR, USD, etc.
    cleaned = re.sub(r"^[A-Za-z₹€£$]+[.\s]*", "", val)
    # Remove commas, spaces, and currency symbols
    cleaned = re.sub(r"[,\s₹€£$]", "", cleaned)
    # Remove any remaining non-digit characters except decimal point
    cleaned = re.sub(r"[^\d.]", "", cleaned)
    if not cleaned:
        return None
    return cleaned


def fix_ocr_city(value: Any) -> str | None:
    """Try to fix common OCR errors in city names.

    Rules:
    - Convert to UPPERCASE for consistency
    - Remove extra spaces
    - Fix common OCR mistakes like ERNAKULAM → ERNAKULAM
    """
    if is_null_value(value):
        return None
    val = str(value).strip()
    if not val:
        return None
    # Normalize: uppercase, strip whitespace
    fixed = val.upper().strip()
    # Fix common OCR city name mistakes
    city_corrections = {
        "ERNKALUM": "ERNAKULAM",
        "ERNKULAM": "ERNAKULAM",
        "GUNTOOR": "GUNTUR",
        "PUNE": "PUNE",
    }
    fixed = city_corrections.get(fixed, fixed)
    return fixed


def fix_ocr_name(value: Any) -> str | None:
    """Try to fix common OCR errors in names.

    Rules:
    - Title case each word
    - Remove extra spaces
    - Do NOT change spelling (OCR errors are hard to auto-fix)
    """
    if is_null_value(value):
        return None
    val = str(value).strip()
    if not val:
        return None
    # Normalize whitespace
    fixed = " ".join(val.split())
    # Title case each word
    fixed = fixed.title()
    return fixed


def is_empty_name(value: Any) -> bool:
    """Check if a name field is empty."""
    if is_null_value(value):
        return True
    if isinstance(value, str) and len(value.strip()) < 2:
        return True
    return False


def validate_row(
    row: dict[str, Any],
    columns: list[str],
    expected_rows: int = 38,
    row_index: int = 0,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a single extracted row.

    Returns (annotated_row, list_of_review_reasons).
    """
    review_reasons = []
    annotated = dict(row)

    # Check for missing fields
    for col in columns:
        if col not in annotated or is_null_value(annotated.get(col)):
            review_reasons.append(f"Missing field: {col}")

    # Check for empty names (if 'Name' column exists)
    if "Name" in annotated:
        if is_empty_name(annotated["Name"]):
            review_reasons.append("Empty or very short name")

    # Check date formats (if 'Date' column exists)
    if "Date" in annotated and not is_null_value(annotated.get("Date")):
        if not is_valid_date_format(str(annotated["Date"])):
            review_reasons.append(f"Unusual date format: {annotated['Date']}")

    # Check phone numbers (if a phone-like column exists)
    for col in columns:
        if any(kw in col.lower() for kw in ("phone", "mobile", "tel", "contact")):
            if not is_null_value(annotated.get(col)) and is_suspicious_phone(
                annotated[col]
            ):
                review_reasons.append(f"Suspicious phone number in '{col}'")

    # Check amounts (if an amount column exists)
    for col in columns:
        if any(kw in col.lower() for kw in ("amount", "price", "total", "sum", "cost")):
            if not is_null_value(annotated.get(col)) and is_suspicious_amount(
                annotated[col]
            ):
                review_reasons.append(f"Suspicious amount in '{col}'")

    # Mark row
    annotated["needs_review"] = len(review_reasons) > 0
    annotated["review_reasons"] = review_reasons
    annotated["source_row"] = row_index

    return annotated, review_reasons


def validate_batch(
    rows: list[dict[str, Any]],
    columns: list[str],
    expected_rows: int = 38,
    tolerance: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Validate a batch of rows from one image.

    Returns:
        (validated_rows, needs_review_rows, global_warnings)
    """
    validated = []
    needs_review = []
    global_warnings = []

    # Row count check
    row_count = len(rows)
    if expected_rows > 0:
        if row_count < (expected_rows - tolerance):
            global_warnings.append(
                f"Row count {row_count} is significantly below expected {expected_rows} "
                f"(tolerance ±{tolerance})"
            )
        elif row_count > (expected_rows + tolerance):
            global_warnings.append(
                f"Row count {row_count} is significantly above expected {expected_rows} "
                f"(tolerance ±{tolerance})"
            )
        elif row_count != expected_rows:
            global_warnings.append(
                f"Row count {row_count} differs from expected {expected_rows} "
                f"(within tolerance)"
            )

    # Check for duplicate rows
    seen = set()
    for i, row in enumerate(rows):
        # Simple duplicate detection using a hash of all values
        row_hash = tuple(
            str(row.get(c, "")) for c in sorted(row.keys()) if c != "needs_review"
        )
        if row_hash in seen:
            global_warnings.append(f"Possible duplicate row at index {i}")
        seen.add(row_hash)

    for i, row in enumerate(rows):
        # Apply OCR fixes before validation
        if "Number" in row:
            row["Number"] = fix_ocr_phone(row["Number"])
        if "City" in row:
            row["City"] = fix_ocr_city(row["City"])
        if "Name" in row:
            row["Name"] = fix_ocr_name(row["Name"])
        if "Amount" in row:
            row["Amount"] = fix_ocr_amount(row["Amount"])

        annotated, reasons = validate_row(row, columns, expected_rows, row_index=i)
        validated.append(annotated)
        if reasons or annotated.get("needs_review"):
            needs_review.append(annotated)

    return validated, needs_review, global_warnings
