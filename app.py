"""AI Image Data Extractor - Main Streamlit Application."""

import asyncio
import io
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

from modules.cost_tracker import CostTracker
from modules.excel_exporter import build_csv, build_excel
from modules.image_processor import get_supported_files
from modules.openai_extractor import extract_single_image, process_batch
from modules.state_manager import (
    get_all_results,
    get_completed_filenames,
    get_failed_images,
    get_failed_filenames,
    get_needs_review_results,
    init_image_state,
    load_state,
    reset_for_new_job,
    save_failed_log,
    save_result_json,
    save_state,
    update_image_state,
)
from modules.validator import validate_batch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "processing.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("app")

# Mask the API key in logs
_orig_log = logging.Logger.callHandlers


def _safe_emit(record):
    if hasattr(record, "msg") and isinstance(record.msg, str):
        record.msg = record.msg.replace(
            st.session_state.get("api_key", ""), "***REDACTED***"
        ) if st.session_state.get("api_key") else record.msg
    return _orig_log(record)


# Load .env if present
load_dotenv()

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Image Data Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if "processing_state" not in st.session_state:
    st.session_state.processing_state = load_state()
if "uploaded_files_map" not in st.session_state:
    st.session_state.uploaded_files_map = {}  # name -> bytes
if "cost_tracker" not in st.session_state:
    st.session_state.cost_tracker = CostTracker()
if "processing" not in st.session_state:
    st.session_state.processing = False
if "show_converter" not in st.session_state:
    st.session_state.show_converter = False
if "converter_files" not in st.session_state:
    st.session_state.converter_files = {}
if "converted_zip" not in st.session_state:
    st.session_state.converted_zip = None


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.header("⚙️ OPENAI SETTINGS")
    st.divider()

    api_key = st.text_input(
        "Enter your OpenAI API key",
        value=st.session_state.api_key,
        type="password",
        help="Your key is stored only in this session and sent only to OpenAI.",
    )
    if api_key:
        st.session_state.api_key = api_key

    if st.button("🔑 Test API Key", width="stretch"):
        if not api_key:
            st.error("Please enter an API key first.")
        else:
            try:
                from openai import OpenAI

                test_client = OpenAI(api_key=api_key)
                test_client.models.list()
                st.success("✅ API key is valid!")
            except Exception as e:
                st.error(f"❌ API key test failed: {e}")

    st.divider()

    model = st.selectbox(
        "Model",
        [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4.1",
            "gpt-4o-vision-preview",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-5.6-luna",
        ],
        index=0,
        help="gpt-4o-mini is cost-efficient for high-volume extraction.",
    )

    concurrency = st.slider(
        "Concurrent requests",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of simultaneous API calls.",
    )

    st.divider()
    st.header("🔍 VALIDATION")
    st.divider()

    expected_rows = st.number_input(
        "Expected rows per image",
        min_value=1,
        value=38,
        help="Approximate number of table rows per image.",
    )
    tolerance = st.number_input(
        "Tolerance (±)",
        min_value=0,
        value=2,
        help="Acceptable deviation from expected row count.",
    )

    st.divider()
    st.header("📝 EXTRACTION SETTINGS")
    st.divider()

    # Column configuration
    default_cols = "Number\nCity\nName\nDate\nAmount"
    cols_text = st.text_area(
        "Column names (one per line)",
        value=default_cols,
        height=120,
        help="Enter the column names you want to extract, one per line.",
    )
    columns = [c.strip() for c in cols_text.strip().split("\n") if c.strip()]

    custom_instructions = st.text_area(
        "Custom extraction instructions (optional)",
        value="",
        height=80,
        help="Additional instructions for the AI model.",
    )

    st.divider()
    st.header("💰 COST ESTIMATION")
    st.divider()

    usd_to_inr = st.number_input(
        "USD → INR exchange rate",
        min_value=1.0,
        value=95.0,
        step=0.5,
    )

    st.divider()
    st.header("🔄 TOOLS")
    st.divider()

    if st.button("🔄 HEIC/HEIF → JPG Converter", width="stretch"):
        st.session_state.show_converter = True
        st.rerun()

    st.divider()
    st.header("📂 PROCESSING")
    st.divider()

    if st.button("🔄 Resume Previous Job", width="stretch"):
        st.session_state.processing_state = load_state()
        st.success("Previous state loaded!")
        st.rerun()

    if st.button("🗑️ Start Fresh Job", width="stretch"):
        st.session_state.processing_state = reset_for_new_job(
            st.session_state.processing_state
        )
        st.session_state.uploaded_files_map = {}
        st.session_state.cost_tracker = CostTracker(model=model)
        save_state(st.session_state.processing_state)
        st.success("State cleared!")
        st.rerun()

    if st.button("🧹 Clear All Cache & Data", width="stretch"):
        import shutil

        # Clear everything: session state, files, converter cache
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Delete processing state file
        state_file = Path("processing_state.json")
        if state_file.exists():
            state_file.unlink()
        # Delete all result/failed/upload JSON files
        for subdir in ("data/results", "data/failed", "data/uploads"):
            sub = Path(subdir)
            if sub.exists():
                shutil.rmtree(sub)
                sub.mkdir(parents=True, exist_ok=True)
        st.success("All cache and data cleared!")
        st.rerun()


# ===========================================================================
# HEIC/HEIF CONVERTER (toggle-able section)
# ===========================================================================
if st.session_state.show_converter:
    st.title("🔄 HEIC/HEIF → JPG Converter")
    st.caption("Convert thousands of HEIC/HEIF images to JPG in one click")

    # Back button
    if st.button("← Back to Data Extractor", width="stretch"):
        st.session_state.show_converter = False
        st.rerun()

    # Settings
    st.header("⚙️ Settings")
    col_q, col_w = st.columns(2)
    with col_q:
        jpg_quality = st.slider("JPG Quality", 1, 100, 95, help="Higher = better quality, larger file size.")
    with col_w:
        max_width = st.number_input("Max width (px, 0 = no limit)", 0, step=100, help="Resize if wider. 0 = keep original.")

    # Upload
    st.header("📷 Upload Images")
    uploaded_conv = st.file_uploader(
        "Upload HEIC, HEIF, JPG, JPEG, PNG, or WEBP images",
        type=["heic", "heif", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="You can upload 2000+ images at once.",
        key="conv_uploader",
    )

    if uploaded_conv:
        for uf in uploaded_conv:
            if uf.name not in st.session_state.converter_files:
                st.session_state.converter_files[uf.name] = uf.getvalue()

        total_files = len(st.session_state.converter_files)
        heic_count = sum(
            1 for n in st.session_state.converter_files
            if Path(n).suffix.lower() in (".heic", ".heif")
        )
        other_count = total_files - heic_count

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Files", total_files)
        c2.metric("HEIC/HEIF", heic_count)
        c3.metric("Other (passthrough)", other_count)

        with st.expander(f"📋 View all files ({total_files})", expanded=False):
            for i, name in enumerate(sorted(st.session_state.converter_files.keys()), 1):
                ext = Path(name).suffix.lower()
                icon = "🖼️" if ext in (".heic", ".heif") else "📄"
                size_kb = len(st.session_state.converter_files[name]) / 1024
                st.text(f"{i}. {icon} {name} ({size_kb:.0f} KB)")

    # Convert
    if st.session_state.converter_files:
        st.header("🚀 Convert")
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            convert_clicked = st.button(
                f"⚡ Convert All ({len(st.session_state.converter_files)} files)",
                width="stretch",
                type="primary",
            )

        with col_btn2:
            if st.button("🗑️ Clear All Files", width="stretch"):
                st.session_state.converter_files = {}
                st.session_state.converted_zip = None
                st.rerun()

        if convert_clicked:
            files = st.session_state.converter_files
            total = len(files)

            progress_bar = st.progress(0, text="Starting conversion...")
            status_text = st.empty()
            stats_container = st.empty()

            start_time = time.time()
            converted_count = 0
            failed_count = 0
            total_bytes_original = 0
            total_bytes_converted = 0

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                sorted_names = sorted(files.keys())

                for i, filename in enumerate(sorted_names):
                    img_bytes = files[filename]
                    total_bytes_original += len(img_bytes)

                    stem = Path(filename).stem
                    out_name = f"{stem}.jpg"
                    # Handle duplicate names
                    if out_name in zf.namelist():
                        idx = 1
                        while f"{stem}_{idx}.jpg" in zf.namelist():
                            idx += 1
                        out_name = f"{stem}_{idx}.jpg"

                    try:
                        img = Image.open(io.BytesIO(img_bytes))

                        # Convert to RGB (required for JPEG)
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGB")
                        elif img.mode == "L":
                            img = img.convert("RGB")

                        # Optional resize
                        if max_width > 0 and img.width > max_width:
                            ratio = max_width / img.width
                            new_height = int(img.height * ratio)
                            img = img.resize((max_width, new_height), Image.LANCZOS)

                        # Save as JPEG
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=jpg_quality, optimize=True)
                        jpg_bytes = buf.getvalue()
                        total_bytes_converted += len(jpg_bytes)

                        zf.writestr(out_name, jpg_bytes)
                        converted_count += 1

                    except Exception as e:
                        logger.error("Failed to convert %s: %s", filename, e)
                        failed_count += 1

                    # Update progress
                    pct = (i + 1) / total
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = (total - i - 1) / speed if speed > 0 else 0

                    progress_bar.progress(
                        pct,
                        text=f"{i + 1}/{total} | "
                        f"Elapsed: {elapsed:.0f}s | "
                        f"ETA: {remaining:.0f}s | "
                        f"Speed: {speed:.1f} files/s",
                    )
                    status_text.text(f"Converting: {filename}")

            elapsed = time.time() - start_time
            status_text.empty()
            progress_bar.empty()

            st.session_state.converted_zip = zip_buffer.getvalue()

            # Show results
            stats_container.success(
                f"✅ Conversion complete!\n\n"
                f"- **Converted:** {converted_count}\n"
                f"- **Failed:** {failed_count}\n"
                f"- **Time:** {elapsed:.1f}s\n"
                f"- **Speed:** {converted_count / elapsed:.1f} files/s\n"
                f"- **Original size:** {total_bytes_original / (1024*1024):.1f} MB\n"
                f"- **Converted size:** {total_bytes_converted / (1024*1024):.1f} MB\n"
                f"- **Compression:** {(1 - total_bytes_converted/total_bytes_original) * 100:.1f}% smaller"
                if total_bytes_original > 0 else ""
            )

            # Clear uploaded files from memory after conversion
            st.session_state.converter_files = {}
            st.info("🧹 Uploaded files cleared from memory after conversion.")

    # Download
    if st.session_state.converted_zip:
        st.header("📥 Download")
        st.download_button(
            label="⬇️ Download All JPGs (ZIP)",
            data=st.session_state.converted_zip,
            file_name="converted_images.zip",
            mime="application/zip",
            width="stretch",
            type="primary",
        )

    st.divider()
    st.caption("HEIC/HEIF → JPG Converter | Powered by Pillow + pillow-heif")

    # STOP here — don't render the rest of the main app
    st.stop()


# ===========================================================================
# MAIN PAGE
# ===========================================================================
st.title("📊 AI Image Data Extractor")
st.caption("Extract table data from images using OpenAI Vision")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
st.header("📷 Upload Images")
uploaded_files = st.file_uploader(
    "Upload JPG, JPEG, PNG, WEBP, HEIC, or HEIF images",
    type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
    accept_multiple_files=True,
    help="You can upload multiple images at once.",
)

if uploaded_files:
    for uf in uploaded_files:
        if uf.name not in st.session_state.uploaded_files_map:
            st.session_state.uploaded_files_map[uf.name] = uf.getvalue()

    st.info(f"📁 {len(st.session_state.uploaded_files_map)} images loaded into memory.")

    # Show filenames
    with st.expander(f"View uploaded files ({len(st.session_state.uploaded_files_map)})"):
        for i, name in enumerate(sorted(st.session_state.uploaded_files_map.keys()), 1):
            status = "✅" if name in get_completed_filenames(st.session_state.processing_state) else "⏳"
            st.text(f"{status} {i}. {name}")

# ---------------------------------------------------------------------------
# Processing state
# ---------------------------------------------------------------------------
state = st.session_state.processing_state
all_filenames = sorted(st.session_state.uploaded_files_map.keys())
completed = get_completed_filenames(state)
pending = [f for f in all_filenames if f not in completed]

# ---------------------------------------------------------------------------
# Processing Statistics
# ---------------------------------------------------------------------------
st.header("📈 Processing Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Images", len(all_filenames))
col2.metric("Processed", len(completed))
col3.metric("Failed", state.get("total_failed", 0))
col4.metric("Total Rows", state.get("total_rows", 0))

# Progress bar
if all_filenames:
    progress = len(completed) / len(all_filenames)
    st.progress(progress, text=f"{len(completed)} / {len(all_filenames)} images processed")

# Cost estimate
tracker: CostTracker = st.session_state.cost_tracker
tracker.usd_to_inr = usd_to_inr
tracker.model = model
cost_summary = tracker.get_summary()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Input Tokens", f"{cost_summary['input_tokens']:,}")
c2.metric("Output Tokens", f"{cost_summary['output_tokens']:,}")
c3.metric("Est. Cost (USD)", f"${cost_summary['estimated_usd']:.4f}")
c4.metric("Est. Cost (INR)", f"₹{cost_summary['estimated_inr']:.2f}")

# ---------------------------------------------------------------------------
# Processing controls
# ---------------------------------------------------------------------------
st.header("🚀 Processing")

if st.session_state.processing:
    st.warning("⏳ Processing is currently running...")
else:
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        if st.button("▶️ Process Current Image", width="stretch"):
            if not pending:
                st.info("No pending images.")
            elif not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                st.session_state.processing = True
                st.rerun()

    with pc2:
        if st.button("📸 Process Selected Images", width="stretch"):
            if not pending:
                st.info("No pending images.")
            elif not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                st.session_state["_process_mode"] = "selected"
                st.session_state.processing = True
                st.rerun()

    with pc3:
        if st.button("🔄 Process All Images", width="stretch"):
            if not pending:
                st.info("No pending images.")
            elif not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                st.session_state["_process_mode"] = "all"
                st.session_state.processing = True
                st.rerun()

    # Retry failed
    failed_names = get_failed_filenames(state)
    if failed_names:
        if st.button(
            f"🔁 Retry Failed Images ({len(failed_names)})",
            width="stretch",
        ):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                # Reset failed images to pending
                for fn in failed_names:
                    update_image_state(state, fn, status="pending", error=None)
                save_state(state)
                st.session_state["_process_mode"] = "retry"
                st.session_state.processing = True
                st.rerun()


# ---------------------------------------------------------------------------
# Run processing
# ---------------------------------------------------------------------------
if st.session_state.processing:
    process_mode = st.session_state.get("_process_mode", "all")

    if process_mode == "all":
        images_to_process = pending
    elif process_mode == "retry":
        images_to_process = list(get_failed_filenames(state))
    else:  # "selected" — default to pending
        images_to_process = pending[:10]  # Process first 10 for "selected"

    if not images_to_process:
        st.success("✅ All images already processed!")
        st.session_state.processing = False
    else:
        st.info(f"🔄 Processing {len(images_to_process)} images...")

        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()
        stats_container = st.empty()
        errors_container = st.container()

        client = AsyncOpenAI(api_key=api_key)

        if not state.get("started_at"):
            state["started_at"] = time.time()
            state["columns"] = columns
            state["custom_instructions"] = custom_instructions

        # Mutable counters (avoids nonlocal at module scope)
        counters = {
            "completed": 0,
            "failed": 0,
            "total_rows": 0,
        }
        start_time = time.time()

        def progress_callback(done: int, total: int):
            pct = done / total if total > 0 else 0
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / speed if speed > 0 else 0
            progress_bar.progress(
                pct,
                text=f"Images: {done}/{total} | "
                f"Elapsed: {elapsed:.0f}s | "
                f"ETA: {remaining:.0f}s | "
                f"Speed: {speed:.1f} img/s",
            )

        async def run_processing():
            for i, fname in enumerate(images_to_process):
                status_text.text(f"Processing {fname} ({i + 1}/{len(images_to_process)})...")
                update_image_state(state, fname, status="processing")

                image_bytes = st.session_state.uploaded_files_map.get(fname)
                if image_bytes is None:
                    update_image_state(
                        state, fname, status="failed", error="File not in memory"
                    )
                    counters["failed"] += 1
                    save_failed_log(fname, "File not in memory", 0)
                    continue

                result = await extract_single_image(
                    client, model, image_bytes, fname, columns, custom_instructions
                )

                if result["error"]:
                    update_image_state(
                        state,
                        fname,
                        status="failed",
                        error=result["error"],
                    )
                    counters["failed"] += 1
                    save_failed_log(fname, result["error"], 0)
                    logger.error("Failed %s: %s", fname, result["error"])
                else:
                    # Validate rows
                    validated, needs_review_rows, warnings = validate_batch(
                        result["rows"],
                        columns,
                        expected_rows=expected_rows,
                        tolerance=tolerance,
                    )

                    has_review = len(needs_review_rows) > 0
                    status = "needs_review" if has_review else "completed"

                    update_image_state(
                        state,
                        fname,
                        status=status,
                        rows_extracted=len(validated),
                        result=validated,
                        needs_review=has_review,
                    )
                    save_result_json(fname, validated)
                    counters["total_rows"] += len(validated)
                    counters["completed"] += 1

                # Record cost
                tracker.record(
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                )

                # Save state periodically
                if (i + 1) % 5 == 0 or i == len(images_to_process) - 1:
                    save_state(state)

                progress_callback(i + 1, len(images_to_process))

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_processing())
            loop.close()
        except Exception as e:
            st.error(f"Processing error: {e}")
            logger.error("Processing error: %s", e)

        # Final update
        save_state(state)
        st.session_state.processing = False
        st.session_state.pop("_process_mode", None)

        elapsed = time.time() - start_time
        status_text.empty()
        stats_container.success(
            f"✅ Processing complete!\n\n"
            f"- **Images processed:** {counters['completed'] + counters['failed']} / {len(images_to_process)}\n"
            f"- **Rows extracted:** {counters['total_rows']}\n"
            f"- **Successful:** {counters['completed']}\n"
            f"- **Failed:** {counters['failed']}\n"
            f"- **Elapsed:** {elapsed:.1f}s\n"
            f"- **Speed:** {len(images_to_process) / elapsed:.2f} images/s"
        )
        st.rerun()


# ===========================================================================
# RESULTS
# ===========================================================================
st.header("📋 Results")

all_results = get_all_results(state)
needs_review = get_needs_review_results(state)
failed_list = get_failed_images(state)

if all_results:
    df = pd.DataFrame(all_results)

    # Display columns
    display_cols = [c for c in ["source_image", "source_row"] + columns if c in df.columns]
    # Remove internal cols from display
    for drop in ("needs_review", "review_reasons"):
        if drop in display_cols:
            display_cols.remove(drop)

    # Search / filter
    search = st.text_input("🔍 Search results", "")
    if search:
        mask = df.apply(
            lambda row: search.lower() in str(row.values).lower(), axis=1
        )
        df_filtered = df[mask]
    else:
        df_filtered = df

    st.dataframe(
        df_filtered[display_cols],
        width="stretch",
        height=400,
    )

    st.caption(f"Showing {len(df_filtered)} / {len(df_filtered)} rows")
else:
    st.info("No results yet. Upload images and start processing.")

# ---------------------------------------------------------------------------
# Needs Review
# ---------------------------------------------------------------------------
if needs_review:
    st.header("⚠️ Needs Review")
    df_review = pd.DataFrame(needs_review)
    review_cols = [c for c in ["source_image", "source_row"] + columns + ["review_reasons"] if c in df_review.columns]
    if "review_reasons" in df_review.columns:
        df_review["review_reasons"] = df_review["review_reasons"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else str(x)
        )
    st.dataframe(df_review[review_cols], width="stretch", height=300)

# ---------------------------------------------------------------------------
# Failed Images
# ---------------------------------------------------------------------------
if failed_list:
    st.header("❌ Failed Images")
    df_failed = pd.DataFrame(failed_list)
    st.dataframe(df_failed, width="stretch")

# ===========================================================================
# DOWNLOADS
# ===========================================================================
st.header("📥 Download Results")

dl1, dl2 = st.columns(2)

with dl1:
    if st.button("📊 Download Excel", width="stretch"):
        if all_results:
            summary = {
                "total_images": len(all_filenames),
                "processed": len(completed),
                "failed": state.get("total_failed", 0),
                "needs_review_count": len(needs_review),
                "total_rows": state.get("total_rows", 0),
                "cost_usd": tracker.estimated_usd,
                "cost_inr": tracker.estimated_inr,
            }
            excel_bytes = build_excel(
                all_results, needs_review, failed_list, summary, columns
            )
            st.download_button(
                label="⬇️ Save extracted_data.xlsx",
                data=excel_bytes,
                file_name="extracted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("No data to export yet.")

with dl2:
    if st.button("📄 Download CSV", width="stretch"):
        if all_results:
            csv_str = build_csv(all_results, columns)
            st.download_button(
                label="⬇️ Save extracted_data.csv",
                data=csv_str,
                file_name="extracted_data.csv",
                mime="text/csv",
            )
        else:
            st.info("No data to export yet.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"AI Image Data Extractor | Model: {model} | "
    f"Session started: {state.get('started_at', 'N/A')}"
)
