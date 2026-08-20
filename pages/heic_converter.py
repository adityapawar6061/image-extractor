"""Bulk HEIC/HEIF to JPG Converter - Streamlit Page."""

import io
import logging
import os
import time
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HEIC/HEIF to JPG Converter",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    jpg_quality = st.slider(
        "JPG Quality",
        min_value=1,
        max_value=100,
        value=95,
        help="Higher = better quality, larger file size.",
    )

    max_width = st.number_input(
        "Max width (px, 0 = no limit)",
        min_value=0,
        value=0,
        step=100,
        help="Resize images if wider than this. Set 0 to keep original size.",
    )

    st.divider()
    st.page_link("app.py", label="🏠 Back to Main App", icon="🏠", use_container_width=True)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🔄 HEIC/HEIF → JPG Converter")
st.caption("Convert thousands of HEIC/HEIF images to JPG in one click")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.header("📷 Upload Images")

uploaded_files = st.file_uploader(
    "Upload HEIC, HEIF, JPG, JPEG, PNG, or WEBP images",
    type=["heic", "heif", "jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="You can upload 2000+ images at once.",
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} files loaded")

    # Store files in session state to avoid re-uploading
    if "converter_files" not in st.session_state:
        st.session_state.converter_files = {}

    for uf in uploaded_files:
        if uf.name not in st.session_state.converter_files:
            st.session_state.converter_files[uf.name] = uf.getvalue()

    total_files = len(st.session_state.converter_files)

    # Show stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Files", total_files)
    col2.metric("HEIC/HEIF", sum(
        1 for n in st.session_state.converter_files
        if Path(n).suffix.lower() in (".heic", ".heif")
    ))
    col3.metric("Other (passthrough)", sum(
        1 for n in st.session_state.converter_files
        if Path(n).suffix.lower() not in (".heic", ".heif")
    ))

    # File list
    with st.expander(f"📋 View all files ({total_files})", expanded=False):
        for i, name in enumerate(sorted(st.session_state.converter_files.keys()), 1):
            ext = Path(name).suffix.lower()
            icon = "🖼️" if ext in (".heic", ".heif") else "📄"
            size_kb = len(st.session_state.converter_files[name]) / 1024
            st.text(f"{i}. {icon} {name} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Convert button
# ---------------------------------------------------------------------------
st.header("🚀 Convert")

if st.session_state.get("converter_files"):
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        convert_clicked = st.button(
            f"⚡ Convert All ({len(st.session_state.converter_files)} files)",
            use_container_width=True,
            type="primary",
        )

    with col_btn2:
        if st.button("🗑️ Clear All Files", use_container_width=True):
            st.session_state.converter_files = {}
            st.session_state.pop("converted_zip", None)
            st.rerun()

    if convert_clicked:
        files = st.session_state.converter_files
        total = len(files)

        progress_bar = st.progress(0, text="Starting conversion...")
        status_text = st.empty()
        stats_container = st.empty()

        start_time = time.time()
        converted_count = 0
        skipped_count = 0
        failed_count = 0
        total_bytes_original = 0
        total_bytes_converted = 0

        # Process in memory — build a ZIP
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            sorted_names = sorted(files.keys())

            for i, filename in enumerate(sorted_names):
                img_bytes = files[filename]
                total_bytes_original += len(img_bytes)

                stem = Path(filename).stem
                out_name = f"{stem}.jpg"
                # Handle duplicate names
                if zf.testzip() is not None or out_name in zf.namelist():
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

        # Store ZIP in session state
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

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
if st.session_state.get("converted_zip"):
    st.header("📥 Download")
    st.download_button(
        label=f"⬇️ Download All JPGs ({len(st.session_state.get('converter_files', {}))} files)",
        data=st.session_state.converted_zip,
        file_name="converted_images.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("HEIC/HEIF → JPG Converter | Powered by Pillow + pillow-heif")
