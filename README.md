# AI Image Data Extractor

Extract tabular data from images using OpenAI Vision API. Built with Python and Streamlit.

## Features

- **Vision-powered extraction** — reads tables from images using GPT-4o / GPT-4o-mini
- **Batch processing** — handles thousands of images with concurrency control
- **Resume capability** — survives crashes, restarts, and internet failures
- **Row validation** — flags rows that need human review
- **Excel & CSV export** — multi-sheet workbooks with full audit trail
- **Cost tracking** — estimates USD and INR costs in real time

---

## Prerequisites

1. **Install Python 3.10+**

   Download from https://www.python.org/downloads/  
   Make sure to check **"Add Python to PATH"** during install.

2. **Verify installation**

   ```cmd
   python --version
   ```

   You should see `Python 3.10.x` or higher.

---

## Setup

### Step 1: Create a virtual environment

```cmd
cd image-extractor
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 3: (Optional) Configure `.env`

Copy the example file and add your API key:

```cmd
copy .env.example .env
```

Open `.env` in any text editor and paste your key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

> The key is only used locally and never committed to Git.

### Step 4: Start the application

```cmd
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Usage Guide

### 1. Enter API Key

In the left sidebar, under **OPENAI SETTINGS**, paste your OpenAI API key into the password field.

### 2. Test API Key

Click the **🔑 Test API Key** button. You'll see:

- ✅ **API key is valid!** — the connection works
- ❌ **API key test failed** — check the key or your network

### 3. Upload One Image

1. Click **Browse files** on the main page
2. Select a single JPG/PNG/WEBP image
3. Verify the image appears in the file list

### 4. Extract One Image

Click **▶️ Process Current Image**. Wait for extraction to complete. The results appear in the **📋 Results** table.

### 5. Verify the ~38-Row Result

Check that the extracted rows match what you see in the image. If the count differs from your expected 38 rows by more than the tolerance (±2), the image will appear in the **⚠️ Needs Review** section.

### 6. Test 10 Images

Upload ~10 images and click **📸 Process Selected Images**. Review the batch results, then download the Excel file.

### 7. Process the Complete 2,000-Image Folder

1. Upload all images (or place them in the upload area)
2. Set **Concurrent requests** to 5–10 in the sidebar
3. Click **🔄 Process All Images**
4. Monitor progress — elapsed time, ETA, and speed are shown

### 8. Resume Interrupted Jobs

If processing stops (crash, restart, internet failure):

1. Re-run `streamlit run app.py`
2. Click **📂 Resume Previous Job** in the sidebar
3. Click **🔄 Process All Images** — only unfinished images are processed

### 9. Download Excel

Click **📊 Download Excel** at the bottom. The workbook contains:

| Sheet | Content |
|-------|---------|
| Extracted Data | All successfully extracted rows |
| Needs Review | Rows flagged by validation |
| Failed Images | Images that errored out |
| Processing Summary | Totals, costs, timestamps |

### 10. Retry Failed Images

Click **🔁 Retry Failed Images (N)** in the sidebar to retry only the images that failed.

### 11. Monitor Estimated API Cost

The **💰 COST ESTIMATION** section in the sidebar shows:

- Input/output tokens used
- Estimated cost in USD
- Estimated cost in INR (default rate: 95 ₹/$)

Adjust the exchange rate in the sidebar if needed.

---

## Project Structure

```
image-extractor/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env.example              # Example environment config
│
├── modules/
│   ├── __init__.py
│   ├── openai_extractor.py   # OpenAI Vision API calls
│   ├── image_processor.py    # Image loading & encoding
│   ├── validator.py          # Data validation rules
│   ├── excel_exporter.py     # Excel/CSV export
│   ├── state_manager.py      # Persistence & resume state
│   └── cost_tracker.py       # Token & cost tracking
│
├── data/
│   ├── uploads/              # Uploaded images (session)
│   ├── results/              # Per-image JSON results
│   └── failed/               # Failed image logs
│
└── logs/
    └── processing.log        # Application logs
```

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Model | gpt-4o-mini | Vision model for extraction |
| Concurrent requests | 5 | Max simultaneous API calls |
| Expected rows per image | 38 | For validation |
| Tolerance | ±2 | Acceptable row count deviation |
| USD → INR | 95 | Exchange rate for cost estimate |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key test failed" | Check key at platform.openai.com, ensure no spaces |
| Rate limit errors | Lower concurrent requests or wait |
| Low row count | Add custom instructions, check image quality |
| App crashes on large batch | Lower concurrency, ensure stable internet |
| Resume not working | Check `processing_state.json` exists in project root |

---

## License

Internal use only. Do not commit API keys to version control.
