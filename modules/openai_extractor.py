"""OpenAI Vision API extractor for image data extraction."""

import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from modules.image_processor import (
    load_and_encode_image,
    build_image_content,
    get_media_type,
)

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """You are an EXTREMELY precise OCR data extraction system. Your ONLY job is to read the table in this image and extract every single row with ZERO errors.

## CRITICAL OCR ACCURACY RULES:
1. Read each digit and character INDIVIDUALLY. Do NOT guess or assume.
2. For phone/mobile numbers (Indian format): MUST be EXACTLY 10 digits. If you see 11 digits, one digit is duplicated or wrong — identify and remove it. If you see 9 digits, one digit is missing — check carefully.
3. For amounts/numbers: Read EACH digit one by one from left to right. Double-check by re-reading the digit. Common mistakes: 0↔6, 1↔7, 3↔8, 5↔6, 9↔0. AVOID these errors.
4. For city names: Use TITLE CASE (e.g., "MUMBAI" not "mumbai", "ERNAKULAM" not "Ernakulam"). Match exactly what is written.
5. For person names: Preserve EXACT spelling as written. Do NOT add or remove letters. Do NOT guess similar-sounding names.
6. Preserve ALL digits in numbers exactly as they appear — no truncation, no extra digits.
7. Preserve dates EXACTLY as visible (DD-MM-YYYY format).
8. If a cell is truly unreadable, use null — but TRY HARD before giving up.
9. Do NOT guess or hallucinate values. If unsure, use null.
10. Read the ENTIRE image from top to bottom. Extract EVERY visible row. Do NOT skip any row.
11. Do NOT merge two rows into one. Each row in the table = one row in the output.
12. Maintain the original row order from top to bottom.
13. Return ONLY valid JSON. No explanations, no markdown, no extra text.

## DOUBLE-CHECK BEFORE RETURNING:
- Re-read every phone number digit-by-digit
- Re-read every amount digit-by-digit
- Verify city names are spelled correctly
- Verify person names match exactly
- Count total rows — make sure you have all of them

The user wants to extract these columns: {columns}

{custom_instructions}

Return a JSON array of objects. Each object represents one row.
Example format:
[
  {{"Number": "1234567890", "City": "MUMBAI", "Name": "John Doe", "Date": "03-01-2022", "Amount": "15000"}},
  ...
]

Return ONLY the JSON array. Nothing else."""


def build_extraction_prompt(
    columns: list[str], custom_instructions: str = ""
) -> str:
    """Build the extraction prompt with the given columns."""
    col_text = ", ".join(columns) if columns else "all visible columns"
    extra = ""
    if custom_instructions:
        extra = f"Additional instructions from user:\n{custom_instructions}"
    return EXTRACTION_PROMPT_TEMPLATE.format(columns=col_text, custom_instructions=extra)


def parse_json_response(text: str) -> list[dict[str, Any]]:
    """Parse JSON from the model response, handling common formatting issues."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON response: %s", text[:200])
    return []


async def extract_single_image(
    client: AsyncOpenAI,
    model: str,
    image_bytes: bytes,
    filename: str,
    columns: list[str],
    custom_instructions: str = "",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Extract data from a single image using OpenAI Vision API.

    Returns a dict with keys: rows, error, input_tokens, output_tokens.
    """
    b64 = load_and_encode_image(image_bytes, filename)
    if b64 is None:
        return {
            "rows": [],
            "error": "Failed to load or encode image",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    prompt = build_extraction_prompt(columns, custom_instructions)
    image_content = build_image_content(b64, filename)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content,
            ],
        }
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            # Build API kwargs — newer models reject temperature and max_tokens
            kwargs = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": 16000,
            }
            # Only set temperature for models that support it
            # Models like gpt-4.1*, gpt-5* only accept default temperature=1
            _newer_models = ("gpt-4.1", "gpt-5")
            if not any(m in model for m in _newer_models):
                kwargs["temperature"] = 0.0

            response = await client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            usage = response.usage

            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            rows = parse_json_response(content)
            if not rows:
                return {
                    "rows": [],
                    "error": "Model returned empty or unparseable response",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

            return {
                "rows": rows,
                "error": None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except RateLimitError as e:
            wait = (2**attempt) * 2  # exponential backoff: 2, 4, 8 seconds
            logger.warning(
                "Rate limit hit for %s (attempt %d/%d), waiting %ds",
                filename,
                attempt + 1,
                max_retries,
                wait,
            )
            last_error = f"Rate limit: {e}"
            await asyncio.sleep(wait)

        except APITimeoutError as e:
            wait = (2**attempt) * 3
            logger.warning(
                "Timeout for %s (attempt %d/%d), waiting %ds",
                filename,
                attempt + 1,
                max_retries,
                wait,
            )
            last_error = f"Timeout: {e}"
            await asyncio.sleep(wait)

        except APIError as e:
            wait = (2**attempt) * 2
            logger.warning(
                "API error for %s (attempt %d/%d): %s",
                filename,
                attempt + 1,
                max_retries,
                e,
            )
            last_error = f"API error: {e}"
            await asyncio.sleep(wait)

        except Exception as e:
            logger.error("Unexpected error for %s: %s", filename, e)
            return {
                "rows": [],
                "error": f"Unexpected: {e}",
                "input_tokens": 0,
                "output_tokens": 0,
            }

    return {
        "rows": [],
        "error": f"Failed after {max_retries} retries. Last error: {last_error}",
        "input_tokens": 0,
        "output_tokens": 0,
    }


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    image_data_list: list[tuple[str, bytes]],
    columns: list[str],
    custom_instructions: str = "",
    concurrency: int = 5,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """Process a batch of images concurrently.

    Args:
        client: AsyncOpenAI client.
        model: Model name.
        image_data_list: List of (filename, image_bytes) tuples.
        columns: Column names to extract.
        custom_instructions: Additional user instructions.
        concurrency: Max concurrent API calls.
        progress_callback: Optional callback(completed, total) for progress updates.

    Returns:
        List of result dicts, one per image.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    completed_count = 0

    async def process_one(idx: int, filename: str, image_bytes: bytes):
        nonlocal completed_count
        async with semaphore:
            result = await extract_single_image(
                client, model, image_bytes, filename, columns, custom_instructions
            )
            result["filename"] = filename
            result["index"] = idx
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, len(image_data_list))
            return result

    tasks = [
        process_one(i, fname, data)
        for i, (fname, data) in enumerate(image_data_list)
    ]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

    # Sort by original index
    results.sort(key=lambda r: r["index"])
    return results
