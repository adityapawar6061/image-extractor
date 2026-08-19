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

EXTRACTION_PROMPT_TEMPLATE = """You are a precise data extraction system. Your ONLY job is to read the table in this image and extract every single row of data.

CRITICAL RULES:
1. Read the ENTIRE image from top to bottom.
2. Extract EVERY visible row. Do NOT skip any row.
3. Do NOT merge two rows into one.
4. Preserve numbers EXACTLY as they appear.
5. Preserve names EXACTLY as visible.
6. Preserve dates EXACTLY as visible.
7. Preserve amounts EXACTLY as visible.
8. If a value is missing, unclear, or unreadable, use null.
9. Do NOT guess or hallucinate values.
10. Maintain the original row order from top to bottom.
11. Return ONLY valid JSON. No explanations, no markdown, no extra text.

The user wants to extract these columns: {columns}

{custom_instructions}

Return a JSON array of objects. Each object represents one row.
Example format:
[
  {{"Number": "12345", "City": "MUMBAI", "Name": "John Doe", "Date": "01-01-2024", "Amount": "15000"}},
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
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=16000,
            )

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
