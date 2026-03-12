import os
import time
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Rate Limiter ---
# Ensures minimum gap between API calls to avoid hitting quota
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 3.0  # seconds between requests (max ~20 req/min)


def _rate_limit():
    """Wait if needed to respect rate limits."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def generate_text(prompt: str, max_output_tokens: int = 600, temperature: float = 0.2) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY on server environment.")

    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001").strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Business Analyst",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }

    # Retry logic with exponential backoff for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        _rate_limit()

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=90,
            )
        except requests.RequestException as e:
            raise ValueError(f"Network error: {e}")

        if response.status_code == 429:
            wait = (attempt + 1) * 5  # 5s, 10s, 15s
            time.sleep(wait)
            continue

        if response.status_code >= 400:
            raise ValueError(
                f"OpenRouter API error {response.status_code}: {response.text[:400]}"
            )

        data = response.json()

        # Extract response text
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenRouter returned no choices.")

        text = (choices[0].get("message") or {}).get("content", "").strip()
        if text:
            return text

        raise ValueError("OpenRouter returned empty content.")

    raise ValueError("Rate limited after multiple retries. Please wait a moment and try again.")
