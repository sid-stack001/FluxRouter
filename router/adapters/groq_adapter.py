"""
GroqAdapter — wraps the Groq Python SDK for FluxRouter.

Reads GROQ_API_KEY from the .env file via python-dotenv.
Returns a ModelResponse dataclass with all per-call metadata needed
by the CascadeController.

Retry policy (Phase 4):
    HTTP 429 (rate-limited): exponential backoff with jitter, max 3 retries.
        wait = min(base * 2^attempt, cap) + uniform(0, jitter_ms) / 1000
        Respects Retry-After header when Groq provides it.
    All other errors: re-raised immediately (no retry).

Note on avg_logprob:
    At startup, GroqAdapter probes whether the target model supports logprobs
    by sending a minimal test request with logprobs=True.  If the model
    returns a 400 error (all Groq-hosted Llama models as of 2026-07),
    avg_logprob is set to None on every ModelResponse and logprob-based
    escalation is disabled.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import BadRequestError, RateLimitError, Groq

# Load .env from the project root (two levels up from this file).
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(_env_path)

# Models known to NOT support logprobs on Groq (400 on request).
_LOGPROB_UNSUPPORTED_MODELS: frozenset[str] = frozenset({
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
})

# ── Retry constants ───────────────────────────────────────────────────────────
_RETRY_MAX_ATTEMPTS: int   = 3       # retries after the first failure (total 4 calls max)
_RETRY_BASE_S:       float = 1.0     # initial backoff in seconds
_RETRY_FACTOR:       float = 2.0     # exponential multiplier
_RETRY_CAP_S:        float = 30.0    # maximum computed wait (before jitter)
_RETRY_JITTER_MS:    int   = 250     # uniform jitter ceiling in milliseconds


@dataclass
class ModelResponse:
    """Everything returned from a single model call."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    avg_logprob: float | None = None
    # True only when the Groq API actually returned logprob data.
    logprobs_available: bool = False


class GroqAdapter:
    """
    Thin wrapper around the Groq Python SDK with retry-on-429 logic.

    Probes logprob support once at construction time; all subsequent
    complete() calls use the discovered capability.

    Usage:
        adapter = GroqAdapter()
        response = adapter.complete("What is 2+2?", model="llama-3.1-8b-instant")
        print(response.logprobs_available)  # False for most Groq models
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Set it in .env, as an environment variable, "
                "or run 'fluxrouter init'."
            )
        self._client = Groq(api_key=key)
        self._logprobs_supported: dict[str, bool] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self, prompt: str, model: str, max_tokens: int | None = None
    ) -> ModelResponse:
        """
        Send a single-turn completion request to Groq with retry-on-429.

        Retry policy:
            - HTTP 429 → exponential backoff (base 1s, ×2 per attempt, cap 30s)
                         + uniform jitter [0, 250ms]
                         + respect Retry-After header when present
            - Other errors → raised immediately (no retry)

        Args:
            prompt:     The user query text.
            model:      Groq model identifier, e.g. "llama-3.1-8b-instant".
            max_tokens: Optional token limit for generation.

        Returns:
            ModelResponse with text, token counts, latency, and metadata.
        """
        want_logprobs = self._check_logprob_support(model)
        kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if want_logprobs:
            kwargs["logprobs"] = True


        t0 = time.perf_counter()
        response = self._call_with_retry(kwargs, model)
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        usage  = response.usage
        avg_logprob, logprobs_available = self._extract_logprob(choice)

        return ModelResponse(
            text=choice.message.content or "",
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            avg_logprob=avg_logprob,
            logprobs_available=logprobs_available,
        )

_MAX_RETRY_WAIT_S = 10.0  # Max seconds to wait for a 429 retry before giving up to prevent multi-minute hangs


class GroqAdapter:
    """
    Adapter for the Groq API (OpenAI-compatible).
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Set it in .env, as an environment variable, "
                "or run 'fluxrouter init'."
            )
        self._client = Groq(api_key=key)
        self._logprobs_supported: dict[str, bool] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self, prompt: str, model: str, max_tokens: int | None = None
    ) -> ModelResponse:
        want_logprobs = self._check_logprob_support(model)
        kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if want_logprobs:
            kwargs["logprobs"] = True

        t0 = time.perf_counter()
        response = self._call_with_retry(kwargs, model)
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        usage  = response.usage
        avg_logprob, logprobs_available = self._extract_logprob(choice)

        return ModelResponse(
            text=choice.message.content or "",
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            avg_logprob=avg_logprob,
            logprobs_available=logprobs_available,
        )

    # ── Retry logic ───────────────────────────────────────────────────────────

    def _call_with_retry(self, kwargs: dict, model: str):
        """
        Execute self._client.chat.completions.create(**kwargs) with 429 retry.

        Attempts: 1 initial + up to _RETRY_MAX_ATTEMPTS retries.
        Non-429 errors bubble up immediately.
        If wait time > _MAX_RETRY_WAIT_S (e.g. daily TPD limits), raises immediately.
        """
        for attempt in range(_RETRY_MAX_ATTEMPTS + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except BadRequestError:
                if kwargs.get("logprobs"):
                    self._logprobs_supported[model] = False
                    kwargs.pop("logprobs", None)
                    return self._client.chat.completions.create(**kwargs)
                raise
            except RateLimitError as exc:
                wait_s = self._compute_wait(exc, attempt)
                if attempt >= _RETRY_MAX_ATTEMPTS or wait_s > _MAX_RETRY_WAIT_S:
                    raise RateLimitError(
                        f"Groq 429 rate limit reached for {model} (wait time {wait_s:.1f}s exceeds limit).",
                        response=getattr(exc, "response", None),
                        body=getattr(exc, "body", None),
                    ) from exc

                print(
                    f"[groq_adapter] 429 rate-limited "
                    f"(attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS}). "
                    f"Waiting {wait_s:.2f}s before retry…"
                )
                time.sleep(wait_s)


    @staticmethod
    def _compute_wait(exc: RateLimitError, attempt: int) -> float:
        """
        Compute wait time for a 429 retry.

        Priority:
          1. Retry-After header (seconds) if provided by Groq.
          2. Exponential backoff: min(base * factor^attempt, cap) + jitter.
        """
        # Try to extract Retry-After from the exception headers.
        retry_after: float | None = None
        try:
            headers = getattr(exc, "response", None)
            if headers is not None:
                ra = headers.headers.get("retry-after") or headers.headers.get("Retry-After")
                if ra:
                    retry_after = float(ra)
        except Exception:  # noqa: BLE001
            pass

        if retry_after is not None:
            jitter = random.uniform(0, _RETRY_JITTER_MS) / 1000
            return retry_after + jitter

        computed = min(_RETRY_BASE_S * (_RETRY_FACTOR ** attempt), _RETRY_CAP_S)
        jitter   = random.uniform(0, _RETRY_JITTER_MS) / 1000
        return computed + jitter

    # ── Logprob helpers ───────────────────────────────────────────────────────

    def _check_logprob_support(self, model: str) -> bool:
        if model in _LOGPROB_UNSUPPORTED_MODELS:
            return False
        if model in self._logprobs_supported:
            return self._logprobs_supported[model]
        supported = self._probe_logprob_support(model)
        self._logprobs_supported[model] = supported
        return supported

    def _probe_logprob_support(self, model: str) -> bool:
        try:
            self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                logprobs=True,
            )
            return True
        except BadRequestError:
            return False

    @staticmethod
    def _extract_logprob(choice) -> tuple[float | None, bool]:
        lp = choice.logprobs
        if lp is None:
            return None, False
        content = getattr(lp, "content", None)
        if not content:
            return None, False
        token_lps = [t.logprob for t in content if t.logprob is not None]
        if not token_lps:
            return None, False
        return sum(token_lps) / len(token_lps), True
