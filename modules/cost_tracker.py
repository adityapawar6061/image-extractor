"""Cost tracker for estimating OpenAI API usage and costs."""

from dataclasses import dataclass, field


# Approximate pricing per 1M tokens (as of gpt-4o / gpt-4o-mini)
# These are rough estimates; actual billing may vary.
MODEL_COSTS_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


@dataclass
class CostTracker:
    """Tracks token usage and estimates cost across a batch of API calls."""

    model: str = "gpt-4o-mini"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    images_processed: int = 0
    usd_to_inr: float = 95.0

    def record(
        self, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Record token usage from a single API response."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.images_processed += 1

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def _get_rates(self) -> tuple[float, float]:
        """Return (input_rate_per_token, output_rate_per_token)."""
        rates = MODEL_COSTS_PER_1M.get(
            self.model, MODEL_COSTS_PER_1M["gpt-4o-mini"]
        )
        return rates["input"] / 1_000_000, rates["output"] / 1_000_000

    @property
    def estimated_usd(self) -> float:
        in_rate, out_rate = self._get_rates()
        return self.total_input_tokens * in_rate + self.total_output_tokens * out_rate

    @property
    def estimated_inr(self) -> float:
        return self.estimated_usd * self.usd_to_inr

    def get_summary(self) -> dict:
        return {
            "model": self.model,
            "images_processed": self.images_processed,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_usd": round(self.estimated_usd, 4),
            "estimated_inr": round(self.estimated_inr, 2),
        }
