import json
import time

from loguru import logger
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from src.classification.base import BaseClassifier
from src.classification.prompt_builder import PromptBuilder
from src.schemas.config import ModelConfig
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment


class GPT4oClassifier(BaseClassifier):
    """Few-shot classifier using OpenAI GPT-4o or GPT-4o-mini API.

    Sends each segment with taxonomy definitions and format instructions.
    Parses JSON responses into Prediction objects.
    """

    def __init__(
        self,
        config: ModelConfig,
        prompt_builder: PromptBuilder,
        api_key: str | None = None,
        model_name: str = "gpt-4o",
    ) -> None:
        self._config = config
        self._prompt_builder = prompt_builder
        self._client = OpenAI(api_key=api_key)  # Uses OPENAI_API_KEY env var if None
        self._model_name = model_name
        self._token_usage: dict[str, int] = {"prompt": 0, "completion": 0}
        self._max_retries = 3

    @property
    def model_id(self) -> str:
        if "mini" in self._model_name:
            return "gpt4o_mini"
        return "gpt4o"

    def classify(self, segment: SpeechSegment) -> list[Prediction]:
        """Classify a single segment using GPT-4o.

        Args:
            segment: Speech segment to classify.

        Returns:
            List of Prediction objects for detected techniques.

        Raises:
            Exception: If all retries are exhausted.
        """
        system_msg = self._prompt_builder.get_system_message()
        user_msg = self._prompt_builder.build_prompt(segment)

        response_text = self._call_api(system_msg, user_msg)
        predictions = self._parse_response(response_text, segment)
        return predictions

    def _call_api(self, system_msg: str, user_msg: str) -> str:
        """Call OpenAI API with retry logic.

        Returns the assistant's response text.
        """
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )

                # Track token usage
                if response.usage:
                    self._token_usage["prompt"] += response.usage.prompt_tokens
                    self._token_usage["completion"] += response.usage.completion_tokens

                return response.choices[0].message.content or "[]"

            except RateLimitError as e:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                if attempt == self._max_retries - 1:
                    raise

            except (APITimeoutError, APIError) as e:
                wait = 2 ** attempt
                logger.warning(f"API error: {e}. Retrying in {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                if attempt == self._max_retries - 1:
                    raise

        return "[]"  # Should not reach here

    def _parse_response(
        self, response_text: str, segment: SpeechSegment
    ) -> list[Prediction]:
        """Parse JSON response into Prediction objects.

        Handles common issues:
        - Markdown code blocks around JSON
        - Malformed JSON (returns empty list)
        - Missing fields (skips that detection)
        """
        # Strip markdown code block if present
        text = response_text.strip()
        if text.startswith("```"):
            # Remove opening and closing ```
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse JSON from GPT-4o for {segment.segment_id}: "
                f"{text[:100]}..."
            )
            return []

        if not isinstance(data, list):
            return []

        predictions: list[Prediction] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            technique = item.get("technique", "")
            span = item.get("span", "")
            confidence = item.get("confidence", 1.0)

            if not technique or not span:
                continue

            # Calculate offsets from span in segment text
            start_offset = segment.text.find(span)
            if start_offset == -1:
                start_offset = 0  # Fallback if exact span not found
            end_offset = start_offset + len(span)

            try:
                prediction = Prediction(
                    technique=technique,
                    span=span,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    confidence=float(confidence),
                    model_id=self.model_id,
                    segment_id=segment.segment_id,
                )
                predictions.append(prediction)
            except Exception:
                continue  # Skip invalid predictions

        return predictions

    @property
    def token_usage(self) -> dict[str, int]:
        """Total token usage across all API calls."""
        return self._token_usage
