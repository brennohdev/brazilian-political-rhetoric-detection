import json
import time

import httpx
from loguru import logger

from src.classification.base import BaseClassifier
from src.classification.prompt_builder import PromptBuilder
from src.schemas.config import ModelConfig
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment


class LLaMAClassifier(BaseClassifier):
    """Few-shot classifier using LLaMA 3.x via Ollama.

    Connects to a local Ollama instance for inference.
    Uses the same prompt structure as GPT-4o for fair comparison.
    """

    def __init__(
        self,
        config: ModelConfig,
        prompt_builder: PromptBuilder,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self._config = config
        self._prompt_builder = prompt_builder
        self._ollama_url = ollama_url.rstrip("/")
        self._model_name = config.quantization or "llama3.1:8b-instruct-q4_0"
        self._max_retries = 3
        self._client = httpx.Client(timeout=120.0)

    @property
    def model_id(self) -> str:
        return "llama3"

    def classify(self, segment: SpeechSegment) -> list[Prediction]:
        """Classify a single segment using LLaMA via Ollama.

        Args:
            segment: Speech segment to classify.

        Returns:
            List of Prediction objects for detected techniques.

        Raises:
            Exception: If all retries are exhausted.
        """
        system_msg = self._prompt_builder.get_system_message()
        user_msg = self._prompt_builder.build_prompt(segment)

        response_text = self._call_ollama(system_msg, user_msg)
        predictions = self._parse_response(response_text, segment)
        return predictions

    def _call_ollama(self, system_msg: str, user_msg: str) -> str:
        """Call Ollama API with retry logic.

        Uses the /api/chat endpoint for chat-style interaction.
        """
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }

        for attempt in range(self._max_retries):
            try:
                response = self._client.post(
                    f"{self._ollama_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "[]")

            except httpx.ConnectError as e:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"Ollama connection failed (attempt {attempt+1}): {e}. "
                    f"Retrying in {wait}s"
                )
                time.sleep(wait)
                if attempt == self._max_retries - 1:
                    raise

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    wait = 5
                    logger.warning(f"Ollama busy, waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise

            except httpx.TimeoutException:
                wait = 2 ** attempt
                logger.warning(f"Ollama timeout (attempt {attempt+1}). Retrying in {wait}s")
                time.sleep(wait)
                if attempt == self._max_retries - 1:
                    raise

        return "[]"

    def _parse_response(
        self, response_text: str, segment: SpeechSegment
    ) -> list[Prediction]:
        """Parse JSON response into Prediction objects.

        LLaMA responses can be messier than GPT-4o — handles:
        - Extra text before/after JSON
        - Markdown code blocks
        - Partial JSON
        """
        text = response_text.strip()

        # Try to extract JSON array from response
        # First: strip markdown code blocks
        if "```" in text:
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        # Second: find the JSON array in the text
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx : end_idx + 1]
        else:
            # No array found — might be empty response
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse JSON from LLaMA for {segment.segment_id}: "
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

            # Calculate offsets
            start_offset = segment.text.find(span)
            if start_offset == -1:
                start_offset = 0
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
                continue

        return predictions

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
