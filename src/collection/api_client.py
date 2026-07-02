import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from src.exceptions import CollectionError
from src.schemas.config import CollectionConfig


@dataclass
class APIResponse:
    """Structured response from the Chamber API."""

    data: list[dict[str, Any]]
    has_next: bool
    next_page: int | None


class ChamberAPIClient:
    """HTTP client for the Chamber of Deputies API.

    Single Responsibility: handles HTTP communication only (rate limiting, retries).
    Does NOT handle persistence or business logic.
    """

    BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"

    def __init__(self, config: CollectionConfig) -> None:
        self._config = config
        self._client = httpx.Client(timeout=30.0)

    def fetch_speeches(
        self,
        deputy_id: int,
        start_date: str,
        end_date: str,
        page: int = 1,
        items_per_page: int = 100,
    ) -> APIResponse:
        """Fetch a page of speeches for a deputy.

        Handles rate limiting with configurable delay.
        Retries on transient errors with exponential backoff.
        """
        url = f"{self.BASE_URL}/deputados/{deputy_id}/discursos"
        params = {
            "dataInicio": start_date,
            "dataFim": end_date,
            "pagina": page,
            "itens": items_per_page,
            "ordem": "ASC",
            "ordenarPor": "dataHoraInicio",
        }

        response_data = self._request_with_retry(url, params)
        dados = response_data.get("dados", [])
        links = response_data.get("links", [])

        has_next = any(link.get("rel") == "next" for link in links)
        next_page = page + 1 if has_next else None

        return APIResponse(data=dados, has_next=has_next, next_page=next_page)

    def fetch_deputies(self, legislature_id: int = 57) -> list[dict[str, Any]]:
        """Fetch the list of deputies for a given legislature.

        Args:
            legislature_id: Legislature number (57 = current, 2023-2027).
        """
        url = f"{self.BASE_URL}/deputados"
        params = {
            "idLegislatura": legislature_id,
            "itens": 600, 
            "ordem": "ASC",
            "ordenarPor": "nome",
        }
        response_data = self._request_with_retry(url, params)
        return response_data.get("dados", [])

    def _request_with_retry(
        self, url: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic and rate limiting.

        Retries up to max_retries times with exponential backoff.
        Waits rate_limit_delay between requests.
        """
        max_retries = self._config.max_retries

        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(self._config.rate_limit_delay)
                response = self._client.get(url, params=params)

                if response.status_code == 429:
                    # Rate limited — wait longer
                    wait_time = 2**attempt * self._config.rate_limit_delay
                    logger.warning(
                        f"Rate limited (429). Waiting {wait_time:.1f}s "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if attempt == max_retries:
                    raise CollectionError(
                        f"HTTP error after {max_retries} retries: {e.response.status_code}",
                        context={"url": url, "status_code": e.response.status_code},
                    ) from e
                wait_time = 2**attempt
                logger.warning(
                    f"HTTP {e.response.status_code} on attempt {attempt}/{max_retries}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == max_retries:
                    raise CollectionError(
                        f"Network error after {max_retries} retries: {e}",
                        context={"url": url, "error": str(e)},
                    ) from e
                wait_time = 2**attempt
                logger.warning(
                    f"Network error on attempt {attempt}/{max_retries}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        # Should not reach here, but just in case
        raise CollectionError("Exhausted retries", context={"url": url})

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "ChamberAPIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
