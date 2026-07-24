"""Sentiment lexical léger sur les actualités récentes.

L'actualité ne pèse que 5 % du score final. Une panne de la source RSS laisse
donc un score neutre et ne bloque jamais un signal technique.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests

LOGGER = logging.getLogger(__name__)

POSITIVE = {
    "hausse",
    "record",
    "contrat",
    "gagne",
    "croissance",
    "relève",
    "rehausse",
    "supérieur",
    "beat",
    "upgrade",
    "growth",
    "profit",
    "partenariat",
}
NEGATIVE = {
    "baisse",
    "alerte",
    "enquête",
    "sanction",
    "dégrade",
    "abaisse",
    "inférieur",
    "perte",
    "miss",
    "downgrade",
    "fraude",
    "recul",
}


@dataclass(frozen=True)
class NewsResult:
    score: float
    headlines: tuple[str, ...]


class NewsClient:
    def __init__(self, timeout_seconds: int = 10, ttl_seconds: int = 1800):
        self.timeout_seconds = timeout_seconds
        self.ttl_seconds = ttl_seconds
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "mon-bot-trading-v4/4.0"
        self._cache: dict[str, tuple[float, NewsResult]] = {}

    def score(self, company_name: str) -> NewsResult:
        cached = self._cache.get(company_name)
        if cached and time.monotonic() - cached[0] < self.ttl_seconds:
            return cached[1]
        try:
            query = quote_plus(f'"{company_name}" action when:1d')
            url = f"https://news.google.com/rss/search?q={query}&hl=fr&gl=FR&ceid=FR:fr"
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            headlines = tuple(
                str(node.text).strip()
                for node in root.findall("./channel/item/title")[:5]
                if node.text
            )
            result = self._from_headlines(headlines)
        except Exception:
            LOGGER.warning(
                "Actualités indisponibles pour %s : score neutre appliqué.",
                company_name,
                exc_info=True,
            )
            result = NewsResult(50.0, ())
        self._cache[company_name] = (time.monotonic(), result)
        return result

    @staticmethod
    def _from_headlines(headlines: tuple[str, ...]) -> NewsResult:
        positive = 0
        negative = 0
        for headline in headlines:
            words = set(
                headline.lower()
                .replace("’", " ")
                .replace("'", " ")
                .replace("-", " ")
                .split()
            )
            positive += len(words & POSITIVE)
            negative += len(words & NEGATIVE)
        score = max(20.0, min(80.0, 50.0 + 8 * (positive - negative)))
        return NewsResult(score, headlines[:3])
