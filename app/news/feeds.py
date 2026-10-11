"""
Latest headlines from a fixed list of public tech / AI news feeds (RSS 2.0 or Atom).

Read-only: plain HTTPS GETs with `requests` (already a dependency) and the standard
library's XML parser. No API key and no new package.

Safety:
- Feed text is written by outsiders, so each headline and summary is sanitized and
  wrapped in <untrusted_news_item> tags (like emails), with '<' and '>' escaped.
- Responses are capped (MAX_FEED_BYTES) and time out (FETCH_TIMEOUT). Documents that
  declare entities are refused outright (no entity-expansion tricks), and ElementTree
  never fetches external resources.
- Only http(s) links are passed on.

AGENT_NEWS_FEEDS (optional) replaces the default list: comma-separated feed URLs,
each optionally prefixed with a name ("The Verge|https://www.theverge.com/rss/index.xml").
"""

from __future__ import annotations

import html
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

from app.agent.untrusted import inline_untrusted, sanitize_text


logger = logging.getLogger(__name__)

DEFAULT_FEEDS = (
    ("Hacker News", "https://hnrss.org/frontpage"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
)
TAG = "untrusted_news_item"
FETCH_TIMEOUT = 10
MAX_FEED_BYTES = 3_000_000
MAX_ITEMS = 25
MAX_SUMMARY_CHARS = 300
MAX_AGE_DAYS = 4
USER_AGENT = "Socio/1.0 (personal news digest)"

_ATOM = "{http://www.w3.org/2005/Atom}"
_TAGS = re.compile(r"(?s)<[^>]*>")
_HIDDEN = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>|<!--.*?-->")
# Short topic words such as "AI" match as whole words only ("ai" is inside "said").
_WORD = re.compile(r"[A-Za-z0-9+#.]+")


@dataclass
class NewsItem:
    source: str
    title: str
    summary: str
    link: str
    published: datetime | None


def configured_feeds() -> list[tuple[str, str]]:
    raw = os.getenv("AGENT_NEWS_FEEDS", "").strip()
    if not raw:
        return list(DEFAULT_FEEDS)
    feeds = []
    for entry in raw.split(","):
        name, _, url = entry.strip().rpartition("|")
        url = url.strip()
        if url.startswith(("https://", "http://")):
            feeds.append((name.strip() or urlparse(url).hostname or url, url))
    return feeds or list(DEFAULT_FEEDS)


def _plain(text: str | None) -> str:
    """Feed HTML -> one line of plain text."""

    text = _HIDDEN.sub(" ", text or "")
    text = html.unescape(_TAGS.sub(" ", text))
    return " ".join(sanitize_text(text).split())


def _safe_link(link: str | None) -> str:
    link = (link or "").strip()
    return link if urlparse(link).scheme in {"http", "https"} and " " not in link else ""


def _date(text: str | None) -> datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        moment = parsedate_to_datetime(text)        # RSS: RFC 822
    except (TypeError, ValueError, IndexError):
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))  # Atom: ISO 8601
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _text(element, *names: str) -> str:
    for name in names:
        found = element.find(name)
        if found is not None and (found.text or "").strip():
            return found.text
    return ""


def parse_feed(data: bytes, source: str) -> list[NewsItem]:
    """RSS 2.0 or Atom bytes -> items. ValueError for anything else (or a document with entities)."""

    if b"<!ENTITY" in data or b"<!entity" in data:
        raise ValueError(f"{source}: feed declares XML entities; refused.")
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise ValueError(f"{source}: not a valid feed ({exc}).") from exc

    items = []
    for entry in root.iter("item"):  # RSS 2.0
        items.append(NewsItem(
            source=source,
            title=_plain(_text(entry, "title")),
            summary=_plain(_text(entry, "description")),
            link=_safe_link(_text(entry, "link")),
            published=_date(_text(entry, "pubDate", "{http://purl.org/dc/elements/1.1/}date")),
        ))
    for entry in root.iter(f"{_ATOM}entry"):  # Atom
        link = ""
        for candidate in entry.findall(f"{_ATOM}link"):
            if candidate.get("rel", "alternate") == "alternate":
                link = candidate.get("href", "")
                break
        items.append(NewsItem(
            source=source,
            title=_plain(_text(entry, f"{_ATOM}title")),
            summary=_plain(_text(entry, f"{_ATOM}summary", f"{_ATOM}content")),
            link=_safe_link(link),
            published=_date(_text(entry, f"{_ATOM}updated", f"{_ATOM}published")),
        ))
    if not items and root.tag not in {"rss", f"{_ATOM}feed"} and root.find("channel") is None:
        raise ValueError(f"{source}: not an RSS or Atom feed.")
    return [item for item in items if item.title]


def fetch_feed(name: str, url: str) -> list[NewsItem]:
    with requests.get(url, timeout=FETCH_TIMEOUT, stream=True, headers={"User-Agent": USER_AGENT}) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > MAX_FEED_BYTES:
                raise ValueError(f"{name}: feed is larger than {MAX_FEED_BYTES} bytes; skipped.")
    return parse_feed(bytes(data), name)


def _matches(item: NewsItem, words: list[str]) -> bool:
    text = f" {item.title} {item.summary} ".lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) for word in words)


def topic_words(topic: str) -> list[str]:
    words = [word.lower().strip(".") for word in _WORD.findall(topic or "")]
    words = [word for word in words if len(word) >= 2 and word not in {"news", "the", "and", "top", "latest", "tech", "updates", "about"}]
    if "ai" in words:
        words += ["artificial intelligence", "llm", "llms", "openai", "anthropic", "gemini", "chatgpt", "machine learning"]
    return words


def latest_news(topic: str = "", max_items: int = 10, *, fetch=None, now: datetime | None = None) -> dict:
    """
    Recent items from every configured feed, newest first, de-duplicated. With a topic,
    only matching items (or, if none match, the latest overall with a note saying so).
    """

    fetch = fetch or fetch_feed
    max_items = max(1, min(int(max_items or 10), MAX_ITEMS))
    now = now or datetime.now(timezone.utc)
    feeds = configured_feeds()

    collected: list[NewsItem] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=min(6, len(feeds))) as pool:
        futures = {pool.submit(fetch, name, url): name for name, url in feeds}
        for future, name in futures.items():
            try:
                collected.extend(future.result())
            except Exception as exc:  # one dead feed never stops the others
                logger.warning("News feed %s failed: %s", name, exc)
                failed.append(name)

    if not collected:
        raise ValueError("I couldn't reach any of the news sites right now. Please try again later.")

    oldest = now - timedelta(days=MAX_AGE_DAYS)
    recent = [item for item in collected if item.published is None or item.published >= oldest] or collected
    recent.sort(key=lambda item: item.published or oldest, reverse=True)

    seen, unique = set(), []
    for item in recent:
        key = item.link or item.title.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    note = ""
    words = topic_words(topic)
    if words:
        matching = [item for item in unique if _matches(item, words)]
        if matching:
            unique = matching
        else:
            note = f"No recent headline mentions {topic!r}; these are the latest tech headlines instead."

    return {
        "items": [news_view(item) for item in unique[:max_items]],
        "sources": [name for name, _ in feeds if name not in failed],
        "sources_unreachable": failed,
        "note": note,
    }


def news_view(item: NewsItem) -> dict:
    """One headline as a tool result: the link and date are checked fields, the text is wrapped."""

    summary = item.summary
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + " …"
    body = (item.title + ("\n" + summary if summary else "")).replace("<", "&lt;").replace(">", "&gt;")
    return {
        "source": item.source,
        "published": item.published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if item.published else "",
        "link": item.link,
        "news": f'<{TAG} source="{inline_untrusted(item.source)}">\n{body}\n</{TAG}>',
    }
