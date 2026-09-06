#!/usr/bin/env python3
"""Generate one original, attributed Markdown article per news topic.

The script uses Google News RSS search feeds for discovery and an OpenAI-compatible
Chat Completions endpoint for paraphrasing. It intentionally supplies only feed
metadata and snippets to the model; generated copy must link back to the original
sources and must not reproduce source text verbatim.

Examples:
  python3 tools/generate_articles.py --dry-run
  OPENAI_API_KEY=... python3 tools/generate_articles.py
  python3 tools/generate_articles.py --location "Anchorage Alaska" --hours 72
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOPICS = [
    "community events",
    "education",
    "nature",
    "food",
    "entertainment",
    "games and consoles",
    "automotive",
    "technology",
    "finance",
    "comics",
]

DEFAULT_OUTPUT_DIR = Path("generated_articles")
DEFAULT_MODEL = os.getenv("ARTICLE_MODEL", "gpt-5-mini")
DEFAULT_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")


@dataclass
class Story:
    title: str
    url: str
    source: str
    published: str
    summary: str

    def as_prompt_block(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Publisher: {self.source}\n"
            f"Published: {self.published}\n"
            f"URL: {self.url}\n"
            f"Feed summary: {self.summary}"
        )


def slugify(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "article"


def strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def text(element: ET.Element | None, tag: str) -> str:
    if element is None:
        return ""
    found = element.find(tag)
    return (found.text or "").strip() if found is not None else ""


def fetch_rss(topic: str, location: str, hours: int, limit: int, timeout: int) -> list[Story]:
    # Try the local search first, then broaden only when a topic has no usable leads.
    queries = [
        f"{location} {topic} news",
        f"{location} {topic}",
        f"{topic} news",
    ]
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    stories: list[Story] = []
    seen_urls: set[str] = set()
    for query in queries:
        params = urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        url = f"https://news.google.com/rss/search?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": "AnchorageViewArticleBot/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        root = ET.fromstring(payload)
        for item in root.findall("./channel/item"):
            title = strip_html(text(item, "title"))
            link = text(item, "link")
            description = strip_html(text(item, "description"))
            pub_date = text(item, "pubDate")
            source = text(item, "source") or "Google News source"
            if not title or not link or link in seen_urls:
                continue
            try:
                parsed_date = dt.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                parsed_date = dt.datetime.now(dt.timezone.utc)
            if parsed_date < cutoff:
                continue
            seen_urls.add(link)
            stories.append(Story(title, link, source, pub_date or "Unknown date", description[:1200]))
            if len(stories) >= limit:
                return stories
        if stories:
            return stories
    return stories


def request_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AnchorageViewArticleBot/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json_object(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("Model response did not contain a JSON object")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Model response was not a JSON object")
    return data


def generate_article(topic: str, stories: list[Story], location: str, model: str, base_url: str, api_key: str, timeout: int) -> dict:
    source_block = "\n\n---\n\n".join(story.as_prompt_block() for story in stories)
    system = (
        "You are the senior editor of a local digital news magazine. Create an original "
        "news-style article from the supplied recent story leads. Do not copy sentences, "
        "quotes, distinctive phrasing, or structure from the sources. Do not invent facts, "
        "names, numbers, dates, or quotes. Attribute every factual claim to a source link. "
        "If the feeds are thin or conflicting, say so clearly instead of filling gaps. "
        "Return JSON only."
    )
    user = f"""Topic: {topic}\nGeographic focus: {location}\n\nWrite one publishable article that synthesizes the strongest recent lead(s) below. Keep it between 450 and 700 words. Use an informative headline, a one-sentence deck, and 4 to 6 paragraphs. Include a short 'Why it matters' sentence at the end. Add a Sources list with the exact URLs provided. The article must be a paraphrased original report, not a summary that closely tracks any single source.\n\nReturn exactly this JSON shape:\n{{\n  \"headline\": \"...\",\n  \"deck\": \"...\",\n  \"body_paragraphs\": [\"...\"],\n  \"why_it_matters\": \"...\",\n  \"sources\": [{{\"title\": \"...\", \"publisher\": \"...\", \"url\": \"...\"}}]\n}}\n\nRecent story leads:\n{source_block}"""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_completion_tokens": 1800,
    }
    response = request_json(f"{base_url}/chat/completions", payload, api_key, timeout)
    content = response["choices"][0]["message"]["content"]
    article = extract_json_object(content)
    validate_article(article, stories)
    return article


def validate_article(article: dict, stories: list[Story]) -> None:
    required = ("headline", "deck", "body_paragraphs", "why_it_matters", "sources")
    missing = [key for key in required if not article.get(key)]
    if missing:
        raise ValueError(f"Generated article missing fields: {', '.join(missing)}")
    if not isinstance(article["body_paragraphs"], list) or len(article["body_paragraphs"]) < 3:
        raise ValueError("Generated article needs at least three body paragraphs")
    if not isinstance(article["sources"], list) or not article["sources"]:
        raise ValueError("Generated article needs at least one source")
    allowed_urls = {story.url for story in stories}
    for source in article["sources"]:
        if source.get("url") not in allowed_urls:
            raise ValueError("Generated source URL was not present in the feed leads")


def render_markdown(topic: str, location: str, article: dict) -> str:
    today = dt.date.today().isoformat()
    lines = [
        f"# {article['headline']}",
        "",
        f"*{article['deck']}*",
        "",
        f"**Topic:** {topic.title()}  ",
        f"**Dateline:** {location}  ",
        f"**Generated:** {today}",
        "",
        "---",
        "",
    ]
    lines.extend(f"{paragraph}\n" for paragraph in article["body_paragraphs"])
    lines.extend(["", f"**Why it matters:** {article['why_it_matters']}", "", "## Sources", ""])
    for source in article["sources"]:
        lines.append(f"- [{source.get('publisher', 'Source')}: {source.get('title', 'Article')}]({source['url']})")
    lines.extend([
        "",
        "> This is an original paraphrased article generated from the linked source leads. "
        "It is not affiliated with, endorsed by, or a replacement for the cited publishers.",
        "",
    ])
    return "\n".join(lines)


def write_article(output_dir: Path, topic: str, content: str, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slugify(topic)}.md"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to replace it")
    path.write_text(content, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", default=os.getenv("NEWS_LOCATION", "Anchorage Alaska"), help="Location added to each feed search")
    parser.add_argument("--hours", type=int, default=96, help="Only use stories published within this many hours")
    parser.add_argument("--stories-per-topic", type=int, default=5, help="Number of feed leads supplied to the model per topic")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Scan feeds and print leads without calling the model or writing files")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause between topic requests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.dry_run and not api_key:
        print("ERROR: set OPENAI_API_KEY or use --dry-run", file=sys.stderr)
        return 2

    failures: list[str] = []
    for index, topic in enumerate(TOPICS):
        print(f"[{index + 1}/{len(TOPICS)}] Scanning {topic}...", flush=True)
        try:
            stories = fetch_rss(topic, args.location, args.hours, args.stories_per_topic, args.timeout)
            if not stories:
                raise RuntimeError("No recent stories found in the selected time window")
            print(f"  found {len(stories)} lead(s)")
            if args.dry_run:
                for story in stories:
                    print(f"  - {story.title} ({story.source})")
                continue
            article = generate_article(topic, stories, args.location, args.model, args.base_url, api_key, args.timeout)
            path = write_article(args.output_dir, topic, render_markdown(topic, args.location, article), args.overwrite)
            print(f"  wrote {path}")
            if index < len(TOPICS) - 1:
                time.sleep(max(0.0, args.pause))
        except Exception as exc:  # Keep other topics running if one feed fails.
            failures.append(f"{topic}: {exc}")
            print(f"  ERROR: {exc}", file=sys.stderr)

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("\nCompleted all requested topics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
