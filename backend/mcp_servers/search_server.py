"""MCP tools for DuckDuckGo search and safe public-page fetching.

The URL checks are defence-in-depth, not a substitute for production network
egress controls (which should block private and metadata IP ranges).
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentFlow Search Server")
MAX_RESULTS = 8
MAX_PAGE_BYTES = 900_000
MAX_PAGE_TEXT = 24_000
MAX_PAGE_LINKS = 120


@mcp.tool()
def search_web(query: str, max_results: int = 8) -> str:
    """Search the public web and return titles, URLs and snippets."""
    query = (query or "").strip()
    if not query:
        return "Search failed: query is empty."

    target_count = max(1, min(int(max_results), MAX_RESULTS))
    results = None

    # For news queries, try DuckDuckGo news first
    is_news = any(w in query.lower() for w in ("news", "headline", "breaking", "update", "latest"))
    if is_news:
        try:
            results = list(DDGS().news(query, region="in-en", max_results=target_count))
        except Exception:
            try:
                results = list(DDGS().news(query, max_results=target_count))
            except Exception:
                results = None

    if not results:
        try:
            results = list(DDGS().text(query, region="in-en", max_results=target_count))
        except Exception:
            results = None

    if not results:
        try:
            results = list(DDGS().text(query, max_results=target_count))
        except Exception as exc:
            return f"Search failed: {type(exc).__name__}: {exc}"

    if not results:
        return "No search results found."

    output = []
    for index, item in enumerate(results, start=1):
        href = item.get("href") or item.get("url", "")
        if not href:
            continue
        output.append(
            f"Result {index}\n"
            f"Title: {item.get('title', 'N/A')}\n"
            f"URL: {href}\n"
            f"Description: {item.get('body') or item.get('snippet', 'N/A')}"
        )
    return "\n\n".join(output) if output else "No usable search results found."


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host or host == "localhost" or host.endswith(".localhost"):
        raise ValueError("Invalid or local hostname.")
    try:
        addresses = socket.getaddrinfo(
            host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"Hostname could not be resolved: {host}") from exc
    if not addresses:
        raise ValueError("Hostname did not resolve to a public address.")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private, reserved, or non-public IP addresses are not allowed.")


class PageTextExtractor(HTMLParser):
    """Extract visible text and anchor links without executing page content."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg", "template"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[dict] = []
        # Metadata handlers append to this list. Initialize it before parsing.
        self.published_dates: list[str] = []
        self.in_title = False
        self.skip_stack: list[str] = []
        self.anchor_stack: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
            value = (attrs.get("content") or "").strip()
            # Publication metadata only: modified timestamps can make old
            # stories look new, so they are deliberately excluded.
            if value and key in {
                "article:published_time", "og:article:published_time", "datepublished",
                "pubdate", "parsely-pub-date", "sailthru.date", "date",
            }:
                self.published_dates.append(value)
        if tag == "time" and attrs.get("datetime"):
            marker = " ".join((attrs.get("itemprop", ""), attrs.get("class", ""), attrs.get("data-testid", ""))).lower()
            # News sites commonly expose the publication time in a time tag.
            # Avoid explicit modified/updated labels.
            if not any(word in marker for word in ("modified", "updated", "last-update")):
                self.published_dates.append(attrs["datetime"].strip())
        if tag == "title":
            self.in_title = True
        if tag in self.SKIP_TAGS:
            self.skip_stack.append(tag)
        if tag == "a" and attrs.get("href"):
            href = urljoin(self.base_url, attrs["href"].strip())
            href, _fragment = urldefrag(href)
            parsed = urlparse(href)
            if parsed.scheme in {"http", "https"} and parsed.hostname:
                label = attrs.get("aria-label") or attrs.get("title") or ""
                self.anchor_stack.append({"text": [label] if label else [], "url": href})
            else:
                self.anchor_stack.append({"text": [], "url": ""})

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag == "a" and self.anchor_stack:
            anchor = self.anchor_stack.pop()
            if anchor["url"]:
                label = " ".join(" ".join(anchor["text"]).split()).strip()
                self.links.append({"text": label[:220], "url": anchor["url"]})
        if tag in self.SKIP_TAGS:
            for index in range(len(self.skip_stack) - 1, -1, -1):
                if self.skip_stack[index] == tag:
                    del self.skip_stack[index]
                    break

    def handle_data(self, data):
        value = " ".join((data or "").split())
        if not value:
            return
        if self.in_title:
            self.title_parts.append(value)
        elif not self.skip_stack:
            self.text_parts.append(value)
        for anchor in self.anchor_stack:
            anchor["text"].append(value)


class SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def extract_jsonld_publication_dates(html: str) -> list[str]:
    """Read datePublished from JSON-LD Article/NewsArticle metadata."""
    dates: list[str] = []
    for match in re.finditer(
        r"(?is)<script\b[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        html or "",
    ):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            continue
        stack = payload if isinstance(payload, list) else [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                kind = item.get("@type", [])
                kinds = [kind] if isinstance(kind, str) else kind
                value = item.get("datePublished")
                if value and (not kinds or any("article" in str(k).lower() or "news" in str(k).lower() for k in kinds)):
                    dates.append(str(value).strip())
                for key in ("@graph", "mainEntity", "mainEntityOfPage"):
                    child = item.get(key)
                    if isinstance(child, (dict, list)):
                        stack.append(child)
    return dates


@mcp.tool()
def fetch_page(url: str) -> str:
    """Fetch a public HTML/text page and return title, final URL, text and links."""
    try:
        validate_public_url(url)
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        opener = build_opener(SafeRedirectHandler)
        with opener.open(request, timeout=18) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Unsupported webpage content type: {content_type or 'unknown'}"
            raw = response.read(MAX_PAGE_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
            body = raw.decode(charset, errors="replace")
            final_url = response.geturl()

        if "text/plain" in content_type:
            return (
                "PAGE TITLE:\nPlain text page\n"
                f"PAGE URL:\n{final_url}\n"
                f"PAGE CONTENT:\n{body[:MAX_PAGE_TEXT]}\n"
                "PAGE LINKS:\n"
            )

        parser = PageTextExtractor(final_url)
        parser.feed(body)
        published_dates = list(parser.published_dates)
        if not published_dates:
            published_dates = extract_jsonld_publication_dates(body)
        page_text = "\n".join(parser.text_parts)
        links, seen_links = [], set()
        for link in parser.links:
            href = link["url"]
            if href in seen_links:
                continue
            seen_links.add(href)
            links.append(link)
            if len(links) >= MAX_PAGE_LINKS:
                break
        link_text = "\n".join(
            f"LINK: {item['text'] or '(no label)'} | {item['url']}" for item in links
        )
        return (
            f"PAGE TITLE:\n{' '.join(parser.title_parts) or 'Title not found'}\n"
            f"PAGE URL:\n{final_url}\n"
            f"PAGE PUBLISHED DATE:\n{published_dates[0] if published_dates else 'Not found'}\n"
            f"PAGE CONTENT:\n{page_text[:MAX_PAGE_TEXT]}\n"
            f"PAGE LINKS:\n{link_text}"
        )
    except Exception as exc:
        return f"Page fetching failed: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
