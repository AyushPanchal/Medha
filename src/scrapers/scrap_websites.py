import os
import re
import time
import pathlib
import urllib.parse
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ---- Config stuff ----
OUTPUT_DIR = pathlib.Path("../../data/processed/markdowns")   # where final .md files land
URL_LIST_FILE = r"../../data/raw/links/manual_collection_webpages.txt"  # file with URLs to scrape
REQUEST_TIMEOUT = 1           # fail request if it takes longer than this
REQUEST_DELAY_SEC = 0.8       # pause between requests (be nice to servers)
MAX_RETRIES = 2               # retry count if something goes wrong
USER_AGENT = "SVNIT-CSE-Scraper/1.0 (+https://svnit.ac.in) python-requests"  # custom header for requests

# make sure output folder exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def url_to_filename(url: str) -> str:
    """
    Turn a URL into a safe markdown filename.
    Basically: strip extensions, handle index pages, replace weird chars with '-'.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"

    # if path ends with '/', call it index
    if path.endswith("/"):
        base = "index"
    else:
        base = os.path.basename(path)
        if base.lower() in ("index", "index.php", "index.html", ""):
            base = "index"
        else:
            # kill .php/.html/etc
            base = re.sub(r"\.(php|html|htm)$", "", base, flags=re.IGNORECASE)

    # clean up filenames (avoid slashes, keep only alphanumeric and '-')
    parts = [p for p in urllib.parse.unquote(path).split("/") if p]
    if len(parts) > 1:
        parent = re.sub(r"[^A-Za-z0-9]+", "-", parts[-2]).strip("-")
        base_norm = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-")
        filename_core = f"{parent}-{base_norm}" if parent else base_norm
    else:
        filename_core = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-") or "index"

    if not filename_core:
        filename_core = "index"

    return f"{filename_core}.md"


def fetch(url: str) -> requests.Response:
    """
    Try to get a webpage with retries.
    Returns response if okay, otherwise throws.
    """
    last_err = None
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if 200 <= resp.status_code < 300:
                return resp   # all good
            elif resp.status_code in (403, 404):
                return resp   # permanent fail, don’t bother retrying
            else:
                last_err = Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = e

        time.sleep(REQUEST_DELAY_SEC)  # wait before retry

    # after retries still dead
    raise last_err if last_err else RuntimeError("Unknown fetch error")


def extract_main_fragment(html: str) -> str:
    """
    Try to grab the 'main' part of the page.
    If no <main>, look for common wrappers (#main, .content, etc).
    If still nothing, fallback to body.
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("main")

    if main is None:
        main = soup.select_one('[role="main"], #main, #content, .main, .content')
    if main is None:
        main = soup.body or soup

    # throw away junk tags we don’t need
    for tag in main.find_all(["script", "style", "noscript"]):
        tag.decompose()
    return str(main)


def html_to_markdown(html: str, base_url: str) -> str:
    """
    Take HTML, spit out markdown.
    Keeps inline images, normalizes headings, etc.
    """
    return md(
        html,
        heading_style="ATX",
        strip=["script", "style"],
        bullets="*",
        escape_asterisks=False,
        escape_underscores=False,
        keep_inline_images=True,
        base_url=base_url,
    )


def process_url(url: str) -> tuple[str, str]:
    """
    Full pipeline for a URL:
    - Fetch
    - Check if it's HTML
    - Extract main content
    - Convert to markdown
    - Add a header with metadata
    Returns (filename, markdown string)
    """
    resp = fetch(url)

    if resp.status_code != 200:
        return url_to_filename(url), f"# Fetch error\n\nStatus: {resp.status_code}\nURL: {url}\n"

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        return url_to_filename(url), f"# Non-HTML content\n\nURL: {url}\nContent-Type: {content_type}\n"

    fragment = extract_main_fragment(resp.text)
    md_text = html_to_markdown(fragment, base_url=url)

    header = f"<!-- Source: {url} | scope: <main> -->\n\n"
    return url_to_filename(url), header + md_text.strip() + "\n"


def main():
    """
    Reads URLs, makes sure they belong to svnit.ac.in, scrapes them,
    and saves markdown files to disk. Errors go into separate error-*.md files.
    """
    if not os.path.exists(URL_LIST_FILE):
        raise FileNotFoundError(f"Missing {URL_LIST_FILE}. Create it with one URL per line.")

    # load URL list (skip empty lines and comments)
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    # sanity check: only within our domain
    for u in urls:
        host = urllib.parse.urlparse(u).netloc
        if not host.endswith("svnit.ac.in"):
            raise ValueError(f"Refusing to scrape non-SVNIT domain URL: {u}")

    # go through URLs one by one
    for i, url in enumerate(urls, 1):
        try:
            fname, content = process_url(url)
            out_path = OUTPUT_DIR / fname
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(content)
            print(f"[{i}/{len(urls)}] Saved -> {out_path}")
        except Exception as e:
            # write error file instead
            err_path = OUTPUT_DIR / f"error-{url_to_filename(url)}.md"
            with open(err_path, "w", encoding="utf-8") as out:
                out.write(f"# Error scraping\n\nURL: {url}\nError: {e}\n")
            print(f"[{i}/{len(urls)}] ERROR -> {url}: {e}")
        time.sleep(REQUEST_DELAY_SEC)


if __name__ == "__main__":
    main()
