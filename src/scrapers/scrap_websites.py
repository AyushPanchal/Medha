import os
import re
import time
import pathlib
import urllib.parse
import requests
from bs4 import BeautifulSoup  # NEW: pip install beautifulsoup4  [web:24]
from markdownify import markdownify as md  # pip install markdownify  [web:20]

# Configuration
OUTPUT_DIR = pathlib.Path("../../data/processed/markdowns")
URL_LIST_FILE = r"../../data/raw/links/manual_collection_webpages.txt"  # put the URLs (one per line) in this file
REQUEST_TIMEOUT = 1
REQUEST_DELAY_SEC = 0.8  # polite delay
MAX_RETRIES = 2
USER_AGENT = "SVNIT-CSE-Scraper/1.0 (+https://svnit.ac.in) python-requests"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def url_to_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    if path.endswith("/"):
        base = "index"
    else:
        base = os.path.basename(path)
        if base.lower() in ("index", "index.php", "index.html", ""):
            base = "index"
        else:
            base = re.sub(r"\.(php|html|htm)$", "", base, flags=re.IGNORECASE)

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
    last_err = None
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if 200 <= resp.status_code < 300:
                return resp
            elif resp.status_code in (403, 404):
                return resp
            else:
                last_err = Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(REQUEST_DELAY_SEC)
    raise last_err if last_err else RuntimeError("Unknown fetch error")


def extract_main_fragment(html: str) -> str:
    """
    Parse HTML and return only the <main> region.
    Falls back to common patterns if <main> is missing.
    """
    soup = BeautifulSoup(html, "html.parser")  # [web:24]
    main = soup.select_one("main")  # [web:31]
    if main is None:
        main = soup.select_one('[role="main"], #main, #content, .main, .content')  # [web:26]
    if main is None:
        main = soup.body or soup  # last resort to avoid empty files  [web:26]
    for tag in main.find_all(["script", "style", "noscript"]):  # clean noise  [web:26]
        tag.decompose()
    return str(main)


def html_to_markdown(html: str, base_url: str) -> str:
    return md(
        html,
        heading_style="ATX",
        strip=["script", "style"],
        bullets="*",
        escape_asterisks=False,
        escape_underscores=False,
        keep_inline_images=True,
        base_url=base_url,
    )  # [web:20]


def process_url(url: str) -> tuple[str, str]:
    resp = fetch(url)
    if resp.status_code != 200:
        return url_to_filename(url), f"# Fetch error\n\nStatus: {resp.status_code}\nURL: {url}\n"
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        return url_to_filename(url), f"# Non-HTML content\n\nURL: {url}\nContent-Type: {content_type}\n"
    fragment = extract_main_fragment(resp.text)  # NEW: only main fragment  [web:24][web:31]
    md_text = html_to_markdown(fragment, base_url=url)
    header = f"<!-- Source: {url} | scope: <main> -->\n\n"
    return url_to_filename(url), header + md_text.strip() + "\n"


def main():
    if not os.path.exists(URL_LIST_FILE):
        raise FileNotFoundError(f"Missing {URL_LIST_FILE}. Create it with one URL per line.")
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    for u in urls:
        host = urllib.parse.urlparse(u).netloc
        if not host.endswith("svnit.ac.in"):
            raise ValueError(f"Refusing to scrape non-SVNIT domain URL: {u}")

    for i, url in enumerate(urls, 1):
        try:
            fname, content = process_url(url)
            out_path = OUTPUT_DIR / fname
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(content)
            print(f"[{i}/{len(urls)}] Saved -> {out_path}")
        except Exception as e:
            err_path = OUTPUT_DIR / f"error-{url_to_filename(url)}.md"
            with open(err_path, "w", encoding="utf-8") as out:
                out.write(f"# Error scraping\n\nURL: {url}\nError: {e}\n")
            print(f"[{i}/{len(urls)}] ERROR -> {url}: {e}")
        time.sleep(REQUEST_DELAY_SEC)


if __name__ == "__main__":
    main()
