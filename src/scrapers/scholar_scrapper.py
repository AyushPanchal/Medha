# src/scholar_scraper.py
import os
import re
import json
import time
import requests
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")
REQUEST_TIMEOUT = 30
PAGE_SIZE = 100
SLEEP_BETWEEN_PAGES = 2.5


# ---------- utils ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" ._")
    return name or "faculty"


def text_or_none(el) -> Optional[str]:
    if not el:
        return None
    t = el.get_text(strip=True)
    return t or None


def normalize_author_url(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    keep = {}
    for k in ("user", "hl"):
        if k in qs and qs[k]:
            keep[k] = qs[k]
    new_q = urlencode(keep, doseq=True)
    normalized = parsed._replace(path="/citations", query=new_q)
    return urlunparse(normalized)


def fetch_author_page(base_url: str, cstart: int, pagesize: int) -> str:
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["cstart"] = [str(cstart)]
    qs["pagesize"] = [str(pagesize)]
    new_q = urlencode(qs, doseq=True)
    url = urlunparse(parsed._replace(query=new_q))
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def load_initial_html(path_or_url: str) -> str:
    if is_url(path_or_url):
        resp = requests.get(normalize_author_url(path_or_url),
                            headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return f.read()


# ---------- parsing ----------
def parse_metrics(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    metrics = {
        "citations_all": None,
        "citations_since_2020": None,
        "h_index_all": None,
        "h_index_since_2020": None,
        "i10_all": None,
        "i10_since_2020": None,
    }
    table = soup.select_one("#gsc_rsb_st")
    if not table:
        return metrics
    rows = table.select("tbody tr")
    for r in rows:
        cells = r.find_all(["td", "th"])
        if len(cells) >= 3:
            label = text_or_none(cells[0]) or ""
            all_val = text_or_none(cells[1])
            since_val = text_or_none(cells[2])
            ll = label.lower()
            if "citations" in ll:
                metrics["citations_all"] = all_val
                metrics["citations_since_2020"] = since_val
            elif "h-index" in ll or "h index" in ll:
                metrics["h_index_all"] = all_val
                metrics["h_index_since_2020"] = since_val
            elif "i10-index" in ll or "i10 index" in ll:
                metrics["i10_all"] = all_val
                metrics["i10_since_2020"] = since_val
    return metrics


def parse_coauthors(soup: BeautifulSoup) -> List[Dict[str, Optional[str]]]:
    coauthors = []
    for li in soup.select("#gsc_rsb_co ul.gsc_rsb_a > li"):
        name_el = li.select_one(".gsc_rsb_a_desc a[href*='/citations?user=']")
        aff_el = li.select_one(".gsc_rsb_a_ext")
        ver_el = li.select(".gsc_rsb_a_ext2")
        coauthors.append({
            "name": text_or_none(name_el),
            "profile_path": name_el.get("href", None) if name_el else None,
            "affiliation": text_or_none(aff_el),
            "verified": text_or_none(ver_el[0]) if ver_el else None,
        })
    return coauthors


def parse_profile_header(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    name = text_or_none(soup.select_one("#gsc_prf_in"))
    affiliation = text_or_none(soup.select_one("#gsc_prf_i .gsc_prf_il"))
    email_line = text_or_none(soup.select_one("#gsc_prf_ivh"))
    interests = [text_or_none(a) for a in soup.select("#gsc_prf_int a.gs_ibl")]
    interests = [x for x in interests if x]
    return {
        "name": name,
        "affiliation": affiliation,
        "email_line": email_line,
        "interests": interests,
    }


def parse_article_rows(soup: BeautifulSoup) -> List[Dict[str, Optional[str]]]:
    articles = []
    for row in soup.select("tr.gsc_a_tr"):
        title_el = row.select_one("a.gsc_a_at")
        authors_el = row.select_one(".gsc_a_at + .gs_gray")
        venue_el = row.select_one(".gs_gray + .gs_gray")
        year_el = row.select_one(".gsc_a_y span")
        cited_el = row.select_one(".gsc_a_c a")
        articles.append({
            "title": text_or_none(title_el),
            "link": ("https://scholar.google.com" + title_el.get("href", "")) if title_el and title_el.get(
                "href") else None,
            "authors": text_or_none(authors_el),
            "venue": text_or_none(venue_el),
            "year": text_or_none(year_el),
            "cited_by": text_or_none(cited_el),
        })
    return articles


# ---------- pagination ----------
def paginate_articles(base_author_url: str) -> List[Dict[str, Optional[str]]]:
    all_articles = []
    cstart = 0
    normalized = normalize_author_url(base_author_url)
    while True:
        html = fetch_author_page(normalized, cstart=cstart, pagesize=PAGE_SIZE)
        soup = BeautifulSoup(html, "html.parser")
        scope = soup.select_one("#gsc_a_b") or soup
        page_items = parse_article_rows(scope)
        if not page_items:
            break
        all_articles.extend(page_items)
        if len(page_items) < PAGE_SIZE:
            break
        cstart += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_PAGES)
    return all_articles


# ---------- rendering ----------
def render_markdown(data: Dict) -> str:
    name = data.get("profile", {}).get("name") or "Unknown Faculty"
    aff = data.get("profile", {}).get("affiliation") or ""
    email_line = data.get("profile", {}).get("email_line") or ""
    interests = data.get("profile", {}).get("interests") or []
    metrics = data.get("metrics", {})
    coauthors = data.get("coauthors", [])
    articles = data.get("articles", [])

    md = []
    md.append("---")
    md.append(f'title: "{name}"')
    md.append(f'affiliation: "{aff}"')
    md.append(f'email_line: "{email_line}"')
    md.append(f'interests: {json.dumps(interests, ensure_ascii=False)}')
    md.append("metrics:")
    for k, v in metrics.items():
        md.append(f"  {k}: \"{v or ''}\"")
    md.append("---\n")

    md.append(f"# {name}")
    if aff:
        md.append(f"- Affiliation: {aff}")
    if email_line:
        md.append(f"- Verified: {email_line}")
    if interests:
        md.append(f"- Interests: {', '.join(interests)}")

    md.append("\n## Metrics")
    md.append(f"- Citations (All): {metrics.get('citations_all') or 'N/A'}")
    md.append(f"- Citations (Since 2020): {metrics.get('citations_since_2020') or 'N/A'}")
    md.append(f"- h-index (All): {metrics.get('h_index_all') or 'N/A'}")
    md.append(f"- h-index (Since 2020): {metrics.get('h_index_since_2020') or 'N/A'}")
    md.append(f"- i10-index (All): {metrics.get('i10_all') or 'N/A'}")
    md.append(f"- i10-index (Since 2020): {metrics.get('i10_since_2020') or 'N/A'}")

    md.append("\n## Co-authors")
    if not coauthors:
        md.append("- None found")
    else:
        for c in coauthors:
            line = c.get("name") or "Unknown"
            aff2 = c.get("affiliation")
            ver = c.get("verified")
            if aff2:
                line += f" — {aff2}"
            if ver:
                line += f" — {ver}"
            md.append(f"- {line}")

    md.append("\n## Articles")
    if not articles:
        md.append("- None found")
    else:
        for i, a in enumerate(articles, 1):
            t = a.get("title") or "Untitled"
            y = a.get("year") or "N/A"
            au = a.get("authors") or "Unknown authors"
            vn = a.get("venue") or "Unknown venue"
            cb = a.get("cited_by") or "0"
            md.append(f"- {i}. {t} ({y}); {au}; {vn}; Cited by: {cb}")

    return "\n".join(md)


# ---------- high-level API ----------
def scrape_author(url_or_html_path: str) -> Dict:
    html = load_initial_html(url_or_html_path)
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("div#gsc_bdy") or soup

    profile = parse_profile_header(root)
    metrics = parse_metrics(root)
    coauthors = parse_coauthors(root)

    if is_url(url_or_html_path):
        articles = paginate_articles(url_or_html_path)
    else:
        scope = root.select_one("#gsc_a_b") or root
        articles = parse_article_rows(scope)

    return {
        "profile": profile,
        "metrics": metrics,
        "coauthors": coauthors,
        "articles": articles,
    }


def write_markdown(data: Dict, out_dir: str) -> str:
    ensure_dir(out_dir)
    name = data.get("profile", {}).get("name") or "faculty"
    fname = sanitize_filename(name) + ".md"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(data))
    return out_path


# Hardcoded run config
INPUT_LIST = "../../data/raw/links/faculty_google_scholar_links.txt"  # one URL per line
OUTPUT_DIR = r"../../data/processed/markdowns"  # target folder


def read_urls(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            yield u


def main():
    ensure_dir(OUTPUT_DIR)
    count = 0
    for url in read_urls(INPUT_LIST):
        try:
            data = scrape_author(url)
            out_path = write_markdown(data, OUTPUT_DIR)
            print(f"[OK] {out_path}")
            count += 1
        except Exception as e:
            print(f"[ERROR] {url} -> {e}")
    print(f"Completed: {count} profiles")


if __name__ == "__main__":
    # Hardcoded invocation; no CLI args needed.
    main()
