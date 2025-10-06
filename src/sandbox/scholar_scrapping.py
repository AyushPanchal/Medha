import os
import re
import json
import time
import requests
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup

INPUT_HTML = "https://scholar.google.com/citations?user=uhsQTk0AAAAJ&hl=en"
OUTPUT_DIR = "faculty_markdown"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")

PAGE_SIZE = 100         # fetch 100 articles per page
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_PAGES = 2.5

def is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")

def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" ._")
    return name or "faculty"

def text_or_none(el):
    if not el:
        return None
    t = el.get_text(strip=True)
    return t or None

def normalize_author_url(url: str) -> str:
    """
    Ensure using /citations endpoint with ?user=... and keep hl if present.
    Drop other params; pagination will add cstart/pagesize.
    """
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

def load_html(path_or_url: str) -> str:
    if is_url(path_or_url):
        # First load without pagination for header/metrics/co-authors
        resp = requests.get(normalize_author_url(path_or_url),
                            headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return f.read()

def parse_metrics(soup):
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
        cells = r.find_all(["td","th"])
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

def parse_coauthors(soup):
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

def parse_profile_header(soup):
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

def parse_article_rows(soup):
    articles = []
    for row in soup.select("tr.gsc_a_tr"):
        title_el = row.select_one("a.gsc_a_at")
        authors_el = row.select_one(".gsc_a_at + .gs_gray")
        venue_el = row.select_one(".gs_gray + .gs_gray")
        year_el = row.select_one(".gsc_a_y span")
        cited_el = row.select_one(".gsc_a_c a")
        articles.append({
            "title": text_or_none(title_el),
            "link": ("https://scholar.google.com" + title_el.get("href", "")) if title_el and title_el.get("href") else None,
            "authors": text_or_none(authors_el),
            "venue": text_or_none(venue_el),
            "year": text_or_none(year_el),
            "cited_by": text_or_none(cited_el),
        })
    return articles

def paginate_articles(base_author_url: str) -> list:
    """
    Fetch all articles using cstart/pagesize=100 pagination and merge.
    """
    all_articles = []
    cstart = 0
    normalized = normalize_author_url(base_author_url)
    while True:
        html = fetch_author_page(normalized, cstart=cstart, pagesize=PAGE_SIZE)
        soup = BeautifulSoup(html, "html.parser")
        # scope to table body if present; else whole soup
        table_scope = soup.select_one("#gsc_a_b") or soup
        page_items = parse_article_rows(table_scope)
        if not page_items:
            break
        all_articles.extend(page_items)
        if len(page_items) < PAGE_SIZE:
            break
        cstart += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_PAGES)
    return all_articles

def render_markdown(data: dict) -> str:
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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load initial page (for profile header, metrics, co-authors)
    html = load_html(INPUT_HTML)
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("div#gsc_bdy") or soup

    profile = parse_profile_header(root)
    metrics = parse_metrics(root)
    coauthors = parse_coauthors(root)

    # Fetch ALL articles via pagination with pagesize=100
    articles = []
    if is_url(INPUT_HTML):
        articles = paginate_articles(INPUT_HTML)
    else:
        # Local static HTML cannot paginate; parse whatever rows are present
        table_scope = root.select_one("#gsc_a_b") or root
        articles = parse_article_rows(table_scope)

    data = {
        "profile": profile,
        "metrics": metrics,
        "coauthors": coauthors,
        "articles": articles,
    }

    faculty_name = sanitize_filename(profile.get("name") or "faculty")
    md = render_markdown(data)
    out_path = os.path.join(OUTPUT_DIR, f"{faculty_name}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")
    print(f"Total articles collected: {len(articles)}")

if __name__ == "__main__":
    main()
