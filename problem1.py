# B23CS1048 — NLU Assignment 2
# Run: python assignment2.py
# Install: pip install requests beautifulsoup4 nltk pdfplumber numpy matplotlib gensim scikit-learn

import os, re, time, random, io, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import Counter, defaultdict
from itertools import product as grid

import requests
from bs4 import BeautifulSoup

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False
    print("[WARN] pdfplumber not found → pip install pdfplumber")

try:
    from gensim.models import Word2Vec as GensimW2V
    GENSIM = True
except ImportError:
    GENSIM = False
    print("[INFO] gensim unavailable — scratch models only")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN = True
except ImportError:
    SKLEARN = False
    print("[WARN] scikit-learn not found → pip install scikit-learn")


# Logger writes every message to both stdout and run_log.txt with a timestamp

class Logger:
    def __init__(self, path="run_log.txt"):
        self.path = path
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            f.write("NLU Assignment 2 — Run Log\n")
            f.write(f"Started : {ts}\n")
            f.write("=" * 70 + "\n\n")

    def write(self, msg=""):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def section(self, title):
        self.write()
        self.write("-" * 60)
        self.write(title)
        self.write("-" * 60)


LOG = Logger("run_log.txt")


# TASK 1 — DATASET PREPARATION

STOP_WORDS = set(stopwords.words("english"))

# Artifact tokens that leak from HTML structure, JS, and DB-backed pages
ARTIFACT_STOPWORDS = {
    "cid", "vksj", "gsa", "issn", "yes", "procedure", "regular",
    "click", "login", "logout", "back", "next", "prev", "previous",
    "page", "pages", "menu", "toggle", "modal", "popup", "widget",
    "submit", "reset", "cancel", "close", "open", "show", "hide",
    "loading", "loaded", "error", "success", "warning",
    "cookie", "cookies", "consent", "analytics", "tracking", "gdpr",
    "required", "optional", "select", "input", "textarea", "checkbox",
    "sitemap", "breadcrumb", "read", "more", "view", "all", "details",
    "download", "upload", "filter", "sort", "clear",
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
}
STOP_WORDS |= ARTIFACT_STOPWORDS

OUTPUT_DIR      = "."
REQUEST_DELAY   = (1.5, 3.0)
MAX_HTML        = 300
MAX_PDFS        = 9999
MIN_PDF_WORDS   = 30
MIN_HTML_TOKENS = 40
TIMEOUT_HTML    = 15
TIMEOUT_PDF     = 60

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# m/Index pages are JS-rendered navigation shells with no real text
JS_SHELL_PATTERN = re.compile(r"/m/Index/", re.IGNORECASE)

SEED_URLS = [
    "https://iitj.ac.in/main/en/iitj",
    "https://iitj.ac.in/main/en/introduction",
    "https://iitj.ac.in/main/en/director",
    "https://iitj.ac.in/main/en/chairman",
    "https://iitj.ac.in/main/en/administrative-contact",
    "https://iitj.ac.in/main/en/faculty-members",
    "https://iitj.ac.in/main/en/adjunct-faculty-members",
    "https://iitj.ac.in/main/en/visiting-faculty-members",
    "https://iitj.ac.in/main/en/scholars-in-residence",
    "https://iitj.ac.in/main/en/research-highlight",
    "https://iitj.ac.in/main/en/news",
    "https://iitj.ac.in/main/en/all-announcement",
    "https://iitj.ac.in/main/en/events",
    "https://iitj.ac.in/main/en/faqs-applicants",
    "https://iitj.ac.in/main/en/important-links",
    "https://iitj.ac.in/main/en/recruitments",
    "https://iitj.ac.in/main/en/how-to-reach-iit-jodhpur",
    "https://iitj.ac.in/main/en/why-pursue-a-career-@-iit-jodhpur",
    "https://iitj.ac.in/institute-repository/en/nirf",
    "https://iitj.ac.in/Institute-Repository/en/Institute-Repository",
    "https://iitj.ac.in/techscape/en/Techscape",
    "https://iitj.ac.in/anti-sexual-harassment-policy/en/anti-sexual-harassment-policy",
    "https://iitj.ac.in/office-of-director/en/office-of-director",
    "https://iitj.ac.in/office-of-deputy-director/en/office-of-deputy-director",
    "https://iitj.ac.in/office-of-registrar/en/office-of-registrar",
    "https://iitj.ac.in/office-of-administration/en/office-of-administration",
    "https://iitj.ac.in/office-of-executive-education/en/office-of-executive-education",
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://iitj.ac.in/office-of-students/en/office-of-students",
    "https://iitj.ac.in/office-of-students/en/campus-life",
    "https://iitj.ac.in/faculty-positions/en/faculty-positions",
    "https://iitj.ac.in/office-of-stores-purchase/en/tender-details",
    "https://iitj.ac.in/office-of-corporate-relations/en/Donate",
    "https://iitj.ac.in/office-of-academics/en/academics",
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://iitj.ac.in/office-of-academics/en/academic-regulations/old-regulation",
    "https://iitj.ac.in/office-of-academics/en/programmes",
    "https://iitj.ac.in/office-of-academics/en/academic-calendar",
    "https://iitj.ac.in/office-of-academics/en/list-of-programs",
    "https://iitj.ac.in/office-of-academics/en/people",
    "https://iitj.ac.in/es/en/engineering-science",
    "https://iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
    "https://iitj.ac.in/admission-postgraduate-programs/en/list-of-provisionally-selected-candidates",
    "https://iitj.ac.in/admission-postgraduate-programs/en/list-of-shortlisted-candidates",
    "https://iitj.ac.in/admission-postgraduate-programs/en/msc-computational-social-science",
    "https://iitj.ac.in/main/en/admission-links",
    "https://iitj.ac.in/computer-science-engineering/en/home",
    "https://iitj.ac.in/computer-science-engineering/en/people",
    "https://iitj.ac.in/computer-science-engineering/en/research",
    "https://iitj.ac.in/computer-science-engineering/en/courses",
    "https://iitj.ac.in/computer-science-engineering/en/syllabus",
    "https://iitj.ac.in/computer-science-engineering/en/academics",
    "https://iitj.ac.in/computer-science-engineering/en/labs",
    "https://iitj.ac.in/computer-science-engineering/en/Research-Highlights",
    "https://iitj.ac.in/electrical-engineering/en/home",
    "https://iitj.ac.in/electrical-engineering/en/people",
    "https://iitj.ac.in/electrical-engineering/en/research",
    "https://iitj.ac.in/electrical-engineering/en/courses",
    "https://iitj.ac.in/electrical-engineering/en/syllabus",
    "https://iitj.ac.in/electrical-engineering/en/labs",
    "https://iitj.ac.in/electrical-engineering/en/Research-Highlights",
    "https://iitj.ac.in/electrical-engineering/en/academics",
    "https://iitj.ac.in/mechanical-engineering/en/home",
    "https://iitj.ac.in/mechanical-engineering/en/people",
    "https://iitj.ac.in/mechanical-engineering/en/research",
    "https://iitj.ac.in/mechanical-engineering/en/courses",
    "https://iitj.ac.in/mechanical-engineering/en/syllabus",
    "https://iitj.ac.in/mechanical-engineering/en/labs",
    "https://iitj.ac.in/mechanical-engineering/en/Research-Highlights",
    "https://iitj.ac.in/civil-infrastructure-engineering/en/home",
    "https://iitj.ac.in/civil-infrastructure-engineering/en/people",
    "https://iitj.ac.in/civil-infrastructure-engineering/en/research",
    "https://iitj.ac.in/civil-infrastructure-engineering/en/courses",
    "https://iitj.ac.in/civil-infrastructure-engineering/en/syllabus",
    "https://iitj.ac.in/civil-infrastructure-engineering/en/labs",
    "https://iitj.ac.in/chemical-engineering/en/home",
    "https://iitj.ac.in/chemical-engineering/en/people",
    "https://iitj.ac.in/chemical-engineering/en/research",
    "https://iitj.ac.in/chemical-engineering/en/courses",
    "https://iitj.ac.in/chemical-engineering/en/syllabus",
    "https://iitj.ac.in/chemical-engineering/en/labs",
    "https://iitj.ac.in/physics/en/home",
    "https://iitj.ac.in/physics/en/people",
    "https://iitj.ac.in/physics/en/research",
    "https://iitj.ac.in/physics/en/courses",
    "https://iitj.ac.in/physics/en/syllabus",
    "https://iitj.ac.in/physics/en/labs",
    "https://iitj.ac.in/physics/en/Research-Highlights",
    "https://iitj.ac.in/chemistry/en/home",
    "https://iitj.ac.in/chemistry/en/people",
    "https://iitj.ac.in/chemistry/en/research",
    "https://iitj.ac.in/chemistry/en/courses",
    "https://iitj.ac.in/chemistry/en/syllabus",
    "https://iitj.ac.in/chemistry/en/labs",
    "https://iitj.ac.in/mathematics/en/home",
    "https://iitj.ac.in/mathematics/en/people",
    "https://iitj.ac.in/mathematics/en/research",
    "https://iitj.ac.in/mathematics/en/courses",
    "https://iitj.ac.in/mathematics/en/syllabus",
    "https://iitj.ac.in/humanities-social-sciences/en/home",
    "https://iitj.ac.in/humanities-social-sciences/en/people",
    "https://iitj.ac.in/humanities-social-sciences/en/research",
    "https://iitj.ac.in/humanities-social-sciences/en/courses",
    "https://iitj.ac.in/bioscience-bioengineering/en/home",
    "https://iitj.ac.in/bioscience-bioengineering/en/people",
    "https://iitj.ac.in/bioscience-bioengineering/en/research",
    "https://iitj.ac.in/bioscience-bioengineering/en/courses",
    "https://iitj.ac.in/bioscience-bioengineering/en/syllabus",
    "https://iitj.ac.in/bioscience-bioengineering/en/labs",
    "https://iitj.ac.in/materials-engineering/en/home",
    "https://iitj.ac.in/materials-engineering/en/people",
    "https://iitj.ac.in/materials-engineering/en/research",
    "https://iitj.ac.in/materials-engineering/en/courses",
    "https://iitj.ac.in/materials-engineering/en/syllabus",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/home",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/people",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/research",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/courses",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/syllabus",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/Events",
    "https://iitj.ac.in/schools/en/cat-cut-offs",
    "https://iitj.ac.in/health-center/en/health-center",
    "https://iitj.ac.in/crf/en/crf",
    "https://iitj.ac.in/dia/en/dia",
    "https://iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility",
]

INTERNAL_DOMAINS = {"iitj.ac.in", "www.iitj.ac.in"}

NON_ENG = re.compile(
    r"[\u0900-\u097F\u0A80-\u0AFF\u0600-\u06FF\u0B00-\u0B7F]+", re.UNICODE
)
BOILERPLATE = [
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "form", "iframe", "svg", "button",
    "select", "option", "input", "textarea", "meta", "link", "head",
]
UI_CLASS_PATTERN = re.compile(
    r"breadcrumb|pagination|cookie|consent|gdpr|widget|"
    r"navbar|sidebar|topbar|toolbar|modal|overlay|popup|"
    r"gsa|search-box|social|share|print|accessibility",
    re.IGNORECASE,
)


def fetch_html(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_HTML)
        r.raise_for_status()
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        LOG.write(f"    [skip-html] {e}")
        return None


def fetch_pdf(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_PDF, stream=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
            return None
        return r.content
    except Exception as e:
        LOG.write(f"    [skip-pdf] {e}")
        return None


def html_to_text(soup):
    for tag in soup(BOILERPLATE):
        tag.decompose()
    # Remove UI chrome containers by CSS class or id
    for tag in soup.find_all(True):
        try:
            if not tag.attrs:
                continue
            cls = " ".join(tag.attrs.get("class") or [])
            tid = tag.attrs.get("id") or ""
            if UI_CLASS_PATTERN.search(cls) or UI_CLASS_PATTERN.search(tid):
                tag.decompose()
        except (AttributeError, TypeError):
            continue
    # Remove jQuery-injected elements with hash-like ids
    for tag in soup.find_all(True):
        try:
            tid = tag.attrs.get("id", "") if tag.attrs else ""
            if tid and re.match(r"^[_a-f0-9]{20,}$", tid, re.I):
                tag.decompose()
        except (AttributeError, TypeError):
            continue
    root = (soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"^content|^main|^body", re.I))
            or soup.find(class_=re.compile(r"content|main-content|page-body", re.I))
            or soup.body or soup)
    raw   = root.get_text(separator=" ")
    lines = [NON_ENG.sub(" ", l) for l in raw.splitlines() if l.strip()]
    return " ".join(lines)


def pdf_to_text(pdf_bytes):
    if not PDF_OK:
        return ""
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                raw = page.extract_text()
                if not raw:
                    continue
                clean = " ".join(
                    NON_ENG.sub(" ", line)
                    for line in raw.splitlines() if line.strip()
                )
                parts.append(clean)
            LOG.write(f"    → {n_pages} pages, text from {len(parts)} pages")
    except Exception as e:
        LOG.write(f"    [pdf-parse-err] {e}")
    return " ".join(parts)


def extract_links(soup, base_url):
    html_links, pdf_links = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript") or href.startswith("mailto"):
            continue
        full   = urljoin(base_url, href).split("#")[0].rstrip("/")
        parsed = urlparse(full)
        if parsed.netloc not in INTERNAL_DOMAINS and parsed.netloc != "":
            continue
        if re.search(r"\.pdf(\?.*)?$", full, re.I):
            pdf_links.append(full)
        elif not re.search(r"\.(jpg|jpeg|png|gif|zip|xls|ppt|mp4|avi)$", full, re.I):
            if not JS_SHELL_PATTERN.search(full):
                html_links.append(full)
    return html_links, pdf_links


def to_tokens(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for t in word_tokenize(text):
        if not t.isalpha():
            continue
        if len(t) < 3 or len(t) > 25:
            continue
        if t in STOP_WORDS:
            continue
        tokens.append(t)
    return tokens


def to_sentences(text):
    out = []
    for s in sent_tokenize(text):
        s = s.lower()
        s = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", s)
        s = re.sub(r"[^a-z\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s.split()) >= 4:
            out.append(s)
    return out


def make_doc(url, title, raw_text, source):
    tokens = to_tokens(raw_text)
    sents  = to_sentences(raw_text)
    if len(tokens) < MIN_HTML_TOKENS:
        return None
    return {"url": url, "title": title, "source": source,
            "tokens": tokens, "sentences": sents}


def crawl():
    visited_html = set()
    visited_pdf  = set()
    html_queue   = list(dict.fromkeys(SEED_URLS))
    pdf_queue    = []
    documents    = []

    LOG.write(f"Seed URLs: {len(html_queue)}  |  JS-shell pages excluded")

    LOG.write("\nPhase 1 — HTML crawl")
    while html_queue and len(visited_html) < MAX_HTML:
        url = html_queue.pop(0)
        if url in visited_html:
            continue
        if JS_SHELL_PATTERN.search(url):
            continue
        visited_html.add(url)
        LOG.write(f"  [HTML {len(visited_html):3d}/{MAX_HTML}] {url}")
        soup = fetch_html(url)
        if soup is None:
            continue
        raw   = html_to_text(soup)
        title = soup.title.get_text().strip() if soup.title else url
        doc   = make_doc(url, title, raw, "html")
        if doc:
            documents.append(doc)
            LOG.write(f"    → {len(doc['tokens'])} tokens  |  {len(doc['sentences'])} sentences  ✓")
        else:
            LOG.write(f"    → skipped (insufficient content)")
        new_html, new_pdfs = extract_links(soup, url)
        for lnk in new_html:
            if lnk not in visited_html and lnk not in html_queue:
                html_queue.append(lnk)
        for pdf in new_pdfs:
            if pdf not in visited_pdf and pdf not in pdf_queue:
                pdf_queue.append(pdf)
        time.sleep(random.uniform(*REQUEST_DELAY))

    html_n = sum(1 for d in documents if d["source"] == "html")
    LOG.write(f"HTML done — {html_n} docs  |  {len(pdf_queue)} PDFs queued")

    if not PDF_OK:
        LOG.write("Phase 2 — SKIPPED (pdfplumber not installed)")
    else:
        LOG.write(f"\nPhase 2 — Parsing ALL {len(pdf_queue)} PDFs")
        for idx, pdf_url in enumerate(pdf_queue, 1):
            if pdf_url in visited_pdf:
                continue
            visited_pdf.add(pdf_url)
            fname = urlparse(pdf_url).path.split("/")[-1]
            LOG.write(f"  [PDF {idx:3d}/{len(pdf_queue)}] {fname}")
            LOG.write(f"    {pdf_url}")
            pdf_bytes = fetch_pdf(pdf_url)
            if not pdf_bytes:
                continue
            raw = pdf_to_text(pdf_bytes)
            if len(raw.split()) < MIN_PDF_WORDS:
                LOG.write(f"    → skipped (image/scanned PDF)")
                continue
            doc = make_doc(pdf_url, fname, raw, "pdf")
            if doc:
                documents.append(doc)
                LOG.write(f"    → {len(doc['tokens'])} tokens  |  {len(doc['sentences'])} sentences  ✓")
            time.sleep(random.uniform(*REQUEST_DELAY))

    html_n = sum(1 for d in documents if d["source"] == "html")
    pdf_n  = sum(1 for d in documents if d["source"] == "pdf")
    LOG.write(f"TOTAL: {len(documents)} docs  (HTML: {html_n}  |  PDF: {pdf_n})")
    return documents


def compute_stats(documents):
    all_tok  = [t for d in documents for t in d["tokens"]]
    all_sent = [s for d in documents for s in d["sentences"]]
    freq     = Counter(all_tok)
    return {
        "total_documents":    len(documents),
        "html_documents":     sum(1 for d in documents if d["source"] == "html"),
        "pdf_documents":      sum(1 for d in documents if d["source"] == "pdf"),
        "total_sentences":    len(all_sent),
        "total_tokens":       len(all_tok),
        "vocabulary_size":    len(freq),
        "avg_tokens_per_doc": round(len(all_tok) / max(len(documents), 1), 1),
        "top_50_words":       freq.most_common(50),
    }, freq


def log_stats(s):
    LOG.write("Dataset statistics:")
    for k in ["total_documents","html_documents","pdf_documents",
              "total_sentences","total_tokens","vocabulary_size","avg_tokens_per_doc"]:
        LOG.write(f"  {k:<25s}: {s[k]}")
    LOG.write("Top 25 words:")
    for w, c in s["top_50_words"][:25]:
        LOG.write(f"  {w:<22s} {c:>6d}")


def save_corpus_files(documents, stats):
    with open(f"{OUTPUT_DIR}/corpus.txt", "w", encoding="utf-8") as f:
        for doc in documents:
            for sent in doc["sentences"]:
                f.write(sent + "\n")
    LOG.write("Saved → corpus.txt")
    light = [{"url": d["url"], "title": d["title"], "source": d["source"],
              "num_tokens": len(d["tokens"]), "num_sentences": len(d["sentences"])}
             for d in documents]
    with open(f"{OUTPUT_DIR}/documents.json", "w", encoding="utf-8") as f:
        json.dump(light, f, indent=2, ensure_ascii=False)
    LOG.write("Saved → documents.json")
    with open(f"{OUTPUT_DIR}/stats_report.txt", "w", encoding="utf-8") as f:
        f.write("IIT JODHPUR CORPUS — DATASET STATISTICS\n" + "="*50 + "\n")
        for k in ["total_documents","html_documents","pdf_documents",
                  "total_sentences","total_tokens","vocabulary_size","avg_tokens_per_doc"]:
            f.write(f"{k:<25s}: {stats[k]}\n")
        f.write("\nTop 50 words:\n")
        for w, c in stats["top_50_words"]:
            f.write(f"  {w:<25s} {c}\n")
    LOG.write("Saved → stats_report.txt")


def run_task1():
    LOG.section("TASK 1 — DATASET PREPARATION")

    # If corpus already exists skip scraping and load directly
    if os.path.exists("corpus.txt"):
        LOG.write("corpus.txt found — loading existing file (skipping scrape)")
        sents = []
        with open("corpus.txt", encoding="utf-8") as f:
            for line in f:
                t = line.strip().split()
                if len(t) >= 3:
                    sents.append(t)
        LOG.write(f"Loaded {len(sents):,} sentences from corpus.txt")
        return sents

    # corpus.txt missing — scrape the website from scratch
    LOG.write("corpus.txt not found — starting web scrape")
    documents = crawl()
    if not documents:
        LOG.write("[ERROR] Nothing collected")
        return None
    stats, _ = compute_stats(documents)
    log_stats(stats)
    save_corpus_files(documents, stats)
    sents = [sent.split() for doc in documents
             for sent in doc["sentences"] if len(sent.split()) >= 3]
    LOG.write(f"Corpus ready: {len(sents):,} sentences")
    return sents


# TASK 2 — MODEL TRAINING

EMBED_DIMS  = [50, 100, 200]
WINDOWS     = [2, 5]
NEG_SAMPLES = [5, 10]
EPOCHS      = 5
MIN_COUNT   = 2
CONFIGS     = list(grid(EMBED_DIMS, WINDOWS, NEG_SAMPLES))
LABELS      = [f"d{d}w{w}n{n}" for d, w, n in CONFIGS]


def load_corpus_file():
    sents = []
    with open("corpus.txt", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 3:
                sents.append(t)
    LOG.write(f"Corpus loaded: {len(sents):,} sentences")
    return sents


class Vocab:
    def __init__(self, sents):
        freq  = Counter(w for s in sents for w in s)
        words = sorted(w for w, c in freq.items() if c >= MIN_COUNT)
        self.w2i  = {w: i+1 for i, w in enumerate(words)}
        self.i2w  = {i+1: w for i, w in enumerate(words)}
        self.size = len(words) + 1
        cnt = np.zeros(self.size)
        for w, i in self.w2i.items():
            cnt[i] = freq[w]
        cnt = cnt ** 0.75
        self.noise = cnt / cnt.sum()
        self.enc = [[self.w2i[w] for w in s if w in self.w2i] for s in sents]
        self.enc = [s for s in self.enc if len(s) >= 2]

    def neg_sample(self, k, excl):
        cands = np.random.choice(self.size, size=k*4, p=self.noise)
        return [c for c in cands if c not in excl and c != 0][:k]


class W2VBase:
    def __init__(self, vs, dim, neg):
        rng        = np.random.default_rng(42)
        self.W_in  = (rng.standard_normal((vs, dim)) * 0.01).astype(np.float32)
        self.W_out = np.zeros((vs, dim), dtype=np.float32)
        self.dim   = dim
        self.neg   = neg

    @staticmethod
    def _sig(x):
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))

    def _step(self, cv, pos, negs, lr):
        vp    = self.W_out[pos]
        score = np.clip(cv @ vp, -10, 10)
        sp    = np.clip(1 / (1 + np.exp(-score)), 1e-10, 1-1e-10)
        loss  = -np.log(sp)
        g     = (sp - 1) * vp
        self.W_out[pos] -= lr * (sp - 1) * cv
        for ni in negs:
            vn      = self.W_out[ni]
            score_n = np.clip(cv @ vn, -10, 10)
            sn      = np.clip(1 / (1 + np.exp(-score_n)), 1e-10, 1-1e-10)
            loss   += -np.log(1 - sn)
            self.W_out[ni] -= lr * sn * cv
            g += sn * vn
        return float(loss), g

    def cosine(self, a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def neighbors(self, word, vocab, n=5):
        if word not in vocab.w2i:
            return []
        idx  = vocab.w2i[word]
        emb  = self.W_in[idx]
        sims = [(vocab.i2w[i], self.cosine(emb, self.W_in[i]))
                for i in range(1, vocab.size) if i != idx]
        return sorted(sims, key=lambda x: -x[1])[:n]

    def analogy(self, a, b, c, vocab, n=1):
        if any(w not in vocab.w2i for w in [a, b, c]):
            return []
        ea   = self.W_in[vocab.w2i[a]]
        eb   = self.W_in[vocab.w2i[b]]
        ec   = self.W_in[vocab.w2i[c]]
        q    = eb - ea + ec
        sims = [(vocab.i2w[i], self.cosine(q, self.W_in[i]))
                for i in range(1, vocab.size) if vocab.i2w[i] not in {a, b, c}]
        return sorted(sims, key=lambda x: -x[1])[:n]


class CBOW(W2VBase):
    def train(self, vocab, window, epochs=5, lr=0.025):
        for ep in range(epochs):
            start = time.time(); total_loss = 0; pairs = 0
            np.random.shuffle(vocab.enc)
            for sent in vocab.enc:
                n = len(sent)
                for i, center in enumerate(sent):
                    ctx = [sent[j] for j in range(max(0,i-window), min(n,i+window+1)) if j!=i]
                    if not ctx:
                        continue
                    cv   = self.W_in[ctx].mean(0)
                    negs = vocab.neg_sample(self.neg, set(ctx)|{center})
                    if len(negs) < self.neg:
                        continue
                    loss, g = self._step(cv, center, negs, lr)
                    self.W_in[ctx] -= lr * g / len(ctx)
                    total_loss += loss; pairs += 1
            avg = total_loss / max(pairs, 1)
            LOG.write(f"[CBOW] Epoch {ep+1}/{epochs} | Loss: {avg:.4f} | Pairs: {pairs} | Time: {time.time()-start:.2f}s")


class SkipGram(W2VBase):
    def train(self, vocab, window, epochs=5, lr=0.025):
        for ep in range(epochs):
            start = time.time(); total_loss = 0; pairs = 0
            np.random.shuffle(vocab.enc)
            for sent in vocab.enc:
                n = len(sent)
                for i, center in enumerate(sent):
                    for j in range(max(0,i-window), min(n,i+window+1)):
                        if j == i:
                            continue
                        ctx  = sent[j]
                        cv   = self.W_in[center].copy()
                        negs = vocab.neg_sample(self.neg, {center, ctx})
                        if len(negs) < self.neg:
                            continue
                        loss, g = self._step(cv, ctx, negs, lr)
                        self.W_in[center] -= lr * g
                        total_loss += loss; pairs += 1
            avg = total_loss / max(pairs, 1)
            LOG.write(f"[SkipGram] Epoch {ep+1}/{epochs} | Loss: {avg:.4f} | Pairs: {pairs} | Time: {time.time()-start:.2f}s")


class GensimWrapper:
    def __init__(self, sg_flag, dim=300, window=5, neg=10):
        self.sg_flag = sg_flag
        self.dim     = dim
        self.window  = window
        self.neg     = neg
        self.model   = None
        self.W_in    = None

    def _build_matrix(self):
        # Faster: directly use gensim's internal matrix
        self.W_in = self.model.wv.vectors
        keys      = self.model.wv.index_to_key
        self.w2i  = {w: i for i, w in enumerate(keys)}
        self.i2w  = {i: w for i, w in enumerate(keys)}

    def train(self, sents, epochs=20):
        tag = "SkipGram" if self.sg_flag else "CBOW"
        LOG.write(f"[Gensim-{tag}] Training (dim={self.dim}, win={self.window}, neg={self.neg})")

        t0 = time.time()

        # Step 1: Initialize model (no sentences yet)
        self.model = GensimW2V(
            vector_size=self.dim,
            window=self.window,
            min_count=MIN_COUNT,
            sg=self.sg_flag,
            negative=self.neg,
            sample=1e-5,                # subsampling (important)
            ns_exponent=0.75,          # default but explicit
            workers=os.cpu_count(),    # use all cores
            seed=42,
        )

        # Step 2: Build vocabulary
        self.model.build_vocab(sents)

        # Step 3: Train
        self.model.train(
            sents,
            total_examples=self.model.corpus_count,
            epochs=epochs
        )

        LOG.write(f"[Gensim-{tag}] Done in {time.time()-t0:.2f}s")

        # Step 4: Build embedding matrix
        self._build_matrix()

    def save(self, path):
        if self.model:
            self.model.save(path)
            LOG.write(f"Saved → {path}")

    @classmethod
    def load(cls, path, sg_flag):
        obj       = cls(sg_flag)
        obj.model = GensimW2V.load(path)
        obj._build_matrix()
        return obj

    def neighbors(self, word, n=5):
        if self.model is None or word not in self.model.wv:
            return []
        return self.model.wv.most_similar(word, topn=n)

    def analogy(self, a, b, c, n=1):
        if self.model is None:
            return []
        try:
            return self.model.wv.most_similar(
                positive=[b, c],
                negative=[a],
                topn=n
            )
        except KeyError:
            return []

def train_grid(sents):
    vocab = Vocab(sents)
    LOG.write(f"Vocab size={vocab.size}  encoded sentences={len(vocab.enc)}")
    R = {"scratch": {"cbow": [], "sg": []}, "gensim": {"cbow": [], "sg": []}}
    for dim, win, neg in CONFIGS:
        tag = f"d{dim}w{win}n{neg}"
        for cls_, key, lbl in [(CBOW, "cbow", "CBOW"), (SkipGram, "sg", "SG")]:
            LOG.write(f"[scratch {lbl}] {tag}")
            t0 = time.time()
            m  = cls_(vocab.size, dim, neg)
            m.train(vocab, win, EPOCHS)
            R["scratch"][key].append({"dim":dim,"win":win,"neg":neg,"label":tag,"time":round(time.time()-t0,2)})
    if GENSIM:
        for dim, win, neg in CONFIGS:
            tag = f"d{dim}w{win}n{neg}"
            for sf, key, lbl in [(0,"cbow","CBOW"),(1,"sg","SG")]:
                LOG.write(f"[gensim {lbl}] {tag}")
                t0 = time.time()
                GensimW2V(sentences=sents, vector_size=dim, window=win,
                          negative=neg, sg=sf, min_count=MIN_COUNT, workers=4, epochs=EPOCHS, seed=42)
                R["gensim"][key].append({"dim":dim,"win":win,"neg":neg,"label":tag,"time":round(time.time()-t0,2)})
    return R, vocab


def train_and_save(sents):
    vocab = Vocab(sents)
    LOG.write("Training scratch CBOW  (dim=100, win=5, neg=10)")
    cbow = CBOW(vocab.size, 100, 10)
    cbow.train(vocab, 5, EPOCHS)
    LOG.write("Training scratch SkipGram  (dim=100, win=5, neg=10)")
    sg = SkipGram(vocab.size, 100, 10)
    sg.train(vocab, 5, EPOCHS)
    np.save("cbow_embeddings.npy", cbow.W_in)
    np.save("sg_embeddings.npy",   sg.W_in)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    LOG.write("Saved: cbow_embeddings.npy  sg_embeddings.npy  vocab.pkl")
    return vocab, cbow, sg


def plot_train_time(R):
    x  = np.arange(len(CONFIGS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Time per Config (seconds)", fontweight="bold")
    for ax, key, title in zip(axes, ["cbow","sg"], ["CBOW","Skip-gram"]):
        st = [r["time"] for r in R["scratch"][key]]
        ax.bar(x, st, 0.4, label="Scratch", color="#2563eb", alpha=0.85)
        if GENSIM and R["gensim"][key]:
            gt = [r["time"] for r in R["gensim"][key]]
            ax.plot(x, gt, marker="D", linewidth=2, markersize=6, color="#16a34a", label="Gensim")
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Time (s)"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("plot_train_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    LOG.write("Saved → plot_train_time.png")


def run_task2(sents):
    LOG.section("TASK 2 — MODEL TRAINING")

    scratch_exist = (os.path.exists("vocab.pkl") and
                     os.path.exists("cbow_embeddings.npy") and
                     os.path.exists("sg_embeddings.npy"))
    gensim_exist  = (GENSIM and
                     os.path.exists("gensim_cbow.model") and
                     os.path.exists("gensim_sg.model"))

    # Load scratch models if they already exist
    if scratch_exist:
        LOG.write("Scratch embeddings found — loading (no retraining)")
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        cbow_emb  = np.load("cbow_embeddings.npy")
        sg_emb    = np.load("sg_embeddings.npy")
        cbow      = CBOW(vocab.size, cbow_emb.shape[1], 10)
        sg        = SkipGram(vocab.size, sg_emb.shape[1], 10)
        cbow.W_in = cbow_emb
        sg.W_in   = sg_emb
        LOG.write(f"  vocab size={vocab.size}  embed dim={cbow_emb.shape[1]}")
    else:
        LOG.write("No saved scratch models — running training grid then saving best config")
        R, _ = train_grid(sents)
        plot_train_time(R)
        vocab, cbow, sg = train_and_save(sents)

    # Load gensim models if they already exist
    if GENSIM:
        if gensim_exist:
            LOG.write("Gensim models found — loading (no retraining)")
            g_cbow = GensimWrapper.load("gensim_cbow.model", sg_flag=0)
            g_sg   = GensimWrapper.load("gensim_sg.model",   sg_flag=1)
        else:
            LOG.write("No saved gensim models — training and saving")
            g_cbow = GensimWrapper(sg_flag=0, dim=300, window=7, neg=10)
            g_cbow.train(sents, epochs=40)
            g_cbow.save("gensim_cbow.model")
            g_sg = GensimWrapper(sg_flag=1, dim=300, window=7, neg=10)
            g_sg.train(sents, epochs=40)
            g_sg.save("gensim_sg.model")
    else:
        g_cbow = g_sg = None

    return vocab, cbow, sg, g_cbow, g_sg


# TASK 3 — SEMANTIC ANALYSIS

PROBE_WORDS = ["research", "student", "phd", "exam", "laboratory"]
ANALOGIES = [
    # Program levels (required)
    ("ug", "btech",        "pg", "mtech"),

    # Admission exams
    ("btech", "jee",       "mtech", "gate"),

    # Leadership roles
    ("director", "institute",   "dean", "academics"),

    # Academic hierarchy
    ("department", "professor", "institute", "director"),

    # Degree completion
    ("btech", "project",   "phd", "thesis"),

    # Evaluation system
    ("course", "grade",    "semester", "gpa"),
]

def run_neighbors(vocab, cbow, sg, g_cbow=None, g_sg=None):
    LOG.write("NEAREST NEIGHBORS — Top 5 per probe word")
    for word in PROBE_WORDS:
        LOG.write(f"\n  Probe: '{word}'")
        LOG.write("    CBOW (scratch):")
        for w, s in cbow.neighbors(word, vocab):
            LOG.write(f"      {w:<20s}  cos={s:.4f}")
        LOG.write("    SkipGram (scratch):")
        for w, s in sg.neighbors(word, vocab):
            LOG.write(f"      {w:<20s}  cos={s:.4f}")
        if g_cbow:
            LOG.write("    Gensim CBOW:")
            for w, s in g_cbow.neighbors(word):
                LOG.write(f"      {w:<20s}  cos={s:.4f}")
        if g_sg:
            LOG.write("    Gensim SkipGram:")
            for w, s in g_sg.neighbors(word):
                LOG.write(f"      {w:<20s}  cos={s:.4f}")


def run_analogies(vocab, cbow, sg, g_cbow=None, g_sg=None):
    LOG.write("\nANALOGY EXPERIMENTS — format: a : b :: c : ?")
    rows = []
    for a, b, c, expected in ANALOGIES:
        LOG.write(f"\n  {a} : {b} :: {c} : ?   (expected: {expected})")
        cb_res = cbow.analogy(a, b, c, vocab)
        sg_res = sg.analogy(a, b, c, vocab)
        cbow_pred = cb_res[0][0] if cb_res else "OOV"
        sg_pred   = sg_res[0][0] if sg_res else "OOV"
        LOG.write(f"    CBOW (scratch)  : {cbow_pred:<20s} {'✓' if cbow_pred==expected else '✗'}")
        LOG.write(f"    SG   (scratch)  : {sg_pred:<20s} {'✓' if sg_pred==expected else '✗'}")
        row = {"query": f"{a}:{b}::{c}:?", "expected": expected,
               "CBOW": cbow_pred, "Skip-gram": sg_pred}
        if g_cbow:
            gc = g_cbow.analogy(a, b, c)
            row["Gensim CBOW"] = gc[0][0] if gc else "OOV"
            LOG.write(f"    Gensim CBOW     : {row['Gensim CBOW']:<20s} {'✓' if row['Gensim CBOW']==expected else '✗'}")
        if g_sg:
            gs = g_sg.analogy(a, b, c)
            row["Gensim SG"] = gs[0][0] if gs else "OOV"
            LOG.write(f"    Gensim SkipGram : {row['Gensim SG']:<20s} {'✓' if row['Gensim SG']==expected else '✗'}")
        rows.append(row)

    LOG.write("\nANALOGY ACCURACY SUMMARY")
    for key in ["CBOW", "Skip-gram", "Gensim CBOW", "Gensim SG"]:
        scored = [r for r in rows if key in r]
        if not scored:
            continue
        correct = sum(1 for r in scored if r.get(key) == r["expected"])
        LOG.write(f"  {key:<22s}: {correct}/{len(scored)}  ({100*correct/len(scored):.0f}%)")
    return rows


def plot_analogy_table(rows, g_cbow=None, g_sg=None):
    col_labels = ["Analogy", "Expected", "CBOW (Scratch)", "Skip-gram (Scratch)"]
    score_keys = ["CBOW", "Skip-gram"]
    if g_cbow: col_labels.append("Gensim CBOW");  score_keys.append("Gensim CBOW")
    if g_sg:   col_labels.append("Gensim SG");    score_keys.append("Gensim SG")
    table_data  = [[r["query"], r["expected"]] + [r.get(k,"—") for k in score_keys] for r in rows]
    cell_colors = []
    for r in rows:
        row_c = ["#f0f0f0", "#ddeeff"]
        for k in score_keys:
            row_c.append("#d4edda" if r.get(k) == r["expected"] else "#f8d7da")
        cell_colors.append(row_c)
    fig, ax = plt.subplots(figsize=(10, 0.7*len(rows)+2))
    ax.axis("off")
    fig.suptitle("Analogy Experiment Results", fontsize=13, fontweight="bold")
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellColours=cell_colors, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2563eb")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#cccccc")
    plt.tight_layout()
    plt.savefig("plot_analogies.png", dpi=150, bbox_inches="tight")
    plt.close()
    LOG.write("Saved → plot_analogies.png")


def plot_neighbors_chart(vocab, cbow, sg, g_cbow=None, g_sg=None):
    panels  = [(cbow, "CBOW (Scratch)"), (sg, "Skip-gram (Scratch)")]
    if g_cbow: panels.append((g_cbow, "Gensim CBOW"))
    if g_sg:   panels.append((g_sg,   "Gensim Skip-gram"))
    n_words = len(PROBE_WORDS)
    n_cols  = len(panels)
    fig, axes = plt.subplots(n_words, n_cols, figsize=(6*n_cols, 3.5*n_words))
    if n_words == 1:
        axes = [axes]
    fig.suptitle("Top-5 Nearest Neighbours (Cosine Similarity)", fontsize=14, fontweight="bold")
    for row, word in enumerate(PROBE_WORDS):
        for col, (model, title) in enumerate(panels):
            ax    = axes[row][col]
            pairs = (model.neighbors(word, n=5) if isinstance(model, GensimWrapper)
                     else model.neighbors(word, vocab, n=5))
            if pairs:
                words_nb = [p[0] for p in pairs]
                sims     = [p[1] for p in pairs]
                colors   = plt.cm.Blues(np.linspace(0.4, 0.9, len(words_nb)))
                bars     = ax.barh(range(len(words_nb)), sims, color=colors)
                ax.set_yticks(range(len(words_nb))); ax.set_yticklabels(words_nb, fontsize=9)
                ax.set_xlim(0, 1)
                for bar, s in zip(bars, sims):
                    ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                            f"{s:.3f}", va="center", fontsize=8)
            else:
                ax.text(0.5, 0.5, f'"{word}" not in vocab',
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_yticks([])
            ax.invert_yaxis()
            if row == 0:          ax.set_title(title, fontsize=10, fontweight="bold")
            if col == 0:          ax.set_ylabel(f'"{word}"', fontsize=10, fontweight="bold")
            if row == n_words-1:  ax.set_xlabel("Cosine Sim")
            ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_neighbors.png", dpi=150, bbox_inches="tight")
    plt.close()
    LOG.write("Saved → plot_neighbors.png")


def run_task3(vocab, cbow, sg, g_cbow=None, g_sg=None):
    LOG.section("TASK 3 — SEMANTIC ANALYSIS")
    run_neighbors(vocab, cbow, sg, g_cbow, g_sg)
    rows = run_analogies(vocab, cbow, sg, g_cbow, g_sg)
    plot_neighbors_chart(vocab, cbow, sg, g_cbow, g_sg)
    plot_analogy_table(rows, g_cbow, g_sg)


# TASK 4 — VISUALIZATION

WORD_GROUPS = {
    "Academic Roles": ["professor", "student", "faculty", "researcher", "phd"],
    "Programs":       ["btech", "mtech", "undergraduate", "postgraduate", "degree"],
    "Departments":    ["engineering", "science", "mathematics", "physics", "chemistry"],
    "Activities":     ["research", "exam", "lecture", "laboratory", "thesis"],
}
ALL_WORDS  = [w for words in WORD_GROUPS.values() for w in words]
ALL_GROUPS = [g for g, words in WORD_GROUPS.items() for _ in words]

GROUP_COLORS  = {"Academic Roles":"#2563eb", "Programs":"#dc2626",
                 "Departments":"#16a34a",    "Activities":"#d97706"}
GROUP_MARKERS = {"Academic Roles":"o", "Programs":"s",
                 "Departments":"^",    "Activities":"D"}


def extract_scratch(model, vocab):
    words, groups, vecs = [], [], []
    for w, g in zip(ALL_WORDS, ALL_GROUPS):
        if w in vocab.w2i:
            words.append(w); groups.append(g)
            vecs.append(model.W_in[vocab.w2i[w]])
    return words, groups, np.array(vecs, dtype=np.float32)


def draw_scatter(ax, coords, words, groups, title):
    for w, g, (x, y) in zip(words, groups, coords):
        ax.scatter(x, y, color=GROUP_COLORS[g], marker=GROUP_MARKERS[g],
                   s=90, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(w, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color="#222222")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    for g, color in GROUP_COLORS.items():
        pts = np.array([c for c, grp in zip(coords, groups) if grp == g])
        if len(pts) < 2:
            continue
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        ax.scatter(cx, cy, color=color, s=20, marker="+", linewidths=1.5, zorder=2, alpha=0.6)


def add_legend(fig):
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in WORD_GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))


def plot_pca(vocab, cbow, sg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("PCA — Word Embedding Clusters", fontsize=14, fontweight="bold", y=1.01)
    panels = [("CBOW (Scratch)", *extract_scratch(cbow, vocab)),
              ("Skip-gram (Scratch)", *extract_scratch(sg, vocab))]
    pca = PCA(n_components=2, random_state=42)
    for ax, (title, words, groups, vecs) in zip(axes, panels):
        if len(vecs) < 2:
            ax.text(0.5, 0.5, "not enough vocab", ha="center", va="center", transform=ax.transAxes)
            continue
        draw_scatter(ax, pca.fit_transform(vecs), words, groups, title)
    add_legend(fig)
    plt.tight_layout()
    plt.savefig("plot_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    LOG.write("Saved → plot_pca.png")


def plot_tsne(vocab, cbow, sg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("t-SNE — Word Embedding Clusters", fontsize=14, fontweight="bold", y=1.01)
    panels = [("CBOW (Scratch)", *extract_scratch(cbow, vocab)),
              ("Skip-gram (Scratch)", *extract_scratch(sg, vocab))]
    for ax, (title, words, groups, vecs) in zip(axes, panels):
        if len(vecs) < 4:
            ax.text(0.5, 0.5, "not enough vocab", ha="center", va="center", transform=ax.transAxes)
            continue
        perp  = min(5, len(vecs)-1)
        tsne  = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000, init="pca")
        draw_scatter(ax, tsne.fit_transform(vecs), words, groups, title)
    add_legend(fig)
    plt.tight_layout()
    plt.savefig("plot_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()
    LOG.write("Saved → plot_tsne.png")


def run_task4(vocab, cbow, sg):
    LOG.section("TASK 4 — VISUALIZATION")
    if not SKLEARN:
        LOG.write("[ERROR] scikit-learn not installed — pip install scikit-learn")
        return
    plot_pca(vocab, cbow, sg)
    plot_tsne(vocab, cbow, sg)


# Entry point

def main():
    # Task 1: load corpus.txt if it exists, otherwise scrape the website
    sents = run_task1()
    if sents is None:
        LOG.write("[FATAL] No corpus available — exiting")
        return

    # Task 2: load saved models if they exist, otherwise train and save them
    vocab, cbow, sg, g_cbow, g_sg = run_task2(sents)

    # Task 3: always run full semantic analysis and log all results
    run_task3(vocab, cbow, sg, g_cbow, g_sg)

    # Task 4: always regenerate visualization plots
    run_task4(vocab, cbow, sg)

    LOG.section("ALL TASKS COMPLETE")
    LOG.write("Files produced this run:")
    for f in ["corpus.txt","documents.json","stats_report.txt",
              "cbow_embeddings.npy","sg_embeddings.npy","vocab.pkl",
              "gensim_cbow.model","gensim_sg.model",
              "plot_train_time.png","plot_neighbors.png",
              "plot_analogies.png","plot_pca.png","plot_tsne.png","run_log.txt"]:
        status = "✓" if os.path.exists(f) else "—"
        LOG.write(f"  {status}  {f}")


if __name__ == "__main__":
    main()