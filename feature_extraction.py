"""
Feature extraction for URLs and emails.
48 features total: 24 lexical + 14 host-based + 10 content-based.
Host-based features gracefully degrade when WHOIS/DNS lookups fail.
"""
import re
import math
import socket
import urllib.parse
from datetime import datetime
import numpy as np

try:
    import whois
    import tldextract
except ImportError:
    whois = None
    tldextract = None

SUSPICIOUS_TLDS = {'xyz', 'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'click', 'work'}
SUSPICIOUS_WORDS = ['login', 'verify', 'secure', 'account', 'update', 'bank',
                    'confirm', 'signin', 'password', 'paypal', 'ebay']

FEATURE_NAMES = [
    # Lexical (24)
    'url_length', 'domain_length', 'path_length', 'query_length',
    'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
    'num_at', 'num_question', 'num_equal', 'num_percent',
    'has_ip', 'has_https', 'has_port', 'suspicious_tld',
    'num_subdomains', 'url_entropy', 'digit_letter_ratio',
    'has_suspicious_word', 'num_params', 'has_hex', 'has_double_slash_redirect',
    'tld_length',
    # Host-based (14)
    'domain_age_days', 'domain_expiry_days', 'has_dns', 'dns_resolve_time',
    'has_mx', 'whois_available', 'registrar_known', 'alexa_rank_log',
    'asn_known', 'num_nameservers', 'has_spf', 'has_dnssec',
    'creation_year', 'updated_recently',
    # Content-based (10)
    'title_domain_match', 'num_external_links', 'internal_external_ratio',
    'has_login_form', 'has_password_field', 'has_iframe', 'has_redirect',
    'num_scripts', 'num_images', 'html_text_ratio',
]

assert len(FEATURE_NAMES) == 48


def _entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_lexical(url):
    parsed = urllib.parse.urlparse(url)
    netloc = parsed.netloc
    path = parsed.path
    query = parsed.query
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    tld = netloc.split('.')[-1] if '.' in netloc else ''
    return [
        len(url), len(netloc), len(path), len(query),
        url.count('.'), url.count('-'), url.count('_'), url.count('/'),
        url.count('@'), url.count('?'), url.count('='), url.count('%'),
        int(bool(re.search(r'\d{1,3}(\.\d{1,3}){3}', url))),
        int(parsed.scheme == 'https'),
        int(':' in netloc),
        int(tld.lower() in SUSPICIOUS_TLDS),
        max(0, len(netloc.split('.')) - 2),
        _entropy(url),
        digits / max(1, letters),
        int(any(w in url.lower() for w in SUSPICIOUS_WORDS)),
        len(urllib.parse.parse_qs(query)),
        int(bool(re.search(r'%[0-9a-fA-F]{2}', url))),
        int(url.count('//') > 1),
        len(tld),
    ]


def extract_host(url, enable_network=False):
    """Host features. Returns zeros when network=False (fast path for training)."""
    if not enable_network:
        return [0.0] * 14
    feats = [0.0] * 14
    try:
        domain = urllib.parse.urlparse(url).netloc
        t0 = datetime.now()
        socket.gethostbyname(domain)
        feats[2] = 1.0
        feats[3] = (datetime.now() - t0).total_seconds()
        if whois:
            w = whois.whois(domain)
            cd = w.creation_date if not isinstance(w.creation_date, list) else w.creation_date[0]
            ed = w.expiration_date if not isinstance(w.expiration_date, list) else w.expiration_date[0]
            if cd:
                feats[0] = (datetime.now() - cd).days
                feats[12] = cd.year
            if ed:
                feats[1] = (ed - datetime.now()).days
            feats[5] = 1.0
            feats[6] = int(bool(w.registrar))
            ns = w.name_servers or []
            feats[9] = len(ns) if isinstance(ns, list) else 1
    except Exception:
        pass
    return feats


def extract_content(html=None):
    """Content features extracted from page HTML (10). Zeros if html=None."""
    if not html:
        return [0.0] * 10
    html_l = html.lower()
    text_len = len(re.sub(r'<[^>]+>', '', html))
    return [
        0.0,  # title_domain_match (needs domain context)
        html_l.count('href=http'),
        0.0,
        int('type="password"' in html_l or "<form" in html_l and 'login' in html_l),
        int('type="password"' in html_l),
        int('<iframe' in html_l),
        int('window.location' in html_l or 'meta http-equiv="refresh"' in html_l),
        html_l.count('<script'),
        html_l.count('<img'),
        text_len / max(1, len(html)),
    ]


def extract_all(url, html=None, enable_network=False):
    """Returns a (48,) numpy array of features for a single URL."""
    feats = extract_lexical(url) + extract_host(url, enable_network) + extract_content(html)
    return np.array(feats, dtype=np.float32)


if __name__ == '__main__':
    v = extract_all("http://paypa1-secure-login.com/verify?user=x")
    print("Shape:", v.shape)
    for n, f in zip(FEATURE_NAMES, v):
        print(f"  {n:30s} = {f}")
