"""WebSearch (DuckDuckGo HTML) and Fetch (page text extraction)."""

import requests
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from rich.panel import Panel

from rich.markup import escape

from bardgent.config import console
from bardgent.permissions import is_tool_permitted
from bardgent.state import ask_approval
from bardgent.utils import with_retries


def WebSearch(query):
    console.print(f'Web Search: [bold green]{query}[/bold green]\n')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = with_retries(
            requests.post, 'https://html.duckduckgo.com/html/',
            data={'q': query}, headers=headers, timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Web search failed after retries: {type(e).__name__}: {e}"
    soup = BeautifulSoup(resp.text, 'html.parser')
    results = []
    for r in soup.select('.result')[:8]:
        a = r.select_one('a.result__a')
        if not a:
            continue
        url = a.get('href', '')
        uddg = parse_qs(urlparse(url).query).get('uddg')
        if uddg:
            url = uddg[0]
        snippet = r.select_one('.result__snippet')
        entry = f"{a.get_text(strip=True)}\n{url}"
        if snippet:
            entry += f"\n{snippet.get_text(strip=True)}"
        results.append(entry)
    return '\n\n'.join(results) if results else '(no results)'


def Fetch(link, state):
    console.print(Panel(escape(link), title='[bold yellow]Fetch wants to run', border_style='yellow'))

    if not is_tool_permitted('Fetch') and not ask_approval(state, 'Fetch', 'Fetch this page?'):
        return 'Fetch rejected by user.'

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 Chrome/120 Safari/537.36'
        )
    }

    try:
        resp = with_retries(requests.get, link, headers=headers, timeout=10)
        if resp.status_code == 403:
            return f"Could not fetch page (403 Forbidden): {link}"
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Fetch failed after retries: {type(e).__name__}: {e}"

    soup = BeautifulSoup(resp.text, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)