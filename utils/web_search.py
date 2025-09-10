"""
Live web search integration.
Supports SerpAPI and Tavily. Falls back to DuckDuckGo html if no API key (basic).
"""

import os, json, traceback, time, urllib.parse, requests
from typing import List, Dict
from config.config import SERPAPI_API_KEY, TAVILY_API_KEY

USER_AGENT = "NeoStatsBot/1.0 (+https://example.com)"

def serpapi_search(q: str, num: int = 5) -> List[Dict]:
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY not set")
    url = "https://serpapi.com/search.json"
    params = {"q": q, "num": num, "api_key": SERPAPI_API_KEY}
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("organic_results", [])[:num]:
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
        })
    return results

def tavily_search(q: str, num: int = 5) -> List[Dict]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY not set")
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": q, "max_results": num}
    r = requests.post(url, json=payload, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", [])[:num]:
        results.append({
            "title": item.get("title"),
            "link": item.get("url"),
            "snippet": item.get("content", "")[:280],
        })
    return results

def ddg_fallback(q: str, num: int = 5) -> List[Dict]:
    # Extremely simple fallback using DuckDuckGo's HTML (no key). Not guaranteed.
    url = f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(q)}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    from bs4 import BeautifulSoup  # lightweight parser
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.select(".result__a")[:num]:
        title = a.get_text(strip=True)
        link = a.get("href")
        out.append({"title": title, "link": link, "snippet": ""})
    return out

def web_search(query: str, num: int = 5) -> List[Dict]:
    """
    Try providers in order: SerpAPI -> Tavily -> DDG fallback.
    """
    providers = []
    if SERPAPI_API_KEY:
        providers.append(("serpapi", serpapi_search))
    if TAVILY_API_KEY:
        providers.append(("tavily", tavily_search))
    providers.append(("ddg", ddg_fallback))

    last_err = None
    for name, fn in providers:
        try:
            return fn(query, num=num)
        except Exception as e:
            last_err = e
            continue
    return [{"title": "Search failed", "link": "", "snippet": f"{last_err}"}]
