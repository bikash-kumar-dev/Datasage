# agents/search_agent.py

from agents.utils import info, warn
import re

# Try to import new ddgs package; if not available we'll fall back to prebuilt responses
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False


class SearchAgent:
    """
    Simple search agent that queries DuckDuckGo (ddgs) when available,
    prints top results and generates a short/medium/long extractive summary
    from the result snippets.

    It also stores the last search so prototype can support quick follow-ups
    (e.g., "tell me more", "explain", "give details").
    """

    def __init__(self):
        self.last_query = None
        self.last_results = None
        self.last_summary = None

    def _collect_sentences(self, texts):
        """
        Split a list of text snippets into sentences and return a list of
        cleaned sentences (preserve order).
        """
        sentences = []
        for t in texts:
            if not t:
                continue
            # naive sentence split
            parts = re.split(r'(?<=[.!?])\s+', t.strip())
            for p in parts:
                p = p.strip()
                if p and p not in sentences:
                    sentences.append(p)
        return sentences

    def _summarize(self, snippets, level="medium"):
        """
        Simple extractive summarizer:
        - collects sentences from snippets (in order),
        - returns first N sentences depending on level.
        Levels: short (1), medium (3), long (6)
        """
        if not snippets:
            return "No text to summarize."

        sentences = self._collect_sentences(snippets)

        if level == "short":
            n = 1
        elif level == "long":
            n = 6
        else:  # medium
            n = 3

        # if less sentences than n, just join what we have
        selected = sentences[:n]
        summary = " ".join(selected)
        return summary if summary else "No summary available."

    def search_and_explain(self, query: str):
        """
        Perform a web search (if ddgs present), print results and return a dict:
        { "query": query, "results": [...], "summary": "..."}
        """
        query = (query or "").strip()
        if not query:
            warn("Empty search query.")
            return None

        info(f"Searching online for: {query}\n")
        self.last_query = query

        # Prebuilt quick explains (fast fallback)
        qlow = query.lower()
        if "overfitting" in qlow:
            summary = ("Overfitting happens when a model memorizes noise and training examples "
                       "instead of learning general patterns. Fixes include cross-validation, "
                       "regularization, more data, early stopping and pruning.")
            print(summary)
            self.last_results = []
            self.last_summary = summary
            return {"query": query, "results": [], "summary": summary}

        # If ddgs not available, fall back to a helpful message
        if not DDGS_AVAILABLE:
            warn("ddgs package not available — returning a short explanation or ask again.")
            fallback = f"Couldn't perform live web search (ddgs not installed). Try: '{query}'."
            print(fallback)
            self.last_results = []
            self.last_summary = fallback
            return {"query": query, "results": [], "summary": fallback}

        # Run the query with ddgs
        results = []
        try:
            with DDGS() as ddgs:
                # ddgs.text returns an iterator of dict-like results; limit to 5
                for i, item in enumerate(ddgs.text(query, max_results=6)):
                    if i >= 6:
                        break
                    # ddgs returns fields like 'title', 'href', 'body'
                    results.append({
                        "title": item.get("title") or "",
                        "link": item.get("href") or "",
                        "snippet": item.get("body") or ""
                    })
        except Exception as e:
            warn(f"Search failed: {e}")
            # store failure
            self.last_results = []
            self.last_summary = f"Search failed: {e}"
            return {"query": query, "results": [], "summary": str(e)}

        # Print results in readable format
        if not results:
            print("No live results found. Try another query.")
            self.last_results = []
            self.last_summary = "No live results found."
            return {"query": query, "results": [], "summary": "No live results found."}

        print("🌍 Real-Time Internet Search Results:\n")
        snippets = []
        for idx, r in enumerate(results, start=1):
            title = r.get("title", "No title")
            snippet = r.get("snippet", "").strip()
            link = r.get("link", "")
            print(f"{idx}. {title}")
            if snippet:
                # keep printed snippet short
                print(snippet[:400] + ("..." if len(snippet) > 400 else ""))
            if link:
                print(link)
            print("-" * 60)
            snippets.append(snippet)

        # Create an automatic medium summary
        summary = self._summarize(snippets, level="medium")

        print("\n📝 Automatic summary (medium):")
        print(summary)
        print("\n(Ask 'tell me more' or just type a follow-up question to get a longer explanation.)\n")

        # Save last results and summary so prototype can handle follow-ups
        self.last_results = results
        self.last_summary = summary

        return {"query": query, "results": results, "summary": summary}

    def follow_up(self, text: str):
        """
        Use last_results to answer follow-up requests. If no last_results, return False.
        Recognizes keywords for 'more detail' and returns a longer summary.
        """
        if not self.last_results:
            return False

        t = (text or "").lower()
        if any(k in t for k in ["more", "details", "explain", "how", "why", "tell me"]):
            # produce a longer summary
            snippets = [r.get("snippet", "") for r in self.last_results]
            long_summary = self._summarize(snippets, level="long")
            print("\n🧾 Extended summary (long):\n")
            print(long_summary)
            print()
            return True

        # If user asked a direct question referencing a keyword present in snippets,
        # try a naive keyword highlight: return sentences that contain that keyword.
        for word in t.split():
            matches = []
            for s in self._collect_sentences([r.get("snippet", "") for r in self.last_results]):
                if word in s.lower():
                    matches.append(s)
            if matches:
                print(f"\n🔎 Sentences containing '{word}':\n")
                for m in matches[:6]:
                    print(m)
                print()
                return True

        return False
