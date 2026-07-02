import json
from pathlib import Path
from typing import List, Dict

# Load the JSON file once when the module is imported (not on every request)
DATA_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"

with open(DATA_PATH, "r") as f:
    KNOWLEDGE_BASE: List[Dict] = json.load(f)


def tokenize(text: str) -> set:
    """
    Split text into lowercase words and remove punctuation.
    
    "What's the Refund Policy?" → {"whats", "the", "refund", "policy"}
    
    This lets us compare words regardless of capitalization or punctuation.
    """
    # Remove punctuation and lowercase everything
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return set(cleaned.split())


def score_article(question_tokens: set, article: Dict) -> float:
    """
    Calculate a relevance score for one article.
    
    We check overlap with:
    - The article title
    - The article content  
    - Manually set tags
    
    Tags get a higher weight because they're hand-picked for this purpose.
    """
    score = 0.0

    # Check title overlap
    title_tokens = tokenize(article["title"])
    score += len(question_tokens & title_tokens) * 2.0  # title matches are worth more

    # Check content overlap
    content_tokens = tokenize(article["content"])
    score += len(question_tokens & content_tokens) * 1.0

    # Check tag overlap — tags are curated keywords, so give them extra weight
    for tag in article.get("tags", []):
        tag_tokens = tokenize(tag)
        score += len(question_tokens & tag_tokens) * 3.0  # highest weight

    return score


def search_knowledge_base(question: str, top_k: int = 2) -> List[Dict]:
    """
    Main retrieval function.
    
    Returns the top_k most relevant articles for the given question.
    If no articles score above 0, returns an empty list — meaning
    the AI will say it doesn't have information on this topic.
    
    Example:
        question = "Can I get a refund?"
        → finds "Refund Policy" article with score 9.0
        → returns [{"title": "Refund Policy", "content": "..."}]
    """
    question_tokens = tokenize(question)

    # Score every article
    scored = [
        (article, score_article(question_tokens, article))
        for article in KNOWLEDGE_BASE
    ]

    # Sort by score, highest first
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return only articles with score > 0 (at least one keyword matched)
    relevant = [article for article, score in scored if score > 0]

    return relevant[:top_k]