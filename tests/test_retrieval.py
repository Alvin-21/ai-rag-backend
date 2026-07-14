import pytest
from services.retrieval import search_knowledge_base

class TestRetrieval:
    """Retrieval system tests"""

    def test_search_finds_relevant_articles(self):
        """Given a question, search returns relevant articles"""
        results = search_knowledge_base("How long does shipping take?")
        
        assert len(results) > 0
        assert any(
            "shipping" in r["title"].lower() or "shipping" in r["content"].lower()
            for r in results
        )

    def test_search_respects_top_k(self):
        """Search returns at most top_k results"""
        results = search_knowledge_base("refund policy", top_k=2)
        assert len(results) <= 2

    def test_search_returns_empty_for_no_matches(self):
        """Given a query with no matches, returns empty list"""
        results = search_knowledge_base("xyzzyyx_123_456", top_k=5)
        assert results == []

    def test_search_is_case_insensitive(self):
        """Search works regardless of case"""
        results1 = search_knowledge_base("REFUND")
        results2 = search_knowledge_base("refund")
        results3 = search_knowledge_base("ReFuNd")
        
        assert len(results1) > 0
        assert results1 == results2 == results3

    def test_search_handles_punctuation(self):
        """Search ignores punctuation"""
        results1 = search_knowledge_base("Can I get a refund?")
        results2 = search_knowledge_base("Can I get a refund")
        
        assert results1 == results2