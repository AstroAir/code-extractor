from __future__ import annotations

from pysearch.search.semantic_advanced import CodeSemanticAnalyzer, SemanticEmbedding, SemanticSearchEngine


def test_semantic_embedding_basic():
    emb = SemanticEmbedding()
    docs = ["def foo():\n    pass\n", "class Bar:\n    pass\n"]
    emb.fit(docs)
    v1 = emb.transform("def foo(): pass")
    v2 = emb.transform("class Bar: pass")
    s = emb.cosine_similarity(v1, v2)
    assert 0.0 <= s <= 1.0


def test_semantic_analyzer_extract_and_similarity():
    analyzer = CodeSemanticAnalyzer()
    content = (
        "\n# data processing utilities\n\n"
        "class Bar:\n"
        "    def baz(self):\n"
        "        \"\"\"Doc\"\"\"\n"
        "        pass\n"
    )
    concepts = analyzer.extract_concepts(content)
    score = analyzer.calculate_similarity("data processing", concepts)
    assert 0.0 <= score <= 1.0


def test_semantic_search_engine_threshold():
    eng = SemanticSearchEngine()
    content = "def alpha():\n    pass\n"
    matches = eng.search_semantic("nonexistentconcept", content, threshold=0.9)
    assert matches == []

