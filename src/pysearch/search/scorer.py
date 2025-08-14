from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..core.config import SearchConfig
from .semantic import semantic_similarity_score
from ..core.types import SearchItem


class RankingStrategy(str, Enum):
    """Different ranking strategies for search results."""

    RELEVANCE = "relevance"
    FREQUENCY = "frequency"
    RECENCY = "recency"
    POPULARITY = "popularity"
    HYBRID = "hybrid"


@dataclass
class ScoringWeights:
    """Configurable weights for different scoring factors."""

    text_match: float = 1.0
    match_density: float = 0.5
    position_bonus: float = 0.3
    file_type_bonus: float = 0.2
    semantic_similarity: float = 0.4
    code_structure: float = 0.3
    file_popularity: float = 0.1
    directory_depth_penalty: float = 0.1
    exact_match_bonus: float = 0.5
    case_match_bonus: float = 0.2


def score_item(
    item: SearchItem, cfg: SearchConfig, query_text: str = "", all_files: set[Path] | None = None
) -> float:
    """
    Enhanced scoring model with configurable weights and multiple ranking factors:
    - Text match frequency and quality (exact matches, case sensitivity)
    - Match density and distribution
    - Code structure analysis (functions, classes, comments)
    - File type and language-specific relevance
    - Semantic similarity
    - File popularity and importance indicators
    - Position and context bonuses
    """
    if all_files is None:
        all_files = set()

    weights = ScoringWeights()  # Use default weights, could be configurable

    # Enhanced text matching score
    # Base score is proportional to the number of pattern matches found
    text_hits = len(item.match_spans)
    text_score = text_hits * weights.text_match

    # Exact match bonus - reward perfect phrase matches
    exact_match_bonus = 0.0
    case_match_bonus = 0.0
    if query_text and item.lines:
        content = "\n".join(item.lines).lower()
        query_lower = query_text.lower()

        # Check for exact phrase matches (case-insensitive)
        # This rewards results where the entire query appears as a phrase
        if query_lower in content:
            exact_match_bonus = weights.exact_match_bonus

        # Check for case-sensitive matches
        # This gives additional points for preserving original case
        if query_text in "\n".join(item.lines):
            case_match_bonus = weights.case_match_bonus

    # Enhanced match density with distribution analysis
    # Calculate how concentrated the matches are within the result block
    block_length = max(1, item.end_line - item.start_line + 1)
    density_score = 0.0
    if text_hits > 0:
        # Basic density: matches per line in the result block
        # Higher density indicates more relevant content
        density = text_hits / block_length
        density_score = min(2.0, density) * weights.match_density  # Cap at 2.0 to prevent dominance

        # Distribution bonus: prefer evenly distributed matches over clustered ones
        # This helps identify code blocks where the pattern appears throughout
        if len(item.match_spans) > 1:
            # MatchSpan is (line_index, (start_col, end_col))
            # Extract column positions to analyze horizontal distribution
            span_positions = [span[1][0] for span in item.match_spans]  # start_col positions
            span_range = max(span_positions) - min(span_positions)
            if span_range > 0:
                # Normalize by line length to get distribution ratio
                line_length = len(item.lines[0]) if item.lines else 100
                distribution_bonus = min(0.5, span_range / line_length)
                density_score += distribution_bonus

    # Code structure analysis
    structure_bonus = _analyze_code_structure(item, query_text) * weights.code_structure

    # Enhanced file type scoring with language detection
    file_type_score = _calculate_file_type_score(item.file) * weights.file_type_bonus

    # Position scoring with context awareness
    position_score = _calculate_position_score(item, block_length) * weights.position_bonus

    # Enhanced semantic similarity
    # Use lightweight semantic analysis to find conceptually related content
    # even when exact text matches are not present
    semantic_score = 0.0
    if query_text and item.lines:
        content = "\n".join(item.lines)
        semantic_score = (
            semantic_similarity_score(content, query_text) * weights.semantic_similarity
        )

    # File popularity and importance
    # Boost scores for files that appear frequently in search results
    # or have characteristics indicating importance (e.g., main files, configs)
    popularity_score = _calculate_popularity_score(item.file, all_files) * weights.file_popularity

    # Directory depth penalty (prefer files closer to root)
    # Files deeper in the directory structure are often less important
    # Apply a small penalty to encourage results from top-level directories
    depth_penalty = len(item.file.parts) * weights.directory_depth_penalty

    # Combine all scores using weighted sum
    # Each component contributes to the final relevance score
    # The weights allow fine-tuning the relative importance of different factors
    total_score = (
        text_score  # Base relevance from pattern matches
        + exact_match_bonus  # Bonus for exact phrase matches
        + case_match_bonus  # Bonus for case-sensitive matches
        + density_score  # Bonus for high match density and distribution
        + structure_bonus  # Bonus for matches in important code structures
        + file_type_score  # Bonus for relevant file types
        + position_score  # Bonus for matches in important positions
        + semantic_score  # Bonus for semantic similarity
        + popularity_score  # Bonus for popular/important files
        - depth_penalty  # Penalty for deeply nested files
    )

    # Ensure score is non-negative
    return max(0.0, total_score)


def _analyze_code_structure(item: SearchItem, query_text: str) -> float:
    """Analyze code structure to give bonus for matches in important locations."""
    if not item.lines:
        return 0.0

    content = "\n".join(item.lines)
    structure_bonus = 0.0

    # Function/method definition bonus
    if re.search(
        r"^\s*(def|function|func)\s+\w*" + re.escape(query_text),
        content,
        re.MULTILINE | re.IGNORECASE,
    ):
        structure_bonus += 0.5

    # Class definition bonus
    if re.search(
        r"^\s*(class|interface|struct)\s+\w*" + re.escape(query_text),
        content,
        re.MULTILINE | re.IGNORECASE,
    ):
        structure_bonus += 0.6

    # Variable/constant definition bonus
    if re.search(
        r"^\s*\w*" + re.escape(query_text) + r"\w*\s*[=:]", content, re.MULTILINE | re.IGNORECASE
    ):
        structure_bonus += 0.3

    # Import/include statement bonus
    if re.search(
        r"^\s*(import|include|require|from)\s+.*" + re.escape(query_text),
        content,
        re.MULTILINE | re.IGNORECASE,
    ):
        structure_bonus += 0.4

    # Comment/documentation bonus (lower priority)
    if re.search(r"^\s*[#/*]+.*" + re.escape(query_text), content, re.MULTILINE | re.IGNORECASE):
        structure_bonus += 0.2

    return min(1.0, structure_bonus)


def _calculate_file_type_score(file_path: Path) -> float:
    """Calculate file type relevance score with language-specific bonuses."""
    file_ext = file_path.suffix.lower()
    file_name = file_path.name.lower()

    # Programming language files (highest priority)
    if file_ext in [".py", ".pyx", ".pyi"]:
        return 1.2
    elif file_ext in [".js", ".ts", ".jsx", ".tsx"]:
        return 1.15
    elif file_ext in [".java", ".kt", ".scala"]:
        return 1.1
    elif file_ext in [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"]:
        return 1.1
    elif file_ext in [".cs", ".go", ".rs", ".rb", ".php", ".swift"]:
        return 1.05

    # Configuration and data files
    elif file_ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"]:
        return 0.9
    elif file_ext in [".xml", ".html", ".css", ".scss", ".less"]:
        return 0.85

    # Documentation files
    elif file_ext in [".md", ".rst", ".txt", ".doc", ".docx"]:
        return 0.8

    # Special files
    elif file_name in ["dockerfile", "makefile", "rakefile", "gemfile"]:
        return 1.0
    elif file_name.startswith("readme"):
        return 0.9

    # Build and package files
    elif file_ext in [".gradle", ".maven", ".sbt"] or file_name in [
        "package.json",
        "cargo.toml",
        "setup.py",
    ]:
        return 0.7

    # Default for unknown types
    else:
        return 0.6


def _calculate_position_score(item: SearchItem, block_length: int) -> float:
    """Calculate position-based scoring with context awareness."""
    # Early position bonus (matches near top of file are often more important)
    position_bonus = 1.0 / (1.0 + math.log(max(1, item.start_line)) * 0.1)

    # Block focus bonus (shorter, focused blocks are better)
    focus_bonus = 1.0 / (1.0 + math.log(max(1, block_length / 5.0)))

    # Header/top-level bonus
    if item.start_line <= 10:
        position_bonus += 0.3
    elif item.start_line <= 50:
        position_bonus += 0.1

    return position_bonus + focus_bonus


def _calculate_popularity_score(file_path: Path, all_files: set[Path]) -> float:
    """Calculate file popularity and importance score."""
    popularity_score = 0.0
    file_name = file_path.name.lower()
    file_stem = file_path.stem.lower()

    # Important file names
    important_names = {
        "main",
        "index",
        "app",
        "core",
        "base",
        "common",
        "util",
        "utils",
        "helper",
        "helpers",
        "config",
        "settings",
        "constants",
        "api",
    }

    # Check for important name patterns
    name_parts = file_stem.replace("-", "_").split("_")
    if any(part in important_names for part in name_parts):
        popularity_score += 0.3

    # Special file patterns
    if file_name.startswith("__init__"):
        popularity_score += 0.4  # Python package files
    elif file_name in ["index.js", "index.ts", "main.py", "app.py"]:
        popularity_score += 0.5  # Entry point files
    elif "test" in file_name or "spec" in file_name:
        popularity_score -= 0.2  # Test files are less important for general searches

    # Directory-based importance
    parts = [part.lower() for part in file_path.parts]
    if "src" in parts or "lib" in parts:
        popularity_score += 0.2
    elif "test" in parts or "tests" in parts:
        popularity_score -= 0.1
    elif "vendor" in parts or "node_modules" in parts:
        popularity_score -= 0.3

    return max(0.0, popularity_score)


def sort_items(
    items: list[SearchItem],
    cfg: SearchConfig,
    query_text: str = "",
    strategy: RankingStrategy = RankingStrategy.HYBRID,
) -> list[SearchItem]:
    """Sort items by relevance score with configurable ranking strategies."""
    if not items:
        return items

    # Collect all files for popularity scoring
    all_files = {item.file for item in items}

    if strategy == RankingStrategy.RELEVANCE:
        return _sort_by_relevance(items, cfg, query_text, all_files)
    elif strategy == RankingStrategy.FREQUENCY:
        return _sort_by_frequency(items, cfg, query_text, all_files)
    elif strategy == RankingStrategy.RECENCY:
        return _sort_by_recency(items, cfg, query_text, all_files)
    elif strategy == RankingStrategy.POPULARITY:
        return _sort_by_popularity(items, cfg, query_text, all_files)
    else:  # HYBRID
        return _sort_hybrid(items, cfg, query_text, all_files)


def _sort_by_relevance(
    items: list[SearchItem], cfg: SearchConfig, query_text: str, all_files: set[Path]
) -> list[SearchItem]:
    """Sort by pure relevance score."""
    scored_items = []
    for item in items:
        score = score_item(item, cfg, query_text, all_files)
        scored_items.append((score, item))

    scored_items.sort(key=lambda x: (-x[0], x[1].file.as_posix(), x[1].start_line))
    return [item for _, item in scored_items]


def _sort_by_frequency(
    items: list[SearchItem], cfg: SearchConfig, query_text: str, all_files: set[Path]
) -> list[SearchItem]:
    """Sort by match frequency (number of matches per item)."""
    scored_items = []
    for item in items:
        frequency_score = len(item.match_spans)
        relevance_score = score_item(item, cfg, query_text, all_files)
        # Combine frequency and relevance
        combined_score = frequency_score * 2.0 + relevance_score * 0.5
        scored_items.append((combined_score, item))

    scored_items.sort(key=lambda x: (-x[0], x[1].file.as_posix(), x[1].start_line))
    return [item for _, item in scored_items]


def _sort_by_recency(
    items: list[SearchItem], cfg: SearchConfig, query_text: str, all_files: set[Path]
) -> list[SearchItem]:
    """Sort by file modification time (most recent first)."""
    scored_items = []
    for item in items:
        try:
            mtime = item.file.stat().st_mtime
            recency_score = mtime / 1000000  # Normalize timestamp
        except (OSError, AttributeError):
            recency_score = 0.0

        relevance_score = score_item(item, cfg, query_text, all_files)
        # Combine recency and relevance
        combined_score = recency_score * 0.3 + relevance_score * 0.7
        scored_items.append((combined_score, item))

    scored_items.sort(key=lambda x: (-x[0], x[1].file.as_posix(), x[1].start_line))
    return [item for _, item in scored_items]


def _sort_by_popularity(
    items: list[SearchItem], cfg: SearchConfig, query_text: str, all_files: set[Path]
) -> list[SearchItem]:
    """Sort by file popularity and importance."""
    scored_items = []
    for item in items:
        popularity_score = (
            _calculate_popularity_score(item.file, all_files) * 5.0
        )  # Amplify popularity
        relevance_score = score_item(item, cfg, query_text, all_files)
        # Combine popularity and relevance
        combined_score = popularity_score * 0.6 + relevance_score * 0.4
        scored_items.append((combined_score, item))

    scored_items.sort(key=lambda x: (-x[0], x[1].file.as_posix(), x[1].start_line))
    return [item for _, item in scored_items]


def _sort_hybrid(
    items: list[SearchItem], cfg: SearchConfig, query_text: str, all_files: set[Path]
) -> list[SearchItem]:
    """Hybrid sorting that combines multiple factors intelligently."""
    scored_items = []

    # Calculate various scores for each item
    for item in items:
        relevance_score = score_item(item, cfg, query_text, all_files)
        frequency_score = len(item.match_spans)

        try:
            mtime = item.file.stat().st_mtime
            recency_score = mtime / 1000000
        except (OSError, AttributeError):
            recency_score = 0.0

        popularity_score = _calculate_popularity_score(item.file, all_files)

        # Adaptive weighting based on query characteristics
        if len(query_text.split()) > 3:
            # Complex queries: prioritize relevance and semantic matching
            combined_score = (
                relevance_score * 0.6
                + frequency_score * 0.2
                + popularity_score * 0.15
                + recency_score * 0.05
            )
        elif query_text.isupper() or query_text.islower():
            # Simple queries: balance frequency and popularity
            combined_score = (
                relevance_score * 0.4
                + frequency_score * 0.3
                + popularity_score * 0.2
                + recency_score * 0.1
            )
        else:
            # Mixed case or special queries: prioritize exact matches
            combined_score = (
                relevance_score * 0.7
                + frequency_score * 0.15
                + popularity_score * 0.1
                + recency_score * 0.05
            )

        scored_items.append((combined_score, item))

    # Multi-level sorting: score, file importance, line number
    scored_items.sort(
        key=lambda x: (
            -x[0],  # Score (descending)
            -_calculate_popularity_score(x[1].file, all_files),  # File importance (descending)
            x[1].file.as_posix(),  # File path (ascending)
            x[1].start_line,  # Line number (ascending)
        )
    )

    return [item for _, item in scored_items]


def cluster_results_by_similarity(
    items: list[SearchItem], similarity_threshold: float = 0.8
) -> list[list[SearchItem]]:
    """
    Cluster search results by content similarity to group related matches.
    Returns list of clusters, each containing similar items.
    """
    if not items:
        return []

    clusters: list[list[SearchItem]] = []
    used_items: set[int] = set()

    for i, item in enumerate(items):
        if i in used_items:
            continue

        # Start new cluster with current item
        cluster = [item]
        used_items.add(i)

        # Find similar items
        for j, other_item in enumerate(items[i + 1 :], i + 1):
            if j in used_items:
                continue

            # Calculate similarity between items
            similarity = _calculate_content_similarity(item, other_item)

            if similarity >= similarity_threshold:
                cluster.append(other_item)
                used_items.add(j)

        clusters.append(cluster)

    return clusters


def _calculate_content_similarity(item1: SearchItem, item2: SearchItem) -> float:
    """Calculate content similarity between two search items."""
    if not item1.lines or not item2.lines:
        return 0.0

    content1 = " ".join(item1.lines).lower()
    content2 = " ".join(item2.lines).lower()

    # Simple word-based similarity
    words1 = set(content1.split())
    words2 = set(content2.split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def group_results_by_file(items: list[SearchItem]) -> dict[Path, list[SearchItem]]:
    """Group search results by file for better organization."""
    grouped: dict[Path, list[SearchItem]] = {}
    for item in items:
        if item.file not in grouped:
            grouped[item.file] = []
        grouped[item.file].append(item)

    # Sort items within each file by line number
    for file_items in grouped.values():
        file_items.sort(key=lambda x: x.start_line)

    return grouped


def deduplicate_overlapping_results(
    items: list[SearchItem], overlap_threshold: int = 5
) -> list[SearchItem]:
    """Remove overlapping search results to avoid redundancy."""
    if not items:
        return items

    # Group by file first
    grouped = group_results_by_file(items)
    deduplicated = []

    for file_path, file_items in grouped.items():
        # Sort by score (assuming items are already sorted)
        current_items: list[SearchItem] = []

        for item in file_items:
            # Check for overlap with existing items
            has_overlap = False
            for existing in current_items:
                # Check if ranges overlap significantly
                overlap_start = max(item.start_line, existing.start_line)
                overlap_end = min(item.end_line, existing.end_line)
                overlap_size = max(0, overlap_end - overlap_start + 1)

                item_size = item.end_line - item.start_line + 1
                existing_size = existing.end_line - existing.start_line + 1

                # If overlap is more than threshold lines or 50% of either item
                if (
                    overlap_size > overlap_threshold
                    or overlap_size > item_size * 0.5
                    or overlap_size > existing_size * 0.5
                ):
                    has_overlap = True
                    break

            if not has_overlap:
                current_items.append(item)

        deduplicated.extend(current_items)

    return deduplicated
