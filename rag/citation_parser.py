"""
Citation Parser — parses LLM responses into structured cited answers.

Parses the LLM response for:
  - Inline [N] citation markers
  - ИСТОЧНИКИ: block with source references
  - ЦИТАТЫ: block with verbatim quotes
  - НЕ ЗНАЮ: prefix for uncertain answers

Verifies quotes are substrings of retrieved chunks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from rag import Chunk, CitedAnswer, Quote, SearchResult, SourceRef


def normalize_text(text: str) -> str:
    """Normalize text for robust substring matching."""
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip common punctuation from start/end
    text = text.strip(' .,;:!?«»"\'"')
    return text


def is_quote_in_texts(quote_text: str, texts: List[str]) -> bool:
    """Check if a quote appears as a substring in any of the provided texts."""
    q_norm = normalize_text(quote_text)
    if len(q_norm) < 10:
        # Too short to reliably verify
        return False
    for t in texts:
        if q_norm in normalize_text(t):
            return True
    return False


def parse_sources_block(text: str) -> List[SourceRef]:
    """Parse the ИСТОЧНИКИ: section of the response."""
    sources: List[SourceRef] = []

    for line in text.split('\n'):
        line = line.strip()
        if not line.startswith('['):
            continue
        
        # Try to extract number
        num_match = re.match(r'\[(\d+)\]\s+(.*)', line)
        if not num_match:
            continue
        
        num = int(num_match.group(1))
        rest = num_match.group(2).strip()
        
        # Try to split by em-dash, hyphen, or pipe
        parts = re.split(r'\s*[—|]\s*|\s+-\s+', rest, maxsplit=1)
        if len(parts) >= 2:
            source = parts[0].strip()
            section_and_id = parts[1].strip()
            
            # Extract chunk_id if in parentheses
            chunk_match = re.search(r'\(([\w\d_-]+)\)', section_and_id)
            
            # Extract score if present
            score_match = re.search(r'score[:\s]*([\d.]+)', section_and_id)
            
            section = re.sub(r'\s*\(.*\)\s*', '', section_and_id)
            section = re.sub(r'\s*\[.*\]\s*', '', section).strip()
            chunk_id = chunk_match.group(1) if chunk_match else ""
            score = float(score_match.group(1)) if score_match else 0.0
            
            sources.append(SourceRef(
                index=num,
                source=source,
                section=section,
                chunk_id=chunk_id,
                score=score,
            ))
        elif len(parts) == 1:
            # Minimal: just source
            source = parts[0].strip()
            # Try to extract from just the source string
            sources.append(SourceRef(
                index=num,
                source=source,
                section="",
                chunk_id="",
                score=0.0,
            ))

    return sources


def parse_quotes_block(text: str) -> List[Quote]:
    """Parse the ЦИТАТЫ: section of the response."""
    quotes: List[Quote] = []
    
    # Match lines like: [1] «quote text»
    # Also handles [1] "quote text" and [1] 'quote text'
    quote_pattern = re.compile(
        r'\[(\d+)\]\s+'
        r'[«"\'](.+?)[»"\']'
        r'(?:\s*(✅|verified|✓))?'
    , re.DOTALL)

    for line in text.split('\n'):
        line = line.strip()
        match = quote_pattern.search(line)
        if match:
            num = int(match.group(1))
            quote_text = match.group(2).strip()
            verified = bool(match.group(3))
            
            quotes.append(Quote(
                index=num,
                text=quote_text,
                verified=verified,
            ))

    return quotes


def parse_cited_response(
    raw: str,
    results: List[SearchResult],
    verbose: bool = False,
) -> CitedAnswer:
    """
    Parse an LLM response into a structured CitedAnswer.

    Expects the response to have:
    - An answer section (before markers)
    - ИСТОЧНИКИ: block
    - ЦИТАТЫ: block

    Args:
        raw: The raw LLM response text
        results: List of SearchResult objects used to build context
        verbose: If True, print parsing debug info

    Returns:
        CitedAnswer with parsed answer, sources, and quotes.
    """
    answer_text = raw
    sources: List[SourceRef] = []
    quotes_list: List[Quote] = []
    is_uncertain = False

    # Check for "I don't know" pattern
    unknown_patterns = [
        re.compile(r'НЕ ЗНАЮ[:\s]', re.IGNORECASE),
        re.compile(r'не могу ответить', re.IGNORECASE),
        re.compile(r'недостаточно информации', re.IGNORECASE),
    ]
    for pat in unknown_patterns:
        if pat.search(raw):
            is_uncertain = True
            break

    # Remove ქ tags before parsing
    cleaned = re.sub(r'ქ.*?ქ', '', raw, flags=re.DOTALL)
    cleaned = re.sub(r'ქ.*?ქ', '', cleaned, flags=re.DOTALL)

    # Split by section markers
    sections = re.split(
        r'\n\s*(?:ИСТОЧНИКИ|ИСТОЧНИКИ:|SOURCES|SOURCES:)\s*\n',
        cleaned,
        flags=re.IGNORECASE,
    )

    if len(sections) >= 2:
        answer_text = sections[0].strip()
        rest = sections[1]

        # Further split for quotes
        quotes_split = re.split(
            r'\n\s*(?:ЦИТАТЫ|ЦИТАТЫ:|QUOTES|QUOTES:|ВЫДЕРЖКИ|Цитаты)\s*\n',
            rest,
            flags=re.IGNORECASE,
        )
        if len(quotes_split) >= 2:
            sources_block = quotes_split[0]
            quotes_block = quotes_split[1]
            
            sources = parse_sources_block(sources_block)
            quotes_list = parse_quotes_block(quotes_block)
        else:
            sources = parse_sources_block(rest)

    # If no ИСТОЧНИКИ: section found at all, try to parse inline [N] markers
    if not sources and len(sections) < 2:
        # Try alternative pattern: look for lines starting with [N] anywhere
        sources = parse_sources_block(cleaned)
        # Also look for quotes
        quotes_list = parse_quotes_block(cleaned)
        # If we found structured data, extract it from answer
        if sources or quotes_list:
            # Remove source/quote lines from answer text
            answer_lines = []
            for line in cleaned.split('\n'):
                stripped = line.strip()
                if not re.match(r'^\[\d+\]', stripped):
                    answer_lines.append(line)
            answer_text = '\n'.join(answer_lines).strip()

    # Verify quotes against retrieved chunk texts
    chunk_texts = [r.chunk.text for r in results]
    verified_count = 0
    for q in quotes_list:
        q.verified = is_quote_in_texts(q.text, chunk_texts)
        if q.verified:
            verified_count += 1

    if verbose:
        print(f"  [parser] sources={len(sources)}, quotes={len(quotes_list)}, "
              f"verified={verified_count}/{len(quotes_list)}, uncertain={is_uncertain}")

    # Calculate confidence as max score from retrieved results
    confidence: Optional[float] = None
    if results:
        confidence = max(r.score for r in results)

    return CitedAnswer(
        answer=answer_text.strip(),
        sources=sources,
        quotes=quotes_list,
        is_uncertain=is_uncertain,
        raw_response=raw,
        confidence=confidence,
    )
