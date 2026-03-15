"""Text normalization, tokenization, noise detection, and scoring utilities for web tools."""

from __future__ import annotations

import re
import unicodedata

_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "au",
        "aujourdhui",
        "aujourd",
        "ce",
        "ces",
        "cette",
        "d",
        "de",
        "des",
        "du",
        "en",
        "est",
        "et",
        "for",
        "il",
        "is",
        "hui",
        "la",
        "le",
        "les",
        "l",
        "of",
        "on",
        "ou",
        "pour",
        "q",
        "qu",
        "que",
        "quelle",
        "quelles",
        "quel",
        "quels",
        "s",
        "the",
        "to",
        "today",
        "un",
        "une",
        "what",
    }
)
_COPULAR_TOKENS = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "est",
        "etaient",
        "etait",
        "furent",
        "is",
        "sera",
        "seront",
        "sont",
        "was",
        "were",
    }
)
_MATCH_BOILERPLATE_PREFIXES = (
    "all rights reserved",
    "cookie ",
    "copyright ",
    "menu ",
    "sign in",
    "skip to",
)
_NOISE_SIGNAL_PHRASES = frozenset(
    {
        "accept cookies",
        "accepter les cookies",
        "acceder aux notifications",
        "account settings",
        "all rights reserved",
        "already a subscriber",
        "article reserve",
        "conditions generales",
        "consentement",
        "contenu reserve",
        "cookie policy",
        "cookie preferences",
        "creer un compte",
        "data protection",
        "deja abonne",
        "deja inscrit",
        "donnees personnelles",
        "en poursuivant",
        "en savoir plus et gerer",
        "gerer les cookies",
        "gerer mes preferences",
        "inscription gratuite",
        "log in to",
        "manage preferences",
        "manage your subscription",
        "mon compte",
        "newsletter signup",
        "nos partenaires",
        "notre politique",
        "nous utilisons des cookies",
        "offre numerique",
        "offre speciale",
        "parametres de confidentialite",
        "partenaires data",
        "politique de confidentialite",
        "privacy policy",
        "privacy settings",
        "profitez de",
        "se connecter",
        "sign in to",
        "sign up for",
        "subscribe to",
        "subscription required",
        "terms of service",
        "tous droits reserves",
        "use of cookies",
        "utilisation de cookies",
        "votre abonnement",
        "votre consentement",
        "your privacy",
        "your subscription",
    }
)
_GENERIC_LINK_TEXTS = frozenset(
    {
        "continue reading",
        "home",
        "learn more",
        "menu",
        "more",
        "next",
        "older posts",
        "previous",
        "read more",
        "see more",
        "view more",
    }
)


def _normalize_text(text: str) -> str:
    normalized_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue

        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


def _fold_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def _text_tokens(text: str) -> tuple[str, ...]:
    return tuple(_fold_for_match(text).split())


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _text_tokens(text)
        if token not in _QUERY_STOPWORDS and (len(token) > 2 or token.isdigit())
    )


def _keyword_overlap_score(text: str, keywords: tuple[str, ...]) -> int:
    if not keywords:
        return 0
    text_tokens = set(text.split())
    return sum(1 for keyword in keywords if keyword in text_tokens)


def _clip_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized

    clipped = normalized[: limit - 3].rstrip(" ,;:.")
    return f"{clipped}..."


def _clip_summary_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized

    for separator in (". ", "; ", ", "):
        boundary = normalized.rfind(separator, 0, limit)
        if boundary >= int(limit * 0.65):
            clipped = normalized[: boundary + (1 if separator == ". " else 0)].rstrip()
            if clipped and clipped[-1] not in ".!?":
                clipped += "."
            return clipped

    return _clip_text(normalized, limit=limit)


def _truncate_sentences(text: str, *, max_sentences: int, max_chars: int) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    if not sentences:
        return _clip_text(text, limit=max_chars)

    selected = " ".join(sentences[:max_sentences]).strip()
    if selected:
        return _clip_text(selected, limit=max_chars)
    return _clip_text(text, limit=max_chars)


def _split_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    )


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().rstrip(" ,;:.!?")


def _join_summary_parts(parts: tuple[str, ...]) -> str:
    cleaned_parts = [part.strip() for part in parts if part.strip()]
    if not cleaned_parts:
        return ""
    joined = ". ".join(_strip_terminal_punctuation(part) for part in cleaned_parts)
    if joined and joined[-1] not in ".!?":
        joined += "."
    return joined


def _build_signature_tokens(text: str) -> tuple[str, ...]:
    return tuple(_text_tokens(text)[:24])


def _evidence_similarity(
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union_size = len(left_set | right_set)
    if union_size == 0:
        return 0.0
    return len(left_set & right_set) / union_size


def _overlap_coefficient(
    left_tokens,
    right_tokens,
) -> float:
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def _substring_similarity(left_text: str, right_text: str) -> float:
    left_folded = _fold_for_match(left_text)
    right_folded = _fold_for_match(right_text)
    if not left_folded or not right_folded:
        return 0.0
    if left_folded in right_folded or right_folded in left_folded:
        return 0.78
    return 0.0


def _extract_subject_tokens(text: str) -> tuple[str, ...]:
    tokens = _text_tokens(text)
    for index, token in enumerate(tokens):
        if token not in _COPULAR_TOKENS:
            continue
        subject = tuple(
            candidate
            for candidate in tokens[:index]
            if candidate not in _QUERY_STOPWORDS
        )
        if len(subject) == 1 and subject[0] in {"elle", "he", "il", "it", "she", "they"}:
            return ()
        if 2 <= len(subject) <= 6:
            return subject
        return ()
    return ()


def _iter_passages(text: str) -> tuple[str, ...]:
    passages = tuple(
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    )
    if passages:
        return passages
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _is_informative_passage(text: str, *, title: str) -> bool:
    words = text.split()
    if len(words) < 8 or _looks_like_title_echo(text, title):
        return False

    lowered = _fold_for_match(text)
    if lowered.startswith(_MATCH_BOILERPLATE_PREFIXES):
        return False
    if _passage_has_noise_signals(text):
        return False
    if _looks_site_descriptive(text):
        return False
    if _looks_promotional(text):
        return False
    return True


def _looks_like_title_echo(text: str, title: str) -> bool:
    title_tokens = _text_tokens(title)
    text_tok = _text_tokens(text)
    if not title_tokens or not text_tok:
        return False

    short_text = len(text_tok) <= max(len(title_tokens) + 6, 18)
    if not short_text:
        return False

    title_signature = " ".join(title_tokens)
    text_signature = " ".join(text_tok[: len(title_tokens) + 6])
    if text_signature.startswith(title_signature):
        return True

    title_token_set = set(title_tokens)
    overlap = sum(1 for token in text_tok if token in title_token_set)
    return overlap >= max(3, len(title_tokens) - 1)


def _looks_boilerplate_text(text: str) -> bool:
    folded_text = _fold_for_match(text)
    return folded_text.startswith(_MATCH_BOILERPLATE_PREFIXES)


def _looks_site_descriptive(text: str) -> bool:
    """Return True if the passage merely describes the site or section rather than conveying facts."""
    folded = _fold_for_match(text)
    _SITE_DESCRIPTIVE_CUES = (
        "retrouvez toute l",
        "retrouvez toutes les",
        "suivez l actualite",
        "toute l actualite",
        "toutes les actualites",
        "toute l information",
        "decouvrez les dernieres",
        "decouvrez toute l",
        "decouvrez toutes les",
        "bienvenue sur",
        "welcome to our",
        "visit our",
        "follow us on",
        "suivez nous sur",
        "retrouvez nous sur",
        "abonnez vous",
        "stay up to date",
        "stay informed",
        "suivez en direct",
        "suivez en temps reel",
        "regardez en direct",
        "a suivre en direct",
        "en direct sur",
        "en continu sur",
        "your source for",
        "your daily source",
        "votre source d",
        "tout savoir sur",
        "l essentiel de l actualite",
        "toute l info",
    )
    return any(cue in folded for cue in _SITE_DESCRIPTIVE_CUES)


def _looks_promotional(text: str) -> bool:
    """Return True if the passage is promotional or service-oriented rather than factual."""
    folded = _fold_for_match(text)
    _PROMOTIONAL_CUES = (
        "abonnez vous",
        "commencez votre",
        "creez votre compte",
        "decouvrez nos offres",
        "essai gratuit",
        "essayez gratuitement",
        "free trial",
        "get started",
        "inscrivez vous",
        "join now",
        "offre d abonnement",
        "offre speciale",
        "profitez de notre",
        "sign up today",
        "start your",
        "subscribe now",
        "try for free",
        "upgrade your",
    )
    return any(cue in folded for cue in _PROMOTIONAL_CUES)


def _passage_has_noise_signals(text: str) -> bool:
    """Return True if the passage contains UI, consent, or navigation noise."""
    folded = _fold_for_match(text)
    return any(phrase in folded for phrase in _NOISE_SIGNAL_PHRASES)


def _passage_noise_score(text: str) -> float:
    """Return a penalty score (0.0 = clean, higher = noisier) for UI/consent noise."""
    folded = _fold_for_match(text)
    hits = sum(1 for phrase in _NOISE_SIGNAL_PHRASES if phrase in folded)
    if hits == 0:
        return 0.0
    tokens = folded.split()
    if not tokens:
        return 0.0
    # Scale penalty: more hits or shorter text = higher noise ratio
    return min(hits * 3.0, 9.0)


def _score_evidence_text(
    text: str,
    *,
    title: str,
    query,
) -> float:
    if not _is_informative_passage(text, title=title):
        return -1.0

    tokens = _text_tokens(text)
    folded_text = _fold_for_match(text)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    score = min(len(tokens) / 12.0, 4.0)
    score += _keyword_overlap_score(folded_text, query.keyword_tokens) * 2.5
    score += unique_ratio * 2.0
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", text):
        score += 0.5
    if _looks_like_title_echo(text, title):
        score -= 2.5
    if _looks_boilerplate_text(text):
        score -= 3.0
    score -= _passage_noise_score(text)
    if _looks_site_descriptive(text):
        score -= 2.0
    if _looks_promotional(text):
        score -= 3.0
    # Penalize very short passages with no keyword overlap (vague/low-signal)
    keyword_overlap = _keyword_overlap_score(folded_text, query.keyword_tokens)
    if len(tokens) < 12 and keyword_overlap == 0:
        score -= 2.0
    return score


def _link_text_looks_generic(text: str) -> bool:
    normalized_text = _fold_for_match(text)
    return normalized_text in _GENERIC_LINK_TEXTS
