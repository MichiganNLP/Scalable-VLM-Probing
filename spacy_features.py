# This module can be a project available on GitHub and [spaCy Universe](https://spacy.io/universe).
# People could use it from `sent._.[attr]`.
from __future__ import annotations

from typing import Literal

import spacy.tokens


def create_model(model_name: str = "en_core_web_trf", prefer_gpu: bool = True) -> spacy.language.Language:
    if prefer_gpu:
        spacy.prefer_gpu()
    return spacy.load(model_name)


def get_first_sentence(doc: spacy.tokens.Doc) -> spacy.tokens.Span:
    return next(iter(doc.sents))


def get_sentence_count(doc: spacy.tokens.Doc) -> int:
    return sum(1 for _ in doc.sents)


def get_tense(sent: spacy.tokens.Span) -> Literal["Past", "Pres", "Fut"] | None:
    """Computes the grammatical tense of an English sentence.

    Examples
    ---
    >>> spacy_model = create_model()
    >>> get_tense(get_first_sentence(spacy_model("The man runs in the forest.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("The dogs will walk.")))
    'Fut'
    >>> get_tense(get_first_sentence(spacy_model("She'll teach the class.")))
    'Fut'
    >>> get_tense(get_first_sentence(spacy_model("I'll always love you.")))
    'Fut'
    >>> get_tense(get_first_sentence(spacy_model("I left already.")))
    'Past'
    >>> get_tense(get_first_sentence(spacy_model("A cat was hungry again.")))
    'Past'
    >>> get_tense(get_first_sentence(spacy_model("They are going to jump the fence.")))
    'Fut'
    >>> get_tense(get_first_sentence(spacy_model("They are gonna jump the fence.")))
    'Fut'
    >>> get_tense(get_first_sentence(spacy_model("They are going to the cinema.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("They have gone to the cinema.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("They've gone to the cinema.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("They have been going to the cinema.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("They had gone to the cinema.")))
    'Past'
    >>> get_tense(get_first_sentence(spacy_model("They'd gone to the cinema.")))
    'Past'
    >>> get_tense(get_first_sentence(spacy_model("They had been going to the cinema.")))
    'Past'
    >>> get_tense(get_first_sentence(spacy_model("They will have gone to the cinema.")))
    'Fut'
    >>> get_tense(get_first_sentence(spacy_model("They will have been going to the cinema.")))
    'Fut'
    """
    root = sent.root
    if ((root.lower_ in {"going", "gon", "gon'"} and any(t.tag_ == "VB" and t.dep_ == "xcomp" for t in root.rights))
            or any(t.lower_ in {"'ll", "will"} and t.tag_ == "MD" for t in root.lefts)):
        return "Fut"
    elif root_morphological_tenses := root.morph.get("Tense"):
        if root.tag_ == "VBN" or (root.tag_ == "VBG" and any(t.lower_ == "been" for t in root.lefts)):
            if any(t.lower_ in {"have", "'ve"} for t in root.lefts):
                return "Pres"
            elif any(t.lower_ in {"had", "'d"} for t in root.lefts):
                return "Past"
            else:
                return root_morphological_tenses[0]
        else:
            return root_morphological_tenses[0]
    elif any(t.tag_ == "MD" and t.lower_ in {"can"} for t in root.lefts):
        return "Pres"
    elif root.pos_ == "NOUN" and any(t.tag_ == "VBG" for t in root.lefts):
        return "Pres"
    else:
        return None


def is_continuous(sent: spacy.tokens.Span) -> bool:
    pass


def is_perfect(sent: spacy.tokens.Span) -> bool:
    pass


def is_person(sent: spacy.tokens.Span) -> Literal[1, 2, 3]:
    pass


def is_plural(sent: spacy.tokens.Span) -> bool:
    pass


def has_any_adjective(doc: spacy.tokens.Doc) -> bool:
    pass


def has_any_adverb(doc: spacy.tokens.Doc) -> bool:
    pass


def get_root_tag(sent: spacy.tokens.Span) -> str:
    return sent.root.tag_
