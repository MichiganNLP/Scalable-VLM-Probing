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


def get_time(span: spacy.tokens.Span) -> Literal["past", "present", "future"]:
    pass


def is_continuous(span: spacy.tokens.Span) -> bool:
    pass


def is_perfect(span: spacy.tokens.Span) -> bool:
    pass


def is_person(span: spacy.tokens.Span) -> Literal[1, 2, 3]:
    pass


def is_plural(span: spacy.tokens.Span) -> bool:
    pass


def has_any_adjective(doc: spacy.tokens.Doc) -> bool:
    pass


def get_root_tag(span: spacy.tokens.Span) -> str:
    pass
