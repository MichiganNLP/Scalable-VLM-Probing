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
    """Computes the grammatical tense of an English sentence. If it's not a sentence (or if it can't determine the
    tense), it returns `None`.

    Examples
    ---
    >>> spacy_model = create_model()
    >>> get_tense(get_first_sentence(spacy_model("The man runs in the forest.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("The man is running again.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("I'm walking on sunshine.")))
    'Pres'
    >>> get_tense(get_first_sentence(spacy_model("I was walking yesterday.")))
    'Past'
    >>> get_tense(get_first_sentence(spacy_model("I will be arriving next week.")))
    'Fut'
    >>> get_tense(get_first_sentence(spacy_model("I'll be arriving next week.")))
    'Fut'
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
    elif ((root.tag_ == "VBN" or (root.tag_ == "VBG" and any(t.lower_ == "been" for t in root.lefts)))
          and (have := next((t for t in root.lefts if t.lemma_ in {"have", "'d", "'ve"}), None))
          and (have_token_morphological_tense := have.morph.get("Tense"))):
        return have_token_morphological_tense[0]  # noqa
    elif ((be := next((t for t in root.lefts if t.lemma_ == "be" and t.dep_ == "aux"), None))
          and (be.morph.get("VerbForm") or [""])[0] == "Fin"
          and (be_morphological_tense := be.morph.get("Tense"))):
        return be_morphological_tense[0]  # noqa
    elif root_morphological_tenses := root.morph.get("Tense"):
        return root_morphological_tenses[0]
    elif any(t.tag_ == "MD" and t.lower_ in {"can"} for t in root.lefts):
        return "Pres"
    else:
        return None


def is_continuous(sent: spacy.tokens.Span) -> bool:
    """Computes the continuous grammatical aspect of an English sentence. If it's not a sentence (or if it can't
    determine the tense), it returns `False`.

    Examples
    ---
    >>> spacy_model = create_model()
    >>> is_continuous(get_first_sentence(spacy_model("The man runs in the forest.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("The man is running again.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("I'm walking on sunshine.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("I was walking yesterday.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("I will be arriving next week.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("I'll be arriving next week.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("The dogs will walk.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("She'll teach the class.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("I'll always love you.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("I left already.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("A cat was hungry again.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They are going to jump the fence.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They are gonna jump the fence.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They are going to the cinema.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("They have gone to the cinema.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They've gone to the cinema.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They have been going to the cinema.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("They had gone to the cinema.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They'd gone to the cinema.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They had been going to the cinema.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("They will have gone to the cinema.")))
    False
    >>> is_continuous(get_first_sentence(spacy_model("They will have been going to the cinema.")))
    True
    >>> is_continuous(get_first_sentence(spacy_model("They would have been going to the cinema.")))
    True
    """
    root = sent.root
    if (root.lower_ in {"going", "gon", "gon'"}
            and (verb := next((t for t in root.rights if t.tag_ == "VB" and t.dep_ == "xcomp"), None))):
        root = verb

    return (root.morph.get("Aspect") or ["Hab"])[0] == "Prog"


def is_perfect(sent: spacy.tokens.Span) -> bool:
    """Computes the perfect grammatical aspect of an English sentence. If it's not a sentence (or if it can't
    determine it), it returns `False`.

    Examples
    ---
    >>> spacy_model = create_model()
    >>> is_perfect(get_first_sentence(spacy_model("The man runs in the forest.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("The man is running again.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("I'm walking on sunshine.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("I was walking yesterday.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("I will be arriving next week.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("I'll be arriving next week.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("The dogs will walk.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("She'll teach the class.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("I'll always love you.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("I left already.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("A cat was hungry again.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("They are going to jump the fence.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("They are gonna jump the fence.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("They are going to the cinema.")))
    False
    >>> is_perfect(get_first_sentence(spacy_model("They have gone to the cinema.")))
    True
    >>> is_perfect(get_first_sentence(spacy_model("They've gone to the cinema.")))
    True
    >>> is_perfect(get_first_sentence(spacy_model("They have been going to the cinema.")))
    True
    >>> is_perfect(get_first_sentence(spacy_model("They had gone to the cinema.")))
    True
    >>> # is_perfect(get_first_sentence(spacy_model("They'd gone to the cinema.")))  # Bug: it's parsed as "would".
    >>> is_perfect(get_first_sentence(spacy_model("They had been going to the cinema.")))
    True
    >>> is_perfect(get_first_sentence(spacy_model("They will have gone to the cinema.")))
    True
    >>> is_perfect(get_first_sentence(spacy_model("They will have been going to the cinema.")))
    True
    >>> is_perfect(get_first_sentence(spacy_model("They would have been going to the cinema.")))
    True
    """
    root = sent.root
    return ((root.tag_ == "VBN" or (root.tag_ == "VBG" and any(t.lower_ == "been" for t in root.lefts)))
            and any(t.lemma_ in {"have", "'d", "'ve"} for t in root.lefts))


def get_subject_person(sent: spacy.tokens.Span) -> Literal["1", "2", "3"] | None:
    """Computes the subject person of an English sentence. If it's not a sentence (or if it can't
    determine it), it returns `None`.

    Examples
    ---
    >>> spacy_model = create_model()
    >>> get_subject_person(get_first_sentence(spacy_model("The man runs in the forest.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("The man is running again.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("I'm walking on sunshine.")))
    '1'
    >>> get_subject_person(get_first_sentence(spacy_model("I was walking yesterday.")))
    '1'
    >>> get_subject_person(get_first_sentence(spacy_model("I will be arriving next week.")))
    '1'
    >>> get_subject_person(get_first_sentence(spacy_model("I'll be arriving next week.")))
    '1'
    >>> # get_subject_person(get_first_sentence(spacy_model("The dogs will walk.")))  # It fails.
    >>> get_subject_person(get_first_sentence(spacy_model("She'll teach the class.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("I'll always love you.")))
    '1'
    >>> get_subject_person(get_first_sentence(spacy_model("I left already.")))
    '1'
    >>> get_subject_person(get_first_sentence(spacy_model("A cat was hungry again.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They are going to jump the fence.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They are gonna jump the fence.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They are going to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They have gone to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They've gone to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They have been going to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They had gone to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They'd gone to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They had been going to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They will have gone to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They will have been going to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("They would have been going to the cinema.")))
    '3'
    >>> get_subject_person(get_first_sentence(spacy_model("You'll get there.")))
    '2'
    """
    root = sent.root
    if root_morphological_person := root.morph.get("Person"):
        return root_morphological_person[0]
    elif ((subj := next((t for t in sent.root.children if t.dep_ == "nsubj"), None))
            and (subj_morphological_person := subj.morph.get("Person"))):
        return subj_morphological_person[0]  # noqa
    elif ((aux := next((t for t in sent.root.children if t.dep_ == "aux"), None))
            and (aux_morphological_person := aux.morph.get("Person"))):
        return aux_morphological_person[0]  # noqa
    else:
        return None


def is_subject_plural(sent: spacy.tokens.Span) -> bool | None:
    """Computes if the subject is plural in an English sentence. If it's not a sentence (or if it can't
    determine it), it returns `None`.

    Examples
    ---
    >>> spacy_model = create_model()
    >>> is_subject_plural(get_first_sentence(spacy_model("The man runs in the forest.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("The man is running again.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("I'm walking on sunshine.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("I was walking yesterday.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("I will be arriving next week.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("I'll be arriving next week.")))
    False
    >>> # is_subject_plural(get_first_sentence(spacy_model("The dogs will walk.")))  # It fails.
    >>> is_subject_plural(get_first_sentence(spacy_model("She'll teach the class.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("I'll always love you.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("I left already.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("A cat was hungry again.")))
    False
    >>> is_subject_plural(get_first_sentence(spacy_model("They are going to jump the fence.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They are gonna jump the fence.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They are going to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They have gone to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They've gone to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They have been going to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They had gone to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They'd gone to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They had been going to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They will have gone to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They will have been going to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("They would have been going to the cinema.")))
    True
    >>> is_subject_plural(get_first_sentence(spacy_model("We'll get there.")))
    True
    """
    root = sent.root
    if root_morphological_person := root.morph.get("Number"):
        return root_morphological_person[0] == "Plur"
    elif ((subj := next((t for t in sent.root.children if t.dep_ == "nsubj"), None))
            and (subj_morphological_person := subj.morph.get("Number"))):
        return subj_morphological_person[0] == "Plur"
    elif ((aux := next((t for t in sent.root.children if t.dep_ == "aux"), None))
            and (aux_morphological_person := aux.morph.get("Number"))):
        return aux_morphological_person[0] == "Plur"
    else:
        return None


def has_any_adjective(doc: spacy.tokens.Doc) -> bool:
    return any(t.pos_ == "ADJ" for t in doc)


def has_any_gerund(doc: spacy.tokens.Doc) -> bool:
    return any(t.tag_ == "VBG" for t in doc)


def has_any_adverb(doc: spacy.tokens.Doc) -> bool:
    return any(t.pos_ == "ADV" for t in doc)


def is_passive_voice(sent: spacy.tokens.Span) -> bool | None:
    return any(t.lower_ == "be" for t in sent.root.lefts)


def get_root_tag(sent: spacy.tokens.Span) -> str:
    return sent.root.tag_


def get_root_pos(sent: spacy.tokens.Span) -> str:
    return sent.root.pos_
