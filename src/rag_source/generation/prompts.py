"""Construction du prompt, et vérification des citations.

Trois principes, tous absents de l'implémentation d'origine :

1. **Les passages sont numérotés et identifiés.** Son prompt collait les extraits
   bout à bout, séparés par des tirets, sans dire d'où ils venaient : le modèle ne
   *pouvait pas* citer ses sources, même en le voulant.
2. **Le modèle a le droit de ne pas savoir.** Sans cette permission explicite, un
   LLM répond toujours quelque chose.
3. **Les citations sont vérifiées après coup.** Une consigne de prompt n'est pas
   une garantie : on relit la réponse, on extrait les ``[n]``, et on signale ceux
   qui ne correspondent à aucun passage fourni.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_source.clients.llm import Message
from rag_source.retrieval.search import Passage

CITATION = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = """\
Tu réponds à des questions à partir de passages numérotés qui te sont fournis.

Règles impératives :
1. Utilise uniquement les passages fournis. Aucune connaissance extérieure, même si \
la réponse te paraît évidente.
2. CHAQUE phrase de ta réponse doit se terminer par le numéro du passage qui la \
justifie, entre crochets : [1], [2]… Une phrase sans numéro est une erreur.
3. Si les passages ne répondent pas à la question, écris uniquement : « Je ne trouve \
pas cette information dans les documents. »
4. Réponds dans la langue de la question, quelle que soit celle des passages.
5. Sois bref : deux ou trois phrases suffisent. Ne recopie pas les passages.

Exemple de réponse correcte :
La qualification est délivrée pour deux ans au maximum [1]. Son renouvellement \
suppose un rapport d'évaluation transmis trois mois avant l'échéance [2]."""

# Formulation choisie par la mesure. Une consigne simple (« cite tes sources ») ne
# suffit pas avec un modèle de 3 milliards de paramètres : sur huit questions du jeu
# de référence, elle ne produisait de citation que dans 5 cas sur 8. La règle
# explicite par phrase, accompagnée d'un exemple, porte ce taux à 8 sur 8, pour un
# coût de génération identique.

NO_CONTEXT_ANSWER = "Je ne trouve pas d'information à ce sujet dans les documents indexés."

CONDENSE_PROMPT = """\
Réécris la question de suivi en une question autonome, compréhensible sans \
l'historique, dans sa langue d'origine. Ne réponds pas à la question, ne l'explique \
pas : renvoie uniquement la question réécrite."""


@dataclass(frozen=True, slots=True)
class Citation:
    """Une source citée dans la réponse."""

    number: int
    passage: Passage

    @property
    def source(self) -> str:
        return self.passage.source

    @property
    def location(self) -> str:
        return self.passage.location


def build_messages(
    question: str,
    passages: list[Passage],
    history: list[tuple[str, str]] | None = None,
) -> list[Message]:
    """Messages envoyés au modèle : consignes, historique éventuel, passages, question."""
    messages = [Message(role="system", content=SYSTEM_PROMPT)]
    for previous_question, previous_answer in history or []:
        messages.append(Message(role="user", content=previous_question))
        messages.append(Message(role="assistant", content=previous_answer))
    messages.append(Message(role="user", content=_context_block(question, passages)))
    return messages


def _context_block(question: str, passages: list[Passage]) -> str:
    blocks = [
        f"[{number}] ({passage.location})\n{passage.text}"
        for number, passage in enumerate(passages, start=1)
    ]
    context = "\n\n".join(blocks)
    return f"Passages :\n\n{context}\n\n---\n\nQuestion : {question}"


def build_condense_messages(question: str, history: list[tuple[str, str]]) -> list[Message]:
    """Messages pour reformuler une question de suivi en question autonome.

    « Et pour le modèle suivant ? » ne veut rien dire pour une recherche vectorielle :
    la reformulation rend la question interrogeable seule.
    """
    exchanges = "\n".join(
        f"Question : {previous_question}\nRéponse : {previous_answer}"
        for previous_question, previous_answer in history
    )
    return [
        Message(role="system", content=CONDENSE_PROMPT),
        Message(role="user", content=f"{exchanges}\n\nQuestion de suivi : {question}"),
    ]


def extract_citations(answer: str, passages: list[Passage]) -> tuple[list[Citation], list[int]]:
    """Citations valides et numéros invalides trouvés dans la réponse.

    Un numéro qui ne désigne aucun passage fourni est une citation inventée : c'est
    le signe le plus net qu'une réponse s'écarte de ses sources, et il serait
    invisible sans cette vérification.
    """
    citations: dict[int, Citation] = {}
    invalid: list[int] = []
    for match in CITATION.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= len(passages):
            citations.setdefault(number, Citation(number=number, passage=passages[number - 1]))
        elif number not in invalid:
            invalid.append(number)
    return [citations[key] for key in sorted(citations)], invalid
