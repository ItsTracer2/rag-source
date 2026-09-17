"""Tests du découpage.

Le compteur de tokens utilisé ici compte les mots : déterministe, lisible, et
indépendant du modèle d'embedding. Les tailles des tests sont donc à lire en mots.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_source.domain import Chunk, DocumentFormat, LoadedDocument, Section, SectionKind
from rag_source.ingest.chunker import BREADCRUMB, ChunkingConfig, chunk_document, chunk_id
from rag_source.ingest.tokenizer import EstimatedTokenCounter, get_token_counter

SHA = "a" * 64
OTHER_SHA = "b" * 64


class WordCounter:
    """Compteur déterministe : un mot = un token."""

    exact = False

    def count(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()


def make_document(*sections: Section, title: str = "Manuel", sha: str = SHA) -> LoadedDocument:
    return LoadedDocument(
        source="manuel.pdf",
        sha256=sha,
        format=DocumentFormat.PDF,
        title=title,
        sections=sections,
    )


def words(count: int, word: str = "mot") -> str:
    return " ".join([word] * count)


class TestBreadcrumb:
    def test_prefix_carries_title_and_headings(self, counter: WordCounter) -> None:
        document = make_document(
            Section(
                text="Le plateau doit tourner librement.", heading_path=("Entretien", "Plateau")
            )
        )
        chunk = chunk_document(document, counter)[0]
        assert chunk.text.startswith(f"Manuel{BREADCRUMB}Entretien{BREADCRUMB}Plateau\n\n")
        assert chunk.text.endswith("Le plateau doit tourner librement.")

    def test_breadcrumb_is_repeated_on_every_chunk(self, counter: WordCounter) -> None:
        document = make_document(Section(text=words(500), heading_path=("Sécurité",)))
        chunks = chunk_document(
            document, counter, ChunkingConfig(target_tokens=100, max_tokens=150)
        )
        assert len(chunks) > 1
        assert all(c.text.startswith(f"Manuel{BREADCRUMB}Sécurité") for c in chunks)


class TestProse:
    def test_short_section_stays_whole(self, counter: WordCounter) -> None:
        document = make_document(Section(text="Une phrase courte."))
        assert len(chunk_document(document, counter)) == 1

    def test_respects_max_tokens(self, counter: WordCounter) -> None:
        config = ChunkingConfig(target_tokens=100, max_tokens=150)
        document = make_document(Section(text=words(1000)))
        chunks = chunk_document(document, counter, config)
        assert chunks and all(c.token_count <= config.max_tokens for c in chunks)

    def test_splits_on_paragraphs_then_sentences(self, counter: WordCounter) -> None:
        paragraph = " ".join(f"Voici la phrase numéro {i} de ce paragraphe." for i in range(6))
        document = make_document(Section(text=f"{paragraph}\n\n{paragraph}"))
        chunks = chunk_document(document, counter, ChunkingConfig(target_tokens=20, max_tokens=40))
        # Aucun chunk ne coupe au milieu d'une phrase : chacun finit par un point.
        bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
        assert all(body.rstrip().endswith(".") for body in bodies)

    def test_overlap_repeats_the_end_of_the_previous_chunk(self, counter: WordCounter) -> None:
        sentences = " ".join(f"Phrase numéro {i} du document." for i in range(40))
        document = make_document(Section(text=sentences))
        config = ChunkingConfig(target_tokens=30, max_tokens=60, overlap_ratio=0.3)
        chunks = chunk_document(document, counter, config)
        assert len(chunks) > 2
        first_body = chunks[0].text.split("\n\n", 1)[1]
        second_body = chunks[1].text.split("\n\n", 1)[1]
        assert second_body.split(".")[0] in first_body

    def test_no_overlap_when_disabled(self, counter: WordCounter) -> None:
        sentences = " ".join(f"Phrase numéro {i} du document." for i in range(40))
        document = make_document(Section(text=sentences))
        config = ChunkingConfig(target_tokens=30, max_tokens=60, overlap_ratio=0.0)
        chunks = chunk_document(document, counter, config)
        bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
        assert len(" ".join(bodies).split()) == len(sentences.split())

    def test_overlap_never_pushes_a_chunk_over_the_limit(self, counter: WordCounter) -> None:
        """Le recouvrement s'ajoute au contenu : il doit céder devant la limite."""
        units = "\n\n".join(f"Paragraphe {i} : " + words(25) + "." for i in range(20))
        document = make_document(Section(text=units))
        config = ChunkingConfig(target_tokens=30, max_tokens=32, overlap_ratio=0.4)
        chunks = chunk_document(document, counter, config)
        assert chunks and all(c.token_count <= config.max_tokens for c in chunks)

    def test_single_endless_sentence_is_hard_split(self, counter: WordCounter) -> None:
        """Texte sans ponctuation (ou langue sans espaces fines) : on coupe aux mots."""
        document = make_document(Section(text=words(300)))
        config = ChunkingConfig(target_tokens=50, max_tokens=60)
        chunks = chunk_document(document, counter, config)
        assert len(chunks) >= 5
        assert all(c.token_count <= config.max_tokens for c in chunks)


class TestAtomicSections:
    def test_record_stays_whole(self, counter: WordCounter) -> None:
        record = Section(
            text="Identifiant : R24\nIntitulé : Cloisonner le SI\nNiveau : 2",
            heading_path=("Suivi", "R24"),
            kind=SectionKind.RECORD,
            metadata={"rec_id": "R24"},
        )
        chunks = chunk_document(make_document(record), counter)
        assert len(chunks) == 1
        assert chunks[0].kind is SectionKind.RECORD
        assert chunks[0].metadata["rec_id"] == "R24"

    def test_oversized_record_repeats_its_identifier(self, counter: WordCounter) -> None:
        record = Section(
            text="Identifiant : R24\n" + "\n".join(f"Colonne {i} : {words(20)}" for i in range(10)),
            kind=SectionKind.RECORD,
        )
        chunks = chunk_document(
            make_document(record), counter, ChunkingConfig(target_tokens=60, max_tokens=80)
        )
        assert len(chunks) > 1
        assert all("Identifiant : R24" in c.text for c in chunks)

    def test_small_table_stays_whole(self, counter: WordCounter) -> None:
        table = Section(
            text="| Code | Action |\n|---|---|\n| E01 | Fermer la porte |",
            kind=SectionKind.TABLE,
        )
        assert len(chunk_document(make_document(table), counter)) == 1

    def test_oversized_table_repeats_its_header(self, counter: WordCounter) -> None:
        rows = "\n".join(f"| E{i:02d} | {words(15)} |" for i in range(20))
        table = Section(text=f"| Code | Action |\n|---|---|\n{rows}", kind=SectionKind.TABLE)
        chunks = chunk_document(
            make_document(table), counter, ChunkingConfig(target_tokens=60, max_tokens=80)
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert "| Code | Action |" in chunk.text
            assert "|---|---|" in chunk.text


class TestMergingAndMetadata:
    def test_tiny_neighbours_are_merged(self, counter: WordCounter) -> None:
        document = make_document(
            Section(text="Titre court.", heading_path=("Intro",)),
            Section(text="Suite du paragraphe.", heading_path=("Intro",)),
        )
        chunks = chunk_document(document, counter)
        assert len(chunks) == 1
        assert "Titre court." in chunks[0].text and "Suite du paragraphe." in chunks[0].text

    def test_sections_with_different_headings_are_not_merged(self, counter: WordCounter) -> None:
        document = make_document(
            Section(text="Texte A.", heading_path=("A",)),
            Section(text="Texte B.", heading_path=("B",)),
        )
        assert len(chunk_document(document, counter)) == 2

    def test_pages_and_kind_are_carried(self, counter: WordCounter) -> None:
        document = make_document(
            Section(text=words(200), heading_path=("Sécurité",), page_start=3, page_end=4)
        )
        chunks = chunk_document(document, counter, ChunkingConfig(target_tokens=50, max_tokens=60))
        assert all((c.page_start, c.page_end) == (3, 4) for c in chunks)
        assert all(c.source == "manuel.pdf" and c.doc_sha256 == SHA for c in chunks)


class TestIdentifiers:
    def test_ids_are_deterministic(self, counter: WordCounter) -> None:
        document = make_document(Section(text=words(300)))
        first = [c.id for c in chunk_document(document, counter)]
        second = [c.id for c in chunk_document(document, counter)]
        assert first == second and len(set(first)) == len(first)

    def test_ids_depend_on_the_file_content(self, counter: WordCounter) -> None:
        """Deux versions d'un fichier ne partagent aucun identifiant : la
        réindexation remplace les chunks au lieu de les laisser périmés, ce que
        faisaient les identifiants positionnels (source:page:index) d'origine."""
        section = Section(text=words(100))
        before = chunk_document(make_document(section), counter)
        after = chunk_document(make_document(section, sha=OTHER_SHA), counter)
        assert {c.id for c in before}.isdisjoint({c.id for c in after})

    def test_chunk_id_is_a_pure_function(self) -> None:
        assert chunk_id(SHA, 0) == chunk_id(SHA, 0)
        assert chunk_id(SHA, 0) != chunk_id(SHA, 1)


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"overlap_ratio": 0.6}, "overlap_ratio"),
            ({"overlap_ratio": -0.1}, "overlap_ratio"),
            ({"target_tokens": 800, "max_tokens": 700}, "target_tokens"),
        ],
    )
    def test_invalid_configurations_are_refused(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            ChunkingConfig(**kwargs)  # type: ignore[arg-type]


class TestRealisticCounter:
    def test_limit_is_respected_with_the_estimating_counter(self) -> None:
        """Vérification avec le compteur réel (par caractères), fil d'Ariane compris."""
        counter = EstimatedTokenCounter()
        config = ChunkingConfig(target_tokens=200, max_tokens=300)
        document = make_document(
            Section(text=". ".join(words(20) for _ in range(200)), heading_path=("Chapitre" * 5,))
        )
        chunks = chunk_document(document, counter, config)
        assert len(chunks) > 3
        assert all(counter.count(c.text) <= config.max_tokens for c in chunks)


class TestTokenCounter:
    def test_estimation_is_used_without_tokenizer(self, tmp_path: Path) -> None:
        counter = get_token_counter(tmp_path / "absent.json")
        assert isinstance(counter, EstimatedTokenCounter)
        assert counter.count("") == 0
        assert counter.count("Un texte de test") > 0

    def test_longer_text_costs_more(self) -> None:
        counter = EstimatedTokenCounter()
        assert counter.count(words(100)) > counter.count(words(10))


def test_chunk_payload_round_trip(counter: WordCounter) -> None:
    document = make_document(
        Section(text="Contenu.", heading_path=("Entretien",), page_start=2, page_end=2)
    )
    chunk: Chunk = chunk_document(document, counter)[0]
    payload = chunk.payload()
    assert payload["heading_path"] == ["Entretien"]
    assert payload["page_start"] == 2
    assert payload["kind"] == "prose"
