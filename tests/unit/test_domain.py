import pytest

from rag_source.domain import Chunk, DocumentFormat, LoadedDocument, Section, SectionKind

SHA = "a" * 64


class TestSection:
    def test_defaults(self) -> None:
        section = Section(text="Contenu")
        assert section.kind is SectionKind.PROSE
        assert section.heading_path == ()
        assert dict(section.metadata) == {}

    def test_rejects_blank_text(self) -> None:
        with pytest.raises(ValueError, match="vide"):
            Section(text="  \n ")

    def test_pages_must_be_set_together(self) -> None:
        with pytest.raises(ValueError, match="ensemble"):
            Section(text="x", page_start=1)

    @pytest.mark.parametrize(("start", "end"), [(0, 1), (3, 2)])
    def test_rejects_invalid_page_range(self, start: int, end: int) -> None:
        with pytest.raises(ValueError, match="Plage de pages"):
            Section(text="x", page_start=start, page_end=end)

    def test_metadata_is_immutable_and_decoupled_from_caller(self) -> None:
        source = {"rec_id": "R24"}
        section = Section(text="x", metadata=source)
        source["rec_id"] = "R99"
        assert section.metadata["rec_id"] == "R24"
        with pytest.raises(TypeError):
            section.metadata["rec_id"] = "R1"  # type: ignore[index]


def test_document_rejects_bad_sha() -> None:
    with pytest.raises(ValueError, match="sha256"):
        LoadedDocument(
            source="a.pdf", sha256="abc", format=DocumentFormat.PDF, title="A", sections=()
        )


def test_chunk_payload_is_serializable() -> None:
    chunk = Chunk(
        id="id-1",
        source="dir/a.xlsx",
        doc_sha256=SHA,
        index=3,
        text="Guide › R24\nTexte",
        token_count=5,
        heading_path=("Guide", "R24"),
        kind=SectionKind.RECORD,
        page_start=None,
        page_end=None,
        metadata={"rec_id": "R24"},
    )
    payload = chunk.payload()
    assert payload == {
        "source": "dir/a.xlsx",
        "doc_sha256": SHA,
        "index": 3,
        "text": "Guide › R24\nTexte",
        "token_count": 5,
        "heading_path": ["Guide", "R24"],
        "kind": "record",
        "page_start": None,
        "page_end": None,
        "metadata": {"rec_id": "R24"},
    }
