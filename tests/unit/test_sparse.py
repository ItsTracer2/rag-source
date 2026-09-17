from rag_source.store.sparse import (
    average_token_length,
    encode_document,
    encode_query,
    term_id,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_strips_accents(self) -> None:
        assert tokenize("Procédure DE Sécurité") == ["procedure", "de", "securite"]

    def test_keeps_identifiers_whole(self) -> None:
        """« MW-104 » ou « R24 » sont les termes que l'on tape le plus souvent."""
        assert tokenize("Référence MW-104 et code E01") == [
            "reference",
            "mw-104",
            "et",
            "code",
            "e01",
        ]
        assert "r24" in tokenize("Que dit la R24 ?")

    def test_drops_punctuation_and_single_characters(self) -> None:
        assert tokenize("a, b. (c) — test !") == ["test"]

    def test_accented_and_unaccented_match(self) -> None:
        assert tokenize("sécurité") == tokenize("securite")


class TestTermId:
    def test_is_stable_across_calls(self) -> None:
        assert term_id("securite") == term_id("securite")

    def test_differs_between_terms(self) -> None:
        assert term_id("securite") != term_id("surete")

    def test_fits_in_32_bits(self) -> None:
        assert 0 <= term_id("un-terme-quelconque") < 2**32


class TestEncodeDocument:
    def test_empty_text_gives_empty_vector(self) -> None:
        assert len(encode_document("...", average_length=10)) == 0

    def test_repeated_term_saturates(self) -> None:
        """Répéter un mot cent fois ne doit pas valoir cent fois un mot cité une fois.

        C'est tout l'intérêt de BM25 : sans saturation, un document qui répète un
        terme écrase les autres au classement.
        """
        once = encode_document("plateau", average_length=1)
        many = encode_document(" ".join(["plateau"] * 100), average_length=1)
        assert many.values[0] > once.values[0]
        assert many.values[0] < once.values[0] * 3

    def test_long_document_is_penalized(self) -> None:
        """À fréquence égale, un terme pèse moins dans un document long."""
        short = encode_document("plateau tournant", average_length=10)
        long_text = "plateau " + " ".join(f"mot{i}" for i in range(200))
        long = encode_document(long_text, average_length=10)
        key = term_id("plateau")
        assert short.values[short.indices.index(key)] > long.values[long.indices.index(key)]

    def test_indices_and_values_match(self) -> None:
        vector = encode_document("un texte de test avec plusieurs mots", average_length=8)
        assert len(vector.indices) == len(vector.values) == len(vector)
        assert all(value > 0 for value in vector.values)


class TestEncodeQuery:
    def test_presence_only(self) -> None:
        vector = encode_query("plateau plateau tournant")
        assert set(vector.values) == {1.0}
        assert len(vector.indices) == 2  # « plateau » n'est compté qu'une fois

    def test_matches_document_terms(self) -> None:
        document = encode_document("Nettoyer le plateau tournant", average_length=4)
        query = encode_query("plateau")
        assert query.indices[0] in document.indices


def test_average_token_length() -> None:
    assert average_token_length([]) == 1.0
    assert average_token_length(["un deux trois", "quatre cinq"]) == 2.5
