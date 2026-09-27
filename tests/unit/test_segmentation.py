from auditframework.indexing.segmentation import sentence_boundaries


def _split(text: str) -> list[str]:
    """Sentenças como o chunker as vê — texto fatiado nas fronteiras."""
    cuts = [0, *sentence_boundaries(text), len(text)]
    return [text[a:b].strip() for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


def test_plain_sentences_are_split():
    assert _split("Primeira frase. Segunda frase.") == ["Primeira frase.", "Segunda frase."]


def test_decimal_and_thousands_separator_do_not_split():
    assert _split("O valor foi de R$ 50.000 no período.") == ["O valor foi de R$ 50.000 no período."]
    assert _split("Quase 2.900 funcionários participaram.") == ["Quase 2.900 funcionários participaram."]


def test_legal_abbreviation_does_not_split():
    assert _split("Conforme o art. 5º da lei.") == ["Conforme o art. 5º da lei."]
    assert _split("A UE (Reg. UE 2024/1689) classifica o risco.") == [
        "A UE (Reg. UE 2024/1689) classifica o risco."
    ]


def test_period_inside_parentheses_does_not_split():
    text = "O estudo (publicado em 2025. Revisado depois) mostrou ganho."
    assert _split(text) == [text]


def test_initials_do_not_split():
    assert _split("Segundo J. Silva e A. C. Souza, houve queda.") == [
        "Segundo J. Silva e A. C. Souza, houve queda."
    ]


def test_etc_followed_by_uppercase_does_not_split():
    assert _split("Usaram tabelas, gráficos, etc. Depois publicaram.") == [
        "Usaram tabelas, gráficos, etc. Depois publicaram."
    ]


def test_ellipsis_and_interrobang_collapse_into_one_boundary():
    assert _split("Será possível?! Talvez sim.") == ["Será possível?!", "Talvez sim."]


def test_period_before_lowercase_is_not_a_boundary():
    assert _split("Versão v2.5 saiu. depois veio outra") == ["Versão v2.5 saiu. depois veio outra"]


def test_boundary_before_a_citation_marker():
    """`.` seguido de `[` conta como fronteira — é o formato "frase.[4]"."""
    assert sentence_boundaries("Primeira frase. [4] Segunda.")[0] == len("Primeira frase. ")
