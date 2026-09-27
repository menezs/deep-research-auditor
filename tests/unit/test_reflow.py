from auditframework.extraction.reflow import reflow_markdown


def test_sentence_broken_across_lines_is_rejoined():
    text = "A produtividade se manteve estável\nem todos os pilotos avaliados."
    assert reflow_markdown(text) == "A produtividade se manteve estável em todos os pilotos avaliados."


def test_hyphenated_word_split_across_lines_joins_without_the_hyphen():
    text = "A pesquisa foi transpa-\nrente no método."
    assert reflow_markdown(text) == "A pesquisa foi transparente no método."


def test_spurious_bullet_in_the_middle_of_a_sentence_is_dropped():
    text = "O estudo mostrou ganho de bem-estar\n- e produtividade estável."
    assert reflow_markdown(text) == "O estudo mostrou ganho de bem-estar e produtividade estável."


def test_finished_sentence_is_not_joined_with_the_next():
    text = "Primeira frase completa.\nSegunda frase completa."
    assert reflow_markdown(text) == text


def test_real_list_is_preserved():
    text = "Resultados:\n- Primeiro item\n- Segundo item"
    assert reflow_markdown(text) == text


def test_heading_and_table_are_never_joined():
    text = "## Um cabeçalho\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert reflow_markdown(text) == text


def test_url_on_the_next_line_is_not_glued_to_the_title():
    """Entrada de lista de fontes: o título não deve absorver a URL da
    linha seguinte, senão o parser de referências perde a separação."""
    text = "Titulo do documento\nhttps://example.com/artigo"
    assert reflow_markdown(text) == text


def test_blank_line_between_paragraphs_survives():
    text = "Parágrafo um termina aqui.\n\nParágrafo dois começa aqui."
    assert reflow_markdown(text) == text


def test_two_blank_lines_block_the_join():
    """Duas linhas em branco são fronteira de parágrafo de verdade, mesmo
    que a linha anterior não tenha terminado em pontuação."""
    text = "Trecho sem pontuação final\n\n\ncontinuação em minúscula"
    assert reflow_markdown(text) == "Trecho sem pontuação final\n\ncontinuação em minúscula"
