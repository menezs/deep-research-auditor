from __future__ import annotations

import pytest

pytest.importorskip("docx")

from auditframework.extraction.loaders import DocxAnswerLoader


def _add_hyperlink(paragraph, url: str, text: str):
    """Injeta um `<w:hyperlink>` de verdade no paragrafo (nao existe API
    publica no python-docx para isso) — necessario para reproduzir o bug
    real que motivou a reescrita do loader: `paragraph.runs` pula runs
    dentro de hyperlink, `paragraph.iter_inner_content()` nao."""
    import docx
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


class TestDocxAnswerLoader:
    def test_superscript_run_becomes_bracket_marker(self, tmp_path):
        import docx

        document = docx.Document()
        p = document.add_paragraph("bilheteira")
        sup_run = p.add_run("1")
        sup_run.font.superscript = True
        p.add_run(".")
        path = tmp_path / "resposta.docx"
        document.save(path)

        text = DocxAnswerLoader().load(path)
        assert "bilheteira[1]." in text

    def test_multi_digit_superscript_is_preserved(self, tmp_path):
        import docx

        document = docx.Document()
        p = document.add_paragraph("saturacao")
        sup_run = p.add_run("13")
        sup_run.font.superscript = True
        path = tmp_path / "resposta.docx"
        document.save(path)

        text = DocxAnswerLoader().load(path)
        assert "saturacao[13]" in text

    def test_plain_number_without_superscript_is_not_touched(self, tmp_path):
        import docx

        document = docx.Document()
        document.add_paragraph("Lancado em 2008 com 38 filmes.")
        path = tmp_path / "resposta.docx"
        document.save(path)

        text = DocxAnswerLoader().load(path)
        assert "2008" in text and "[2008]" not in text
        assert "38" in text and "[38]" not in text

    def test_heading_style_becomes_markdown_heading(self, tmp_path):
        import docx

        document = docx.Document()
        document.add_paragraph("Referências citadas", style="Heading 4")
        document.add_paragraph("Corpo normal.")
        path = tmp_path / "resposta.docx"
        document.save(path)

        text = DocxAnswerLoader().load(path)
        lines = [line for line in text.splitlines() if line.strip()]
        assert lines[0] == "#### Referências citadas"
        assert lines[1] == "Corpo normal."

    def test_hyperlink_text_is_preserved_not_dropped(self, tmp_path):
        """Regressao: `paragraph.runs` (usado pelo loader antigo) pula
        runs dentro de `<w:hyperlink>`, perdendo a URL inteira quando ela
        e um hyperlink de verdade (nao texto solto) — confirmado contra o
        .docx real do Gemini antes desta correcao."""
        import docx

        document = docx.Document()
        p = document.add_paragraph("Titulo do artigo, ")
        _add_hyperlink(p, "https://example.com/artigo", "https://example.com/artigo")
        path = tmp_path / "resposta.docx"
        document.save(path)

        text = DocxAnswerLoader().load(path)
        assert "https://example.com/artigo" in text

    def test_superscript_inside_hyperlink_is_still_wrapped(self, tmp_path):
        import docx

        document = docx.Document()
        p = document.add_paragraph("texto")
        _add_hyperlink(p, "https://example.com/nota", "1")
        # marca o run recem-criado dentro do hyperlink como superscript
        hyperlink_run = p._p.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r")[-1]
        rpr = hyperlink_run.makeelement(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr", {}
        )
        vert_align = hyperlink_run.makeelement(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}vertAlign",
            {"{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val": "superscript"},
        )
        rpr.append(vert_align)
        hyperlink_run.insert(0, rpr)
        path = tmp_path / "resposta.docx"
        document.save(path)

        text = DocxAnswerLoader().load(path)
        assert "texto[1]" in text
