from pathlib import Path

import pytest

from auditframework.extraction.pdf_marker_order import misplaced_marker_runs, repair_marker_order


class TestRepairMarkerOrder:
    """Função pura: recebe o Markdown convertido e os pares vindos da
    geometria do PDF. Tudo aqui é sobre quando NÃO mover."""

    PAIR = [("[1] [2]", "vaccination occurs before age 17.")]

    def test_marker_emitted_before_its_line_is_moved_to_the_end_of_it(self):
        markdown = "protection when <u>[1] [2]</u> vaccination occurs before age 17.\n\n## Fim\n"
        assert repair_marker_order(markdown, self.PAIR) == (
            "protection when vaccination occurs before age 17.<u>[1] [2]</u>\n\n## Fim\n"
        )

    def test_paragraph_break_after_the_marker_survives_the_move(self):
        """A quebra tem de sobreviver: engoli-la transformaria `\\n\\n- item`
        num ` - item` no meio da linha, e o bullet virava lixo no trecho."""
        markdown = "protection when <u>[1] [2]</u> \n\n- vaccination occurs before age 17.\n"
        assert repair_marker_order(markdown, self.PAIR) == (
            "protection when \n\n- vaccination occurs before age 17.<u>[1] [2]</u>\n"
        )

    def test_marker_already_in_the_right_place_is_left_alone(self):
        markdown = "protection when vaccination occurs before age 17. <u>[1] [2]</u>\n"
        assert repair_marker_order(markdown, self.PAIR) == markdown

    def test_nothing_moves_when_the_following_text_is_another_line(self):
        markdown = "protection when <u>[1] [2]</u> outra frase completamente diferente.\n"
        assert repair_marker_order(markdown, self.PAIR) == markdown

    def test_ambiguous_occurrence_is_left_alone(self):
        """O mesmo par aparecendo duas vezes: não há como saber qual mover."""
        linha = "protection when <u>[1] [2]</u> vaccination occurs before age 17.\n\n"
        assert repair_marker_order(linha + linha, self.PAIR) == linha + linha

    def test_reference_list_entry_is_never_moved(self):
        """Na lista de fontes o `[N]` vem ANTES do título por definição do
        formato; movê-lo para depois quebraria o parser de referências."""
        markdown = (
            "Corpo da resposta [9].\n\n"
            "## References\n\n"
            "[1] vaccination occurs before age 17.\nhttps://example.com/a\n"
        )
        assert repair_marker_order(markdown, [("[1]", "vaccination occurs before age 17.")]) == markdown

    def test_host_too_short_to_identify_is_left_alone(self):
        markdown = "texto <u>[1] [2]</u> ok.\n"
        assert repair_marker_order(markdown, [("[1] [2]", "ok.")]) == markdown

    def test_corrupted_host_line_is_left_alone(self):
        """Quando o conversor perdeu uma letra da linha (ligadura sem
        ToUnicode: "before" -> "be?ore"), não há como distinguir isso de
        "esta é outra linha" — então não se move. O chunker ainda recupera a
        afirmação inteira pela cauda da frase."""
        markdown = "protection when <u>[1] [2]</u> vaccination occurs be�ore age 17.\n"
        assert repair_marker_order(markdown, self.PAIR) == markdown

    def test_empty_pairs_is_a_no_op(self):
        markdown = "qualquer texto [1] aqui.\n"
        assert repair_marker_order(markdown, []) == markdown


class TestMisplacedMarkerRuns:
    def _pdf(self, path: Path, *, marker_y: float, marker_x: float, marker_size: float) -> None:
        pymupdf = pytest.importorskip("pymupdf")
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((70, 110), "the incidence of cervical cancer with the greatest protection when", fontsize=10.5)
        page.insert_text((70, 130), "vaccination occurs before age 17.", fontsize=10.5)
        page.insert_text((marker_x, marker_y), "[1] [2]", fontsize=marker_size)
        doc.save(str(path))
        doc.close()

    def test_superscript_marker_to_the_right_of_a_lower_line_is_detected(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        self._pdf(pdf, marker_y=127, marker_x=236, marker_size=7.9)
        assert misplaced_marker_runs(pdf) == [("[1] [2]", "vaccination occurs before age 17.")]

    def test_marker_in_body_size_is_not_a_superscript(self, tmp_path):
        pdf = tmp_path / "b.pdf"
        self._pdf(pdf, marker_y=127, marker_x=236, marker_size=10.5)
        assert misplaced_marker_runs(pdf) == []

    def test_marker_below_its_line_is_already_in_reading_order(self, tmp_path):
        pdf = tmp_path / "c.pdf"
        self._pdf(pdf, marker_y=134, marker_x=236, marker_size=7.9)
        assert misplaced_marker_runs(pdf) == []

    def test_unreadable_file_yields_no_pairs_instead_of_raising(self, tmp_path):
        bogus = tmp_path / "nao-e-pdf.pdf"
        bogus.write_text("isto nao e um PDF", encoding="utf-8")
        assert misplaced_marker_runs(bogus) == []
