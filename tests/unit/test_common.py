from pathlib import Path

from auditframework.common import make_run_id


def test_make_run_id_includes_file_stem(tmp_path: Path):
    answer_file = tmp_path / "resposta.md"
    answer_file.write_text("conteudo de teste", encoding="utf-8")
    run_id = make_run_id(answer_file)
    assert run_id.startswith("resposta_")


def test_make_run_id_differs_for_different_content(tmp_path: Path):
    file_a = tmp_path / "a.md"
    file_b = tmp_path / "b.md"
    file_a.write_text("conteudo A", encoding="utf-8")
    file_b.write_text("conteudo B", encoding="utf-8")
    assert make_run_id(file_a) != make_run_id(file_b)
