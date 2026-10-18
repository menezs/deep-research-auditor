import pytest
from auditframework.ingestion.barrier import MAX_BARRIER_WORDS, detect_barrier

@pytest.mark.parametrize("texto, tipo", [
    ("This website uses a security service to protect against malicious bots. This page is displayed while the website verifies you are not a bot.", "anti_bot"),
    ("Checking your browser before accessing pubmed.ncbi.nlm.nih.gov ... Click here if you are not automatically redirected after 5 seconds.", "anti_bot"),
    ("Verificando sua conexão Estamos confirmando que este acesso é legítimo antes de liberar o conteúdo. Isso leva apenas alguns segundos.", "anti_bot"),
    ("Error 403 Forbidden Forbidden Error 54113 Details: cache-gru-sbgr1930053-GRU 1791938954 1612442889 Varnish cache server", "http_error"),
    ("404 Not Found nginx", "http_error"),
    ("JavaScript is disabled In order to continue, we need to verify that you're not a robot. This requires JavaScript.", "anti_bot"),
    ("Essa pagina depende do javascript para abrir, favor habilitar o javascript do seu browser!", "requires_js_login_region"),
    ("# Access limited to the U.S. At this time, epocrates is not available outside the U.S.", "requires_js_login_region"),
    ("Now downloading WER9750-645-672-eng-fre.pdf... Back", "interstitial"),
    ("Please click here if you are not redirected within a few seconds.", "interstitial"),
    ("About Press Copyright Contact us Creators Advertise Developers Terms Privacy Policy & Safety How YouTube works Test new features", "empty_shell"),
    ("Relatório de Recomendação nº 878 - Testagem Molecular para Detecção de HPV", "no_content"),
])
def test_barreiras_reais_sao_detectadas(texto, tipo):
    barreira = detect_barrier(texto)
    assert barreira is not None and barreira.kind == tipo
    assert barreira.evidence == " ".join(texto.split())[:120]

def test_pagina_curta_real_nao_e_barreira():
    texto = ("A Proclamação da República Em 1889, sob fortes questionamentos dos fazendeiros, dos militares e da "
             "Igreja, o imperador D. Pedro II é deposto. ") * 3  # ~70 palavras, sem padrão
    assert detect_barrier(texto) is None

def test_documento_longo_que_menciona_cookies_e_captcha_nao_e_barreira():
    texto = "Este artigo discute captcha, consentimento e we use cookies em sites de notícias. " * 40
    assert len(texto.split()) >= MAX_BARRIER_WORDS
    assert detect_barrier(texto) is None
