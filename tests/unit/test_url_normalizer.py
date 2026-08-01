from auditframework.extraction.url_normalizer import normalize_url


def test_scheme_and_www_are_unified():
    a = normalize_url("http://www.example.com/artigo")
    b = normalize_url("https://example.com/artigo")
    assert a == b


def test_trailing_slash_is_ignored():
    assert normalize_url("https://example.com/artigo/") == normalize_url("https://example.com/artigo")


def test_tracking_params_are_stripped():
    tracked = "https://example.com/artigo?utm_source=chatgpt&utm_medium=share&id=42"
    clean = "https://example.com/artigo?id=42"
    assert normalize_url(tracked) == normalize_url(clean)


def test_query_param_order_does_not_matter():
    a = normalize_url("https://example.com/x?b=2&a=1")
    b = normalize_url("https://example.com/x?a=1&b=2")
    assert a == b


def test_percent_encoded_space_is_normalized():
    a = normalize_url("https://example.com/a%20b")
    b = normalize_url("https://example.com/a b")
    assert a == b


def test_doi_variations_collapse_to_same_key():
    a = normalize_url("https://doi.org/10.1000/xyz123")
    b = normalize_url("10.1000/xyz123")
    c = normalize_url("https://DOI.ORG/10.1000/XYZ123")
    assert a == b == c


def test_distinct_urls_stay_distinct():
    assert normalize_url("https://example.com/a") != normalize_url("https://example.com/b")
