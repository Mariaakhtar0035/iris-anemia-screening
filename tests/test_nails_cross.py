from src.nails_cross import parse_name


def test_file_name_labels_with_the_dataset_typos():
    assert parse_name("Non-AnemicP-010 (3)") == ("Non-AnemicP-010", 0)
    assert parse_name("Anemic-FN-001") == ("Anemic-FN-001", 1)
    assert parse_name("Anrmic-Fin-007 (10)") == ("Anrmic-Fin-007", 1)    # typo in the file names
    assert parse_name("Non-anmeic-012") == ("Non-anmeic-012", 0)
