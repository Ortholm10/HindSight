from hindsight_core.skills import available_types, load_skill


def test_every_taxonomy_type_has_a_skill_file():
    assert available_types() == [f"L{n:02d}" for n in range(1, 13)]


def test_skill_files_are_plain_text():
    assert "shift(1)" in load_skill("L03")
