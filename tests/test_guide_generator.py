from src.guide.guide_generator import generate_investigation_checklist


def test_checklist_length_within_bounds_for_no_keywords(monkeypatch):
    monkeypatch.setattr("src.guide.guide_generator.ANTHROPIC_API_KEY", "")
    items = generate_investigation_checklist("경미한 접촉사고 청구.", [])
    assert 3 <= len(items) <= 5


def test_checklist_maps_delayed_report_keyword_to_specific_checkpoint(monkeypatch):
    monkeypatch.setattr("src.guide.guide_generator.ANTHROPIC_API_KEY", "")
    items = generate_investigation_checklist("사고 발생 한 달 후 청구 접수.", ["지연신고"])
    assert any("신고 지연 사유" in item for item in items)


def test_checklist_covers_each_known_keyword(monkeypatch):
    monkeypatch.setattr("src.guide.guide_generator.ANTHROPIC_API_KEY", "")
    keywords = ["나이롱환자", "한방병원 장기입원", "지연신고", "블랙박스 미장착"]
    items = generate_investigation_checklist("장기 통원치료 및 지연신고 정황.", keywords)
    assert 3 <= len(items) <= 5
    assert any("통원치료" in item for item in items)
    assert any("한방병원" in item for item in items)


def test_checklist_deduplicates_repeated_keywords(monkeypatch):
    monkeypatch.setattr("src.guide.guide_generator.ANTHROPIC_API_KEY", "")
    items = generate_investigation_checklist("지연신고 정황.", ["지연신고", "지연신고"])
    assert items.count(next(i for i in items if "신고 지연 사유" in i)) == 1


def test_checklist_items_are_checklist_style_not_dialogue(monkeypatch):
    monkeypatch.setattr("src.guide.guide_generator.ANTHROPIC_API_KEY", "")
    items = generate_investigation_checklist("사고 경위서 접수.", ["과다 청구"])
    assert all("?" not in item for item in items)
