from dataclasses import dataclass

import tools.video.stock_sources as stock_sources
from lib.corpus import Corpus
from lib.location_filter import check_location
from tools.video.corpus_builder import CorpusBuilder

UTAH = {
    "include_terms": ["utah", "zion", "moab", "canyonlands"],
    "exclude_terms": ["jordan", "arizona", "california"],
}


def test_no_spec_keeps_everything() -> None:
    assert check_location("Wadi Rum, southern Jordan", None) == (True, "")


def test_excluded_place_is_rejected() -> None:
    keep, reason = check_location("Wadi Rum ... in southern Jordan 60 km east of Aqaba", UTAH)
    assert keep is False
    assert "Jordan" in reason


def test_include_term_overrides_exclude() -> None:
    keep, _ = check_location("Lake Powell on the border of Utah and Arizona", UTAH)
    assert keep is True


def test_matching_is_whole_word_and_case_insensitive() -> None:
    assert check_location("Michael Jordanson timelapse", UTAH)[0] is True
    assert check_location("ZION NATIONAL PARK", UTAH)[0] is True
    assert check_location("Moab, UT|Arches", {"include_terms": ["moab, ut"]})[0] is True


def test_require_include_rejects_unlabelled_clips() -> None:
    strict = dict(UTAH, require_include=True)
    assert check_location("red rock canyon at sunset", strict)[0] is False
    assert check_location("", strict)[0] is False
    assert check_location("red rock canyon at sunset", UTAH)[0] is True


@dataclass
class _Candidate:
    clip_id: str
    source_tags: str


class _Source:
    name = "fake"

    def is_available(self) -> bool:
        return True

    def search(self, query, filters):
        return [
            _Candidate("wadi-rum", "Wadi Rum valley in southern Jordan"),
            _Candidate("zion", "Zion National Park, Utah"),
            _Candidate("generic", "desert road"),
        ]


def test_corpus_builder_drops_wrong_location_before_download(monkeypatch, tmp_path) -> None:
    processed: list[str] = []

    def fake_process(self, *, cand, **kwargs):
        processed.append(cand.clip_id)
        return type("Rec", (), {"clip_id": cand.clip_id})()

    monkeypatch.setattr(stock_sources, "available_sources", lambda: [_Source()])
    monkeypatch.setattr(stock_sources, "source_summary", lambda: {})
    monkeypatch.setattr(CorpusBuilder, "_process_candidate", fake_process)
    monkeypatch.setattr(Corpus, "has", lambda self, clip_id: False)

    result = CorpusBuilder().execute({
        "corpus_dir": str(tmp_path / "corpus"),
        # The query names Utah; it must not vouch for the Jordan clip.
        "queries": [{"query": "utah red rock desert"}],
        "filters": {"location": UTAH},
    })

    assert result.success, result.error
    assert processed == ["zion", "generic"]
    assert result.data["clips_rejected_location"] == 1
    assert result.data["location_rejections"][0]["clip_id"] == "wadi-rum"
