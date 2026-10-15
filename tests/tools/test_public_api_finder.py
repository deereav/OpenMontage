import os
import time

from tools.analysis.public_api_finder import PublicApiFinder, parse_readme

SAMPLE_README = """\
### APIs Covered Under APILayer Suite!
| API | Description | Call this API |
|:---|:---|:---|
| [IPstack](https://ipstack.com/) | Locate visitors by IP | [Run in Postman](https://example.com) |

## Index

### Geology
API | Description | Auth | HTTPS | CORS |
|:---|:---|:---|:---|:---|
| [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/fdsnws/event/1/) | Earthquakes data real-time | No | Yes | No |

### Weather
API | Description | Auth | HTTPS | CORS |
|:---|:---|:---|:---|:---|
| [Weatherstack](https://weatherstack.com/?utm_source=Github&utm_medium=Referral&lang=en) | Real-Time & Historical World Weather Data API | `apiKey` | Yes | Unknown |
| [Open-Meteo](https://open-meteo.com/) | Global weather forecast API for non-commercial use | No | Yes | Yes |
| [Dropbox Weather](https://example.com/dw) | Weather files | `OAuth` | Yes | Unknown | |
"""


def _finder(tmp_path) -> PublicApiFinder:
    (tmp_path / "README.md").write_text(SAMPLE_README, encoding="utf-8")
    return PublicApiFinder(cache_dir=tmp_path)


def test_parse_readme_skips_sponsor_block_and_normalizes_rows() -> None:
    entries = parse_readme(SAMPLE_README)

    assert [e["name"] for e in entries] == [
        "USGS Earthquake Hazards Program",
        "Weatherstack",
        "Open-Meteo",
        "Dropbox Weather",
    ]
    weatherstack = entries[1]
    assert weatherstack["auth"] == "apiKey"
    assert weatherstack["url"] == "https://weatherstack.com/?lang=en"
    assert entries[0]["auth"] == "none"
    # Trailing extra "|" cell is tolerated.
    assert entries[3]["auth"] == "OAuth"


def test_search_ranks_and_filters(tmp_path) -> None:
    result = _finder(tmp_path).execute({"query": "weather", "auth": "none", "cors_only": True})

    assert result.success, result.error
    assert [e["name"] for e in result.data["results"]] == ["Open-Meteo"]
    assert result.data["catalog"]["total_apis"] == 4
    assert result.data["catalog"]["stale"] is False


def test_query_matches_description(tmp_path) -> None:
    result = _finder(tmp_path).execute({"query": "earthquake"})

    assert result.data["results"][0]["category"] == "Geology"
    assert result.data["total_matches"] == 1


def test_list_categories(tmp_path) -> None:
    result = _finder(tmp_path).execute({"list_categories": True})

    assert result.data["categories"] == [
        {"category": "Geology", "count": 1},
        {"category": "Weather", "count": 3},
    ]


def test_stale_cache_is_used_when_refresh_fails(tmp_path, monkeypatch) -> None:
    finder = _finder(tmp_path)
    old = time.time() - 30 * 24 * 3600
    os.utime(tmp_path / "README.md", (old, old))

    import requests

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", offline)
    result = finder.execute({"query": "weather"})

    assert result.success, result.error
    assert result.data["catalog"]["stale"] is True
    assert "offline" in result.data["catalog"]["refresh_error"]
