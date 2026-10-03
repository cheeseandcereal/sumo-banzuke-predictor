"""The "Guess the Banzuke" benchmark (banzuke.gtb, `banzuke gtb`): the archive
parser against a hand-written page with every layout quirk met so far, the
field percentiles, placing a model's rows in a field, the cache policy, and
the command end to end without touching the network."""
import numpy as np
import pandas as pd
import pytest

from banzuke import gtb
from banzuke.cli import main
from banzuke.cli.gtb import parse_percentiles
from conftest import fake_results


def page(rows, header=("Place", "Shikona", "Basho<br />Count", "Rank", "Correct<br />Guesses",
                       "Bulls-Eye", "Hits", "Total<br />Points", "Record")):
    """An archive page like GTBScoreBasho.aspx: a layout table around the
    result table, header cells with <br />, data rows with optional tr/td
    classes and a '*' after debut players."""
    body = "".join(f"<tr{cls}>{cells}</tr>" for cls, cells in rows)
    return f"""<html><body><table class="layout"><tr><td class="layoutright">
    <h2>GTB Results for Aki 2026</h2><table class="gtbscore" border="0"><thead><tr>
    {''.join(f'<th>{h}</th>' for h in header)}</tr></thead><tbody>{body}</tbody></table>
    <font size='-1'>* - Welcome, new players!</font></td></tr></table></body></html>"""


def cells(place, name, be, hits, link=True, debut=False, top=False, full=True):
    """One result row; full=False drops the archive's optional Rank and
    Record columns (absent on some basho)."""
    name_html = f"<a href='GTBPlayerBasho.aspx?p=1&b=202609'>{name}</a>" if link else name
    td = "<td class=\"gtbtoplist\">" if top else "<td>"
    rank = '<td class="right">3</td>' if full else ""
    record = '<td class="wlrecord">10-5</td>' if full else ""
    return (f'<td class="right">{place}</td>{td}{name_html}{" *" if debut else ""}</td>'
            f'<td class="right">9</td>{rank}<td class="right">{be + hits}</td>'
            f'<td class="right">{be}</td><td class="right">{hits}</td>'
            f'<td class="right">{2 * be + hits}</td>{record}')


ENTRIES = [(1, "Alpha", 33, 3, {}), (2, "Beta", 30, 5, {"top": True}),
           (3, "Gamma", 28, 2, {"debut": True}), (4, "Delta", 20, 4, {}),
           (6, "Epsilon", 10, 0, {})]  # places may skip numbers, as the archive's do
CLASSES = ["", " class=\"gtbyoung\"", " class=\"gtbdebut\"", "", ""]
ROWS = [(cls, cells(*e[:4], **e[4])) for cls, e in zip(CLASSES, ENTRIES)]
SHORT_HEADER = ("Place", "Shikona", "Basho<br />Count", "Correct<br />Guesses", "Bulls-Eye", "Hits",
                "Total<br />Points")


def test_parse_page_handles_the_archive_layout():
    df = gtb.parse_page(page(ROWS))
    assert list(df.columns) == gtb.COLUMNS[1:]
    assert df["player"].tolist() == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    assert df["points"].tolist() == [69, 65, 58, 44, 20]
    assert df["place"].tolist() == [1, 2, 3, 4, 6]
    assert (df["points"] == 2 * df["bullseyes"] + df["hits"]).all()
    # the optional Rank/Record columns may be missing: header-driven parsing
    short = page([(cls, cells(*e[:4], full=False, **e[4])) for cls, e in zip(CLASSES, ENTRIES)],
                 header=SHORT_HEADER)
    assert gtb.parse_page(short)["points"].tolist() == [69, 65, 58, 44, 20]
    assert gtb.parse_page(page([])).empty  # results not posted yet: header only


@pytest.mark.parametrize("html", [
    "<html><body>nothing here</body></html>",
    page(ROWS, header=("Place", "Shikona", "Points")),
    page([("", cells(1, "Alpha", 33, 3).replace('<td class="right">69</td>', '<td class="right">68</td>'))]),
])
def test_parse_page_rejects_unexpected_pages(html):
    with pytest.raises(gtb.ArchiveError):
        gtb.parse_page(html)


def test_field_percentiles_and_winner():
    entries = pd.concat([gtb.parse_page(page(ROWS)).assign(basho=b) for b in (202607, 202609)])
    f = gtb.field(entries, (50, 75, 100))
    assert f.index.tolist() == [202607, 202609]
    assert list(f.columns) == ["entries", "pts_p50", "pts_p75", "pts_p100", "be_p50", "be_p75",
                               "be_p100", "winner"]
    row = f.loc[202609]
    assert row["entries"] == 5 and row["winner"] == "Alpha"
    assert row["pts_p50"] == 58 and row["pts_p75"] == 65 and row["pts_p100"] == 69
    assert row["be_p50"] == 28 and row["be_p100"] == 33
    assert gtb.field(entries, (90,))["pts_p90"].iloc[0] == np.percentile([69, 65, 58, 44, 20], 90)


def test_place_in_field_averages_seeds_then_ranks():
    entries = gtb.parse_page(page(ROWS)).assign(basho=202609)
    results = pd.DataFrame([{"basho": 202609, "seed": s, "gtb_points": p, "exact_n": e}
                            for s, p, e in ((0, 66, 30), (1, 64, 28))]  # mean 65: ties Beta
                           + [{"basho": 202611, "seed": 0, "gtb_points": 80, "exact_n": 40}])
    p = gtb.place_in_field(entries, results, "Ar")
    assert p.index.tolist() == [202609]  # 202611 has no field
    row = p.loc[202609]
    assert row["Ar_pts"] == 65 and row["Ar_be"] == 29
    assert row["place"] == 2  # one entry above; the tie shares Beta's place
    assert row["pct"] == (3 + 0.5) / 5  # beats three, half credit for the tie
    assert row["behind"] == 4


def test_fetch_caches_pages_with_results_only(monkeypatch, tmp_path):
    served = {202609: page(ROWS), 202611: page([])}
    calls = []

    def fake_get(url):
        basho = int(url.rsplit("=", 1)[1])
        calls.append(basho)
        return served[basho]

    monkeypatch.setattr(gtb, "_get", fake_get)
    monkeypatch.setattr(gtb, "DELAY_S", 0)
    monkeypatch.setattr(gtb, "GTB_CACHE", tmp_path)
    entries = gtb.load_results([202609, 202611], progress=False)
    assert entries["basho"].tolist() == [202609] * 5 and list(entries.columns) == gtb.COLUMNS
    assert (tmp_path / "202609.html").exists() and not (tmp_path / "202611.html").exists()
    gtb.load_results([202609, 202611], progress=False)
    assert calls == [202609, 202611, 202611]  # cached page reused, empty one asked again
    gtb.fetch_results(202609, fresh=True)
    assert calls[-1] == 202609


def test_percentile_grammar():
    assert parse_percentiles("50,75,100") == (50.0, 75.0, 100.0) and parse_percentiles(90) == (90.0,)
    for bad in ("x", "", "50,101", "-1"):
        with pytest.raises(Exception, match="--percentiles expects"):
            parse_percentiles(bad)


def test_command_runs_offline_against_csv_rows(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gtb, "_get", lambda url: page(ROWS))
    monkeypatch.setattr(gtb, "DELAY_S", 0)
    monkeypatch.setattr(gtb, "GTB_CACHE", tmp_path / "gtb")
    rows = fake_results({(m, b): p / 2 for b in (202607, 202609) for m, p in (("Ar", 65), ("Aq", 44))},
                        gtb_points=lambda r: 2 * r["exact_n"])
    csv = tmp_path / "x_per_basho.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    argv = ["gtb", "--start", "202607", "--end", "202609", "--csv", str(csv)]
    with pytest.raises(SystemExit) as e:
        main(argv)  # two labels in the file, none chosen
    assert e.value.code == 2 and "pick one with --model" in capsys.readouterr().err
    assert not (tmp_path / "gtb").exists()  # rejected before fetching anything
    assert main([*argv, "--model", "Ar", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("=== Guess the Banzuke field: 2 banzuke 202607-202609")
    assert "Ar: 65.0 points/basho (32.5 exact), mean place 2 of 5, beats 70% of the field" in out
    field = pd.read_csv(tmp_path / "o_field.csv", index_col=0)
    assert field.index.tolist() == [202607, 202609] and field["place"].tolist() == [2, 2]
    assert len(pd.read_csv(tmp_path / "o_entries.csv")) == 10
