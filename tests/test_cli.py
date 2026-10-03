"""The `banzuke` command line: every (sub)command answers --help, usage errors
exit with status 2 and a message rather than a traceback, the shared option
grammar parses, and the commands that need no training run end to end."""
import pytest

from banzuke.cli import build_parser, main
from banzuke.cli._common import CommandError, parse_seeds, parse_sets

HELP = [
    [], ["predict"], ["data"], ["data", "update"], ["data", "fetch"], ["data", "build"],
    ["backtest"], ["analyze"], ["conventions"], ["gtb"],
    ["explain"], ["explain", "build"], ["explain", "sheet"], ["explain", "detail"],
    ["explain", "explain"], ["explain", "docket"], ["explain", "calibration"],
    ["precedent"], ["precedent", "landing"], ["precedent", "cells"], ["precedent", "pair"],
    ["rules"],
]


@pytest.mark.parametrize("argv", HELP, ids=lambda a: " ".join(a) or "(top)")
def test_every_command_has_help(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main([*argv, "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    canonical = [{"explain": "detail"}.get(a, a) if i else a for i, a in enumerate(argv)]
    assert out.startswith("usage: banzuke" + "".join(f" {a}" for a in canonical))
    assert "options:" in out


def test_version_and_dispatch_table(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("banzuke ")
    # every leaf parser knows how to run and how to report its own usage
    ap = build_parser()
    for action in ap._subparsers._group_actions[0].choices.values():
        leaves = [action]
        if action._subparsers:
            leaves = list(action._subparsers._group_actions[0].choices.values())
        for leaf in leaves:
            assert callable(leaf.get_default("run")) and leaf.get_default("_parser") is leaf


@pytest.mark.parametrize("argv, message", [
    (["backtest", "--models", "Zz", "--end", "200403"], "unknown model"),
    (["predict", "--model", "Zz"], "unknown model"),
    (["analyze", "--model", "Zz"], "unknown model"),
    (["backtest", "--models", "R", "--start", "209901"], "no target basho"),
    (["backtest", "--models", "R", "--set", "oops", "--end", "200403"], "--set expects KEY=VALUE"),
    (["backtest", "--models", "R", "--seeds", "x", "--end", "200403"], "--seeds expects"),
    (["gtb", "--percentiles", "x"], "--percentiles expects"),
    (["gtb", "--start", "209901"], "no target basho"),
    (["gtb", "--model", "Zz", "--end", "202001"], "unknown model"),
    (["precedent", "landing", "Z9 1-1"], "bad rank spec"),
    (["precedent", "pair", "K 5-10", "M1 8-7", "--where", "nonsense column"], "--where 'nonsense column'"),
    (["precedent", "landing", "K1 5-10", "--where", "no_such_col > 1"], "--where 'no_such_col > 1'"),
    (["explain", "sheet"], "give a target basho or --all"),
    (["explain", "sheet", "202101", "--seeds", "7-9"], "no rows cache"),
    (["rules", "--cache", "/nonexistent"], "no cached frames"),
])
def test_usage_errors_exit_2_with_message(argv, message, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: banzuke " + argv[0]) and f"banzuke {argv[0]}" in err
    assert message in err


def test_missing_data_is_a_usage_error(monkeypatch, tmp_path, capsys):
    from banzuke import paths

    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["conventions"])
    assert e.value.code == 2
    assert "banzuke data update" in capsys.readouterr().err


def test_option_grammar():
    assert parse_seeds("0-2") == (0, 1, 2) and parse_seeds("0,3") == (0, 3) and parse_seeds(1) == (1,)
    assert parse_sets(["base.n_estimators=200", "gap=0.25", "context=false", "twin_unit=sk"]) == {
        "base": {"n_estimators": 200}, "gap": 0.25, "context": False, "twin_unit": "sk"}
    for bad in (["noequals"], ["a=1", "b"]):
        with pytest.raises(CommandError):
            parse_sets(bad)
    with pytest.raises(CommandError):
        parse_seeds("one")


def test_conventions_and_precedent_run(capsys):
    assert main(["conventions"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("=== convention audit") and "yokozuna never demoted" in out
    assert main(["precedent", "landing", "K1 5-10", "--where", "year>=2010"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("landing of K1 5-10:") and "cells: M" in out and "M3W" in out
    assert main(["precedent", "pair", "K 5-10", "M9-10 10-5"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("K 5-10 above M9-10 10-5:") and "2004+" in out


def _latest_candidates():
    from banzuke import forecast
    from banzuke.paths import load_transitions

    trans = load_transitions()
    cands = forecast.candidates(trans, forecast.latest_basho(trans))
    return cands.sort_values("position").reset_index(drop=True)


def test_predict_runs_with_every_override_kind(capsys):
    # R trains in milliseconds; the names come from the data so a data update
    # does not break the test: two adjacent mid-table maegashira, the M1E
    # man pinned far below where any model puts him, the top man demoted
    from banzuke.ranks import CLS_CHARS, MAEGASHIRA

    cands = _latest_candidates()
    top = cands.iloc[0]
    m = cands.loc[cands["rank_class"] == MAEGASHIRA, "shikona"].tolist()
    first, a, b = m[0], m[6], m[7]
    demote = f"{top['shikona']}={CLS_CHARS[int(top['rank_class']) + 1]}"
    assert main(["predict", "--model", "R", "--above", f"{b} > {a}", "--pin", f"{first}=M15E",
                 "--count", "K=3", "--class", demote]) == 0
    out = capsys.readouterr().out
    assert out.startswith("predicted makuuchi banzuke for") and "EAST" in out and "WEST" in out
    pinned = next(line for line in out.splitlines() if f" {first:<14} (" in line)
    assert "[pin]" in pinned and "<-" in out
    assert "\noverrides:\n" in out and f"'{b} > {a}'" in out
    assert "\nwarnings:\n" in out and "demoted by override" in out


def test_predict_interactive_loop(monkeypatch, capsys):
    cands = _latest_candidates()
    a, b = cands.loc[cands["rank_class"] == 4, "shikona"].iloc[[6, 7]]
    lines = iter([f"above {b} > {a}", "flags", f"unset {b}", "flags", "bogus", f"pin {a}=Z9Z", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    # a bad override on the command line is reported and dropped, not fatal
    assert main(["predict", "--model", "R", "--interactive", "--pin", f"{a}=Q1E"]) == 0
    out = capsys.readouterr().out
    assert "error: bad slot 'Q1E'" in out
    assert f'banzuke predict --above "{b} > {a}"' in out and "(no overrides)" in out
    assert "unknown command 'bogus'" in out and "(override not applied)" in out
    assert out.count("predicted makuuchi banzuke for") == 3  # initial, above, unset


def test_backtest_summarize_reads_csv(tmp_path, capsys):
    import pandas as pd

    from banzuke.harness import METRICS

    rows = [{"config": "base", "model": m, "seed": 0, "basho": 202001 + 2 * b,
             **dict.fromkeys(METRICS, 1.0), "exact": e / 42, "exact_n": e}
            for b in range(6) for m, e in (("Ar", 20), ("Aq", 18))]
    csv = tmp_path / "x_per_basho.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    assert main(["backtest", "--summarize", str(csv), "--baseline", "Ar"]) == 0
    out = capsys.readouterr().out
    assert "paired vs Ar" in out and "confirm 2020+" in out and "Aq" in out
