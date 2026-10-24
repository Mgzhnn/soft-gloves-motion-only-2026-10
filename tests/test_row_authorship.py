"""Per-row authorship of request timelines.

The importer stamped one annotator and the provenance ``human_timeline_csv_retrospective`` on every row, so a timeline
whose rows were mostly agent drafts accepted by the annotator read as entirely human-written. An optional
``row_author`` column now records who wrote each row; without the column the importer's output is unchanged, so
existing ledgers and every record that hashes them stay valid. Software only, participant N = 0.
"""
import csv
import json

import pytest

import import_requests
from sg_offline.ledger import read_opportunity_ledger, to_opportunities

HEADER = ["start_s", "end_s", "left", "right", "left_start_s", "left_end_s", "note"]
ROWS = [dict(start_s="0", end_s="1", left="NQ", right="NQ", left_start_s="", left_end_s="", note="rest"),
        dict(start_s="1", end_s="2", left="G", right="NQ", left_start_s="1.2", left_end_s="1.8", note="left reach"),
        dict(start_s="2", end_s="3", left="NQ", right="NQ", left_start_s="", left_end_s="", note="holding")]


def _csv(path, rows, header):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    return path


def _import(tmp_path, name, rows, header):
    src = _csv(tmp_path / f"{name}.csv", rows, header); out = tmp_path / f"{name}.json"
    import_requests.main([str(src), str(out), "--participant", "p", "--session", "s", "--annotator", "ann",
                          "--clock", "c", "--duration-s", "3"])
    return out


def test_importer_records_who_wrote_each_row(tmp_path):
    authors = ["agent_draft_accepted", "annotator", "agent_draft_edited"]
    rows = [dict(r, row_author=a) for r, a in zip(ROWS, authors)]
    out = _import(tmp_path, "authored", rows, HEADER + ["row_author"])
    data = json.loads(out.read_text())
    assert [o["row_author"] for o in data["opportunities"]] == authors
    ledger = read_opportunity_ledger(out)
    assert [e.row_author for e in ledger["entries"]] == authors and {e.annotator for e in ledger["entries"]} == {"ann"}
    scored, _ = to_opportunities(ledger, 0.)
    assert [o.provenance.endswith(f"|row_author={a}") for o, a in zip(scored, authors)] == [True, True, True]


@pytest.mark.parametrize("value", ["", "annotator_one", "agent"])
def test_importer_refuses_a_missing_or_unknown_row_author(tmp_path, value):
    rows = [dict(r, row_author="annotator") for r in ROWS]; rows[1]["row_author"] = value
    with pytest.raises(ValueError, match="row_author must be one of"):
        _import(tmp_path, "bad", rows, HEADER + ["row_author"])
    assert not (tmp_path / "bad.json").exists()


def test_reader_refuses_an_unknown_row_author(tmp_path):
    out = _import(tmp_path, "plain", ROWS, HEADER)
    data = json.loads(out.read_text()); data["opportunities"][0]["row_author"] = "someone"
    bad = tmp_path / "bad_ledger.json"; bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="row_author must be one of"):
        read_opportunity_ledger(bad)


def test_GUARD_timeline_without_the_column_imports_unchanged(tmp_path):
    out = _import(tmp_path, "plain", ROWS, HEADER)
    data = json.loads(out.read_text())
    assert all("row_author" not in o for o in data["opportunities"])
    ledger = read_opportunity_ledger(out)
    assert all(e.row_author is None for e in ledger["entries"])
    scored, _ = to_opportunities(ledger, 0.)
    assert all("row_author" not in o.provenance for o in scored)
    from sg_offline.ledger import ROW_AUTHORS
    assert ROW_AUTHORS == ("annotator", "agent_draft_accepted", "agent_draft_edited")
