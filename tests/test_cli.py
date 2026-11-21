"""Tests for the command-line interface."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import pytest

from herhim import cli
from herhim.catalog.loader import load_catalogue_cards
from herhim.catalog.quality import QualityFinding, QualityReport
from herhim.errors import EXIT_INVALID_INPUT, EXIT_INVALID_STATE, EXIT_OK, EXIT_USAGE, ProviderError
from herhim.guardrails.audit import AuditLog
from herhim.models.persona import Persona
from herhim.providers.offline import OfflineProvider
from herhim.providers.openai_compat import OpenAICompatibleProvider


class _FailingProvider:
    """A provider that always fails, for the CLI error path."""

    name = "failing"

    def complete(self, messages, *, temperature=0.9, max_tokens=400, context=None):
        """Always raise a provider error."""
        raise ProviderError("provider exploded")


def _namespace(**overrides: object) -> argparse.Namespace:
    """Build an argparse namespace carrying the CLI's defaults."""
    defaults: dict[str, object] = {
        "command": None,
        "catalogue": "both",
        "persona_id": None,
        "json_output": False,
        "provider": "offline",
        "max_turns": 3,
        "output": None,
        "file": None,
        "engine": "samples",
        "host": "127.0.0.1",
        "port": 8080,
        "log": None,
        "verbose": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_card(directory: Path, persona: Persona) -> Path:
    """Write one persona card as JSON and return its path."""
    path = directory / f"{persona.persona_id}.json"
    path.write_text(json.dumps(persona.to_dict(), indent=2), encoding="utf-8")
    return path


def _her_persona() -> Persona:
    """Return the first bundled "her" persona."""
    return load_catalogue_cards("her")[0]


# -- the parser and the entry point --------------------------------------


def test_main_without_a_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """A bare invocation shows the help text and reports a usage error."""
    assert cli.main([]) == EXIT_USAGE
    assert "usage: herhim" in capsys.readouterr().out


def test_the_parser_is_named_after_the_console_script() -> None:
    """The parser advertises the installed command name."""
    assert cli.build_parser().prog == "herhim"


def test_main_dispatches_to_the_selected_command(capsys: pytest.CaptureFixture[str]) -> None:
    """A known command reaches its handler."""
    assert cli.main(["catalogue", "--catalogue", "her"]) == EXIT_OK
    assert "her-" in capsys.readouterr().out


def test_the_module_can_be_run_as_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the module directly exits with the command's status."""
    monkeypatch.setattr(sys, "argv", ["herhim"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert excinfo.value.code == EXIT_USAGE


# -- catalogue ------------------------------------------------------------


def test_catalogue_lists_both_catalogues(capsys: pytest.CaptureFixture[str]) -> None:
    """The default listing covers the "her" and "him" catalogues."""
    assert cli.main(["catalogue"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "her-" in out
    assert "him-" in out


def test_catalogue_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """The JSON listing carries every persona from both catalogues."""
    assert cli.main(["catalogue", "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    expected = len(load_catalogue_cards("her")) + len(load_catalogue_cards("him"))
    assert report["schema_version"] == "1.0.0"
    assert report["catalogue"] == "both"
    assert len(report["personas"]) == expected


def test_catalogue_rejects_an_unknown_name(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown catalogue name is refused."""
    assert cli.cmd_catalogue(_namespace(catalogue="other")) == EXIT_INVALID_INPUT
    assert "unknown catalogue" in capsys.readouterr().err


def test_catalogue_reports_a_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A catalogue that cannot be read is reported on stderr."""
    monkeypatch.setattr(cli, "load_catalogue_cards", _raise_on_load)
    assert cli.cmd_catalogue(_namespace(catalogue="her")) == EXIT_INVALID_STATE
    assert "error loading her" in capsys.readouterr().err


# -- show -----------------------------------------------------------------


def test_show_prints_the_persona_details(capsys: pytest.CaptureFixture[str]) -> None:
    """A persona is described field by field."""
    persona = _her_persona()
    assert cli.cmd_show(_namespace(persona_id=persona.persona_id)) == EXIT_OK
    out = capsys.readouterr().out
    assert persona.persona_id in out
    assert persona.display_name in out
    assert persona.tagline in out


def test_show_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """The JSON form is the persona's full card."""
    persona = _her_persona()
    args = _namespace(persona_id=persona.persona_id, json_output=True)
    assert cli.cmd_show(args) == EXIT_OK
    card = json.loads(capsys.readouterr().out)
    assert card["name"] == persona.display_name


def test_show_requires_a_persona_id(capsys: pytest.CaptureFixture[str]) -> None:
    """A missing persona id is a usage error."""
    assert cli.cmd_show(_namespace(persona_id=None)) == EXIT_USAGE
    assert "missing --persona-id" in capsys.readouterr().err


def test_show_rejects_an_unknown_catalogue(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown catalogue name is refused."""
    args = _namespace(persona_id="her-001-sassy", catalogue="other")
    assert cli.cmd_show(args) == EXIT_INVALID_INPUT
    assert "unknown catalogue" in capsys.readouterr().err


def test_show_reports_a_missing_persona(capsys: pytest.CaptureFixture[str]) -> None:
    """A persona id that does not exist is reported."""
    assert cli.cmd_show(_namespace(persona_id="does-not-exist")) == EXIT_INVALID_STATE
    assert "persona not found" in capsys.readouterr().err


def test_show_surfaces_load_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A catalogue that cannot be read is reported before the lookup fails."""
    monkeypatch.setattr(cli, "load_catalogue_cards", _raise_on_load)
    assert cli.cmd_show(_namespace(persona_id="her-001-sassy")) == EXIT_INVALID_STATE
    err = capsys.readouterr().err
    assert "warning: skipping her" in err
    assert "persona not found" in err


# -- provider selection ---------------------------------------------------


def test_select_provider_builds_the_offline_provider() -> None:
    """``--provider offline`` selects the offline provider."""
    provider, problem = cli._build_provider(_namespace(provider="offline"))
    assert problem == ""
    assert isinstance(provider, OfflineProvider)


def test_select_provider_refuses_an_unconfigured_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """An openai-compatible provider without configuration is refused."""
    monkeypatch.delenv("HERHIM_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("HERHIM_LLM_MODEL", raising=False)
    provider, problem = cli._build_provider(_namespace(provider="openai-compatible"))
    assert provider is None
    assert "not configured" in problem


def test_select_provider_uses_a_configured_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured endpoint is used instead of the offline provider."""
    monkeypatch.setenv("HERHIM_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("HERHIM_LLM_MODEL", "llama3")
    provider, problem = cli._build_provider(_namespace(provider="openai-compatible"))
    assert problem == ""
    assert isinstance(provider, OpenAICompatibleProvider)


def test_select_provider_refuses_an_unknown_provider() -> None:
    """A provider that is not registered is refused."""
    provider, problem = cli._build_provider(_namespace(provider="nonsense"))
    assert provider is None
    assert "unknown provider" in problem


# -- chat -----------------------------------------------------------------


def test_chat_rejects_an_unknown_catalogue(capsys: pytest.CaptureFixture[str]) -> None:
    """Chatting needs one specific catalogue."""
    assert cli.cmd_chat(_namespace(catalogue="both")) == EXIT_INVALID_INPUT
    assert "unknown catalogue" in capsys.readouterr().err


def test_chat_reports_a_missing_persona(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A catalogue that cannot be read stops the chat."""
    monkeypatch.setattr(cli, "load_catalogue_cards", _raise_on_load)
    assert cli.cmd_chat(_namespace(catalogue="her")) == EXIT_INVALID_STATE
    assert "error loading persona" in capsys.readouterr().err


def test_chat_runs_until_the_user_leaves(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A turn is answered and the goodbye keyword ends the session."""
    answers = iter(["hello there", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.cmd_chat(_namespace(catalogue="her", max_turns=5)) == EXIT_OK
    out = capsys.readouterr().out
    assert "Chat with" in out
    assert "Mood:" in out


def test_chat_stops_at_end_of_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End of input ends the session without an error."""
    monkeypatch.setattr("builtins.input", _raise_eof)
    assert cli.cmd_chat(_namespace(catalogue="her", max_turns=5)) == EXIT_OK
    assert "Chat with" in capsys.readouterr().out


def test_chat_skips_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty input is ignored rather than counted as a turn."""
    answers = iter(["", "   ", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.cmd_chat(_namespace(catalogue="her", max_turns=5)) == EXIT_OK


def test_chat_reports_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing provider is reported and the session carries on."""
    monkeypatch.setattr(cli, "_build_provider", lambda args: (_FailingProvider(), ""))
    answers = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.cmd_chat(_namespace(catalogue="her", max_turns=5)) == EXIT_OK
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "provider exploded" in err


def test_chat_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """The JSON form prints the session state before any turn."""
    args = _namespace(catalogue="her", json_output=True, max_turns=0)
    assert cli.cmd_chat(args) == EXIT_OK
    state = json.loads(capsys.readouterr().out)
    assert "session_id" in state
    assert "mood" in state


def test_chat_json_output_includes_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The JSON form prints the result of every turn."""
    answers = iter(["hello there"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers, ""))
    args = _namespace(catalogue="her", json_output=True, max_turns=1)
    assert cli.cmd_chat(args) == EXIT_OK
    out = capsys.readouterr().out
    assert '"persona_id"' in out
    assert '"rationale"' in out


# -- card -----------------------------------------------------------------


def test_card_export_writes_one_file_per_persona(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exporting writes a card file for every persona in the catalogue."""
    monkeypatch.chdir(tmp_path)
    assert cli.cmd_card(_namespace(command="export", catalogue="her")) == EXIT_OK
    written = sorted(tmp_path.glob("her-*.json"))
    assert len(written) == len(load_catalogue_cards("her"))
    assert "Wrote" in capsys.readouterr().out


def test_card_export_rejects_an_unknown_catalogue(capsys: pytest.CaptureFixture[str]) -> None:
    """Exporting needs one specific catalogue."""
    args = _namespace(command="export", catalogue="other")
    assert cli.cmd_card(args) == EXIT_INVALID_INPUT
    assert "unknown catalogue" in capsys.readouterr().err


def test_card_export_reports_a_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A catalogue that cannot be read stops the export."""
    monkeypatch.setattr(cli, "load_catalogue_cards", _raise_on_load)
    args = _namespace(command="export", catalogue="her")
    assert cli.cmd_card(args) == EXIT_INVALID_STATE
    assert "error:" in capsys.readouterr().err


def test_card_import_files_a_valid_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid card is written into its catalogue directory."""
    persona = _her_persona()
    source = _write_card(tmp_path, persona)
    monkeypatch.setattr(cli, "catalogue_dir", lambda catalogue: tmp_path / "data" / catalogue)
    args = _namespace(command="import", file=str(source))
    assert cli.cmd_card(args) == EXIT_OK
    target = tmp_path / "data" / "her" / f"{persona.persona_id}.json"
    assert target.exists()
    assert "Imported" in capsys.readouterr().out


def test_card_import_requires_a_file(capsys: pytest.CaptureFixture[str]) -> None:
    """Importing without a source file is a usage error."""
    assert cli.cmd_card(_namespace(command="import", file=None)) == EXIT_USAGE
    assert "missing --file" in capsys.readouterr().err


def test_card_import_refuses_an_invalid_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A card that fails validation is refused and not filed."""
    persona = _her_persona()
    card = persona.to_dict()
    card["extensions"]["herhim"]["tagline"] = ""
    source = tmp_path / "broken.json"
    source.write_text(json.dumps(card, indent=2), encoding="utf-8")
    monkeypatch.setattr(cli, "catalogue_dir", lambda catalogue: tmp_path / "data" / catalogue)
    args = _namespace(command="import", file=str(source))
    assert cli.cmd_card(args) == EXIT_INVALID_STATE
    assert "validation errors" in capsys.readouterr().err


def test_card_import_reports_a_broken_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A file that is not JSON is reported."""
    source = tmp_path / "broken.json"
    source.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(cli, "catalogue_dir", lambda catalogue: tmp_path / "data" / catalogue)
    args = _namespace(command="import", file=str(source))
    assert cli.cmd_card(args) == EXIT_INVALID_STATE
    assert "error:" in capsys.readouterr().err


def test_card_rejects_an_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Only export and import are supported."""
    assert cli.cmd_card(_namespace(command="sideways")) == EXIT_USAGE
    assert "unknown command" in capsys.readouterr().err


# -- voices ---------------------------------------------------------------


def test_voices_lists_the_available_voices(capsys: pytest.CaptureFixture[str]) -> None:
    """The sample engine reports its voices."""
    assert cli.main(["voices"]) == EXIT_OK
    assert "en_US" in capsys.readouterr().out


def test_voices_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """The JSON form names the engine and every voice."""
    assert cli.main(["voices", "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["engine"] == "samples"
    assert len(report["voices"]) > 0


# -- serve, doctor, and audit --------------------------------------------


def test_serve_starts_the_local_web_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The serve command hands its address to the web server."""
    captured: dict[str, object] = {}

    def fake_run_server(host="127.0.0.1", port=8080, catalogues=("her", "him")):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli, "run_server", fake_run_server)
    assert cli.cmd_serve(_namespace(host="127.0.0.2", port=8123)) == EXIT_OK
    assert captured["host"] == "127.0.0.2"
    assert captured["port"] == 8123


def test_serve_reports_a_port_that_cannot_be_bound(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A server that cannot bind is reported rather than raised."""

    def boom(host="127.0.0.1", port=8080, catalogues=("her", "him")):
        raise OSError("address already in use")

    monkeypatch.setattr(cli, "run_server", boom)
    assert cli.cmd_serve(_namespace()) == EXIT_INVALID_STATE
    assert "could not start server" in capsys.readouterr().err


def test_doctor_reports_a_clean_catalogue(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No findings means the diagnostics pass."""
    monkeypatch.setattr(cli, "analyse_catalogue", _clean_report)
    assert cli.cmd_doctor(_namespace()) == EXIT_OK
    assert "All diagnostics passed" in capsys.readouterr().out


def test_doctor_reports_a_finding(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A high-severity finding is listed and the exit code changes."""
    monkeypatch.setattr(cli, "analyse_catalogue", _finding_report)
    assert cli.cmd_doctor(_namespace()) == EXIT_INVALID_STATE
    assert "ISSUE: her: personas overlap" in capsys.readouterr().out


def test_doctor_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The JSON form carries the issue list."""
    monkeypatch.setattr(cli, "analyse_catalogue", _finding_report)
    assert cli.cmd_doctor(_namespace(json_output=True)) == EXIT_INVALID_STATE
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert len(report["issues"]) == 2


def test_doctor_reports_a_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A catalogue that cannot be read is listed as an issue."""
    monkeypatch.setattr(cli, "load_catalogue_cards", _raise_on_load)
    assert cli.cmd_doctor(_namespace()) == EXIT_INVALID_STATE
    assert "ISSUE: her: catalogue unavailable" in capsys.readouterr().out


def test_audit_prints_the_last_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each audited decision is printed on its own line."""
    monkeypatch.chdir(tmp_path)
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("guardrail.scope", "her-001-sassy", "allow")
    args = _namespace(log=str(tmp_path / "audit.jsonl"))
    assert cli.cmd_audit(args) == EXIT_OK
    assert "guardrail.scope: her-001-sassy -> allow" in capsys.readouterr().out


def test_audit_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The JSON form carries every audited event."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("relationship.stage", "her-001-sassy", "record")
    args = _namespace(log=str(tmp_path / "audit.jsonl"), json_output=True)
    assert cli.cmd_audit(args) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["events"][0]["kind"] == "relationship.stage"


def test_audit_handles_a_missing_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A log that does not exist yet is not an error."""
    monkeypatch.chdir(tmp_path)
    assert cli.cmd_audit(_namespace()) == EXIT_OK


# -- helpers --------------------------------------------------------------


def _raise_on_load(catalogue: str) -> list[Persona]:
    """Stand in for a catalogue loader that cannot read the data."""
    raise RuntimeError("catalogue unavailable")


def _raise_eof(prompt: str = "") -> str:
    """Stand in for an input stream that has ended."""
    raise EOFError


def _clean_report(catalogue: str, personas: list[Persona]) -> QualityReport:
    """Stand in for a catalogue that passes every quality rule."""
    return QualityReport(catalogue=catalogue, findings=())


def _finding_report(catalogue: str, personas: list[Persona]) -> QualityReport:
    """Stand in for a catalogue with one high-severity finding."""
    finding = QualityFinding("CAT-CON-001", "high", catalogue, "personas overlap")
    return QualityReport(catalogue=catalogue, findings=(finding,))


def _mixed_report(catalogue: str, personas: list[Persona]) -> QualityReport:
    """A report with both high and medium findings for branch coverage."""
    finding_high = QualityFinding("CAT-CON-001", "high", catalogue, "personas overlap")
    finding_medium = QualityFinding("CAT-COV-001", "medium", catalogue, "missing coverage")
    return QualityReport(catalogue=catalogue, findings=(finding_high, finding_medium))


# -- chat with unknown provider ----------------------------------------------


def test_chat_rejects_an_unknown_provider(capsys: pytest.CaptureFixture[str]) -> None:
    """Chat fails when the provider is not registered."""
    args = _namespace(catalogue="her", provider="nonexistent")
    assert cli.cmd_chat(args) == EXIT_INVALID_INPUT
    err = capsys.readouterr().err
    assert "unknown provider" in err


# -- doctor branch coverage -------------------------------------------------


def test_doctor_with_mixed_severity_findings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Doctor only reports high/critical findings, not medium/low."""
    monkeypatch.setattr(cli, "analyse_catalogue", _mixed_report)
    assert cli.cmd_doctor(_namespace()) == EXIT_INVALID_STATE
    out = capsys.readouterr().out
    assert "ISSUE: her: personas overlap" in out
    assert "missing coverage" not in out
