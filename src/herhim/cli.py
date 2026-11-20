"""Command-line interface for Her & Him.

This module provides the main entry point for the ``herhim`` command.
It supports catalogue management, chat sessions, and utility functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from herhim.catalog.loader import load_catalogue_cards
from herhim.catalog.quality import analyse_catalogue
from herhim.catalog.registry import PersonaRegistry
from herhim.catalog.validate import validate_persona
from herhim.errors import EXIT_INVALID_INPUT, EXIT_INVALID_STATE, EXIT_OK, EXIT_USAGE
from herhim.experience.her import load_her_surface
from herhim.experience.him import load_him_surface
from herhim.experience.session import Session
from herhim.guardrails.audit import AuditLog
from herhim.models.persona import Persona
from herhim.paths import catalogue_dir
from herhim.providers.base import LLMProvider
from herhim.providers.offline import OfflineProvider
from herhim.providers.openai_compat import OpenAICompatibleProvider
from herhim.server import run_server
from herhim.voice.factory import available_engines, build_voice_engine

__all__ = ["EXIT_INVALID_INPUT", "EXIT_INVALID_STATE", "EXIT_OK", "EXIT_USAGE", "main"]

#: The providers the chat command can build today.
PROVIDER_FACTORIES: dict[str, Callable[[], LLMProvider]] = {
    "offline": OfflineProvider,
    "openai-compatible": OpenAICompatibleProvider,
}


def cmd_catalogue(args: argparse.Namespace) -> int:
    """List personas in a catalogue or both."""
    if args.catalogue not in ("her", "him", "both"):
        print(f"unknown catalogue: {args.catalogue}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    personas: list[Persona] = []
    catalogs = ("her", "him") if args.catalogue == "both" else (args.catalogue,)
    for cat in catalogs:
        try:
            personas.extend(load_catalogue_cards(cat))
        except Exception as exc:
            print(f"error loading {cat}: {exc}", file=sys.stderr)
            return EXIT_INVALID_STATE

    if args.json_output:
        report = {
            "schema_version": "1.0.0",
            "catalogue": args.catalogue,
            "personas": [p.summary() for p in personas],
        }
        print(json.dumps(report, indent=2))
    else:
        for persona in personas:
            print(f"{persona.persona_id}: {persona.display_name}")

    return EXIT_OK


def cmd_show(args: argparse.Namespace) -> int:
    """Show details of one persona."""
    if not args.persona_id:
        print("missing --persona-id", file=sys.stderr)
        return EXIT_USAGE

    if args.catalogue not in ("her", "him", "both"):
        print(f"unknown catalogue: {args.catalogue}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    personas: list[Persona] = []
    catalogs = ("her", "him") if args.catalogue == "both" else (args.catalogue,)
    for cat in catalogs:
        try:
            personas.extend(load_catalogue_cards(cat))
        except Exception as exc:
            print(f"warning: skipping {cat}: {exc}", file=sys.stderr)

    persona = None
    for candidate in personas:
        if candidate.persona_id == args.persona_id:
            persona = candidate
            break

    if persona is None:
        print(f"persona not found: {args.persona_id}", file=sys.stderr)
        return EXIT_INVALID_STATE

    if args.json_output:
        print(json.dumps(persona.to_dict(), indent=2))
    else:
        print(f"ID: {persona.persona_id}")
        print(f"Name: {persona.display_name}")
        print(f"Tagline: {persona.tagline}")
        print(f"Gender: {persona.gender}")
        print(f"Locale: {persona.voice.locale}")
        print(f"Voice: {persona.voice.voice_name} ({persona.voice.voice_id})")
        print(f"Traits: {persona.traits.to_dict()}")
        print(f"Baseline: {persona.affect_baseline.to_dict()}")

    return EXIT_OK


def _build_provider(args: argparse.Namespace) -> tuple[LLMProvider | None, str]:
    """Build the provider the chat command should use.

    Returns the provider and an empty message, or ``None`` and the reason why
    no provider could be built.
    """
    factory = PROVIDER_FACTORIES.get(args.provider)
    if factory is None:
        return None, f"unknown provider: {args.provider}"
    provider = factory()
    if isinstance(provider, OpenAICompatibleProvider) and not provider.configured:
        return None, (
            "the openai-compatible provider is not configured: "
            "set HERHIM_LLM_BASE_URL and HERHIM_LLM_MODEL"
        )
    return provider, ""


def cmd_chat(args: argparse.Namespace) -> int:
    """Start an interactive chat session."""
    if args.catalogue not in ("her", "him"):
        print(f"unknown catalogue: {args.catalogue}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    provider, problem = _build_provider(args)
    if provider is None:
        print(problem, file=sys.stderr)
        return EXIT_INVALID_INPUT

    # Load the surface to get the default persona
    surface = load_her_surface() if args.catalogue == "her" else load_him_surface()

    # Get the default persona
    try:
        personas = load_catalogue_cards(args.catalogue)
        registry = PersonaRegistry(personas)
        persona = registry.get(surface.default_persona)
    except Exception as exc:
        print(f"error loading persona: {exc}", file=sys.stderr)
        return EXIT_INVALID_STATE

    session = Session(
        persona=persona,
        provider=provider,
        voice_engine=build_voice_engine("samples"),
    )

    if args.json_output:
        initial_state = session.state()
        print(json.dumps(initial_state, indent=2))
    else:
        print(f"Chat with {persona.display_name} [{persona.persona_id}]")
        mood = session.engine.mood.value
        stage = session.engine.relationship.stage().value
        print(f"Mood: {mood}, Stage: {stage}")
        print("---")

    turn_count = 0
    while turn_count < args.max_turns:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "bye"):
            break

        turn_count += 1
        try:
            result = session.turn(user_text)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"{persona.display_name}: {result.text}")
            print(f"  mood={result.mood}, stage={result.state['stage']}")

    return EXIT_OK


def cmd_card_export(args: argparse.Namespace) -> int:
    """Export every persona card in one catalogue to disk."""
    if args.catalogue not in ("her", "him"):
        print(f"unknown catalogue: {args.catalogue}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        personas = load_catalogue_cards(args.catalogue)
        for persona in personas:
            output = persona.to_dict()
            path = Path(args.output or f"{persona.persona_id}.json")
            path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"Wrote {path}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_STATE

    return EXIT_OK


def cmd_card_import(args: argparse.Namespace) -> int:
    """Import one persona card from a JSON file into its catalogue."""
    if not args.file:
        print("missing --file", file=sys.stderr)
        return EXIT_USAGE

    try:
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        persona = Persona.from_card(data)
        problems = validate_persona(persona)
        if problems:
            print(f"validation errors: {'; '.join(problems)}", file=sys.stderr)
            return EXIT_INVALID_STATE
        # Write to data directory
        target_dir = catalogue_dir(persona.catalogue)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{persona.persona_id}.json"
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Imported {target}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_STATE

    return EXIT_OK


def cmd_card(args: argparse.Namespace) -> int:
    """Export or import persona cards."""
    if args.command == "export":
        return cmd_card_export(args)
    if args.command == "import":
        return cmd_card_import(args)

    print(f"unknown command: {args.command}", file=sys.stderr)
    return EXIT_USAGE


def cmd_voices(args: argparse.Namespace) -> int:
    """List available voices."""
    engine = build_voice_engine(args.engine)

    voices = engine.list_voices()
    if args.json_output:
        report = {
            "schema_version": "1.0.0",
            "engine": engine.name,
            "voices": [v.to_dict() for v in voices],
        }
        print(json.dumps(report, indent=2))
    else:
        for voice in voices:
            print(f"{voice.voice_id}: {voice.voice_name} ({voice.locale})")

    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the local web server."""
    try:
        run_server(host=args.host, port=args.port)
    except OSError as exc:
        print(f"could not start server: {exc}", file=sys.stderr)
        return EXIT_INVALID_STATE

    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run diagnostic checks."""
    issues: list[str] = []
    for catalogue in ("her", "him"):
        try:
            personas = load_catalogue_cards(catalogue)
            report = analyse_catalogue(catalogue, personas)
            if not report.ok:
                for finding in report.findings:
                    if finding.severity in ("critical", "high"):
                        issues.append(f"{catalogue}: {finding.message}")
        except Exception as exc:
            issues.append(f"{catalogue}: {exc}")

    if args.json_output:
        result = {
            "schema_version": "1.0.0",
            "ok": len(issues) == 0,
            "issues": issues,
        }
        print(json.dumps(result, indent=2))
    elif issues:
        for issue in issues:
            print(f"ISSUE: {issue}")
    else:
        print("All diagnostics passed")

    return EXIT_OK if not issues else EXIT_INVALID_STATE


def cmd_audit(args: argparse.Namespace) -> int:
    """View audit history."""
    audit_path = Path(args.log or ".herhim-audit.jsonl")
    log = AuditLog.load(audit_path)

    if args.json_output:
        print(json.dumps(log.to_dict(), indent=2))
    else:
        for event in log.events[-10:]:
            print(f"{event.kind}: {event.subject} -> {event.decision}")

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="herhim",
        description="Her & Him - Two complete AI-companion personality catalogues",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument("--verbose", "-v", action="count", dest="verbose", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # catalogue command
    cp = subparsers.add_parser("catalogue", help="List catalogues")
    cp.add_argument(
        "--catalogue",
        choices=["her", "him", "both"],
        default="both",
        help="Which catalogue",
    )
    cp.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    cp.set_defaults(func=cmd_catalogue)

    # show command
    sp = subparsers.add_parser("show", help="Show persona details")
    sp.add_argument("--persona-id", required=False, dest="persona_id", help="Persona ID")
    sp.add_argument(
        "--catalogue",
        choices=["her", "him", "both"],
        default="both",
        help="Catalogue to search",
    )
    sp.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    sp.set_defaults(func=cmd_show)

    # chat command
    chat_p = subparsers.add_parser("chat", help="Start interactive chat")
    chat_p.add_argument(
        "--catalogue",
        choices=["her", "him"],
        default="her",
        help="Which catalogue",
    )
    chat_p.add_argument(
        "--provider",
        choices=sorted(PROVIDER_FACTORIES),
        default="offline",
        help="LLM provider to use",
    )
    chat_p.add_argument("--model", default=None, help="Model to use (for future)")
    chat_p.add_argument("--max-turns", type=int, default=100, help="Maximum turns")
    chat_p.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    chat_p.set_defaults(func=cmd_chat)

    # card command
    card_p = subparsers.add_parser("card", help="Export/import persona cards")
    card_p.add_argument("command", choices=["export", "import"], help="Export or import")
    card_p.add_argument("--catalogue", choices=["her", "him"], help="Catalogue for export")
    card_p.add_argument("--output", "-o", help="Output file")
    card_p.add_argument("--file", "-f", help="Input file for import")
    card_p.set_defaults(func=cmd_card)

    # voices command
    vp = subparsers.add_parser("voices", help="List available voices")
    vp.add_argument("--engine", default="samples", choices=available_engines(), help="Voice engine")
    vp.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    vp.set_defaults(func=cmd_voices)

    # serve command
    sv = subparsers.add_parser("serve", help="Start local web server")
    sv.add_argument("--host", default="127.0.0.1", help="Bind address")
    sv.add_argument("--port", type=int, default=8080, help="Port number")
    sv.set_defaults(func=cmd_serve)

    # doctor command
    doc_p = subparsers.add_parser("doctor", help="Run diagnostics")
    doc_p.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    doc_p.set_defaults(func=cmd_doctor)

    # audit command
    audit_p = subparsers.add_parser("audit", help="View audit history")
    audit_p.add_argument("--log", help="Audit log path")
    audit_p.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    audit_p.set_defaults(func=cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    handler: Callable[[argparse.Namespace], int] = args.func
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
