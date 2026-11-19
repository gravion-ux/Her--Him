"""Web server CLI entry point."""

from __future__ import annotations

import argparse
import sys

from herhim.errors import EXIT_OK, EXIT_USAGE
from herhim.server import run_server

__all__ = ["EXIT_OK", "EXIT_USAGE", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the server."""
    parser = argparse.ArgumentParser(
        prog="herhim-serve",
        description="Start the local web server for Her & Him",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the server CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())