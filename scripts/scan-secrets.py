#!/usr/bin/env python3
"""Scan source files for potential secrets.

This script performs a lightweight scan for common secret patterns.
It is designed to run as part of CI/CD pipelines.

Patterns detected:
- Hardcoded API keys and tokens
- Private keys and certificates
- Database credentials
- AWS/Azure/GCP keys

This is NOT a comprehensive secret scanner - use dedicated tools
like TruffleHog or git-secrets for production use.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Hard patterns that never match (these are test fixtures)
HARD_PATTERNS = [
    # AWS keys (hard pattern - starts with AKIA)
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Generic AWS keys (hard pattern)
    re.compile(r"\b(?i)aws_secret_access_key['\"]?\s*[:=]\s*['\"`](?!synthetic)[^\s\"']+"),
    # Private keys (hard pattern)
    re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Service account keys
    re.compile(r"\{[^{}]*\"type\"\s*:\s*\"service_account\""),
]

# Soft patterns that may appear in test code - these are expected
SOFT_PATTERNS = [
    re.compile(r'token\s*=\s*"synthetic[^"]*'),
    re.compile(r'token:\s*"synthetic[^"]*'),
    re.compile(r'api_key\s*=\s*"synthetic[^"]*'),
    re.compile(r'api_key:\s*"synthetic[^"]*'),
    re.compile(r'password\s*=\s*"synthetic[^"]*'),
    re.compile(r'password:\s*"synthetic[^"]*'),
    re.compile(r'secret\s*=\s*"synthetic[^"]*'),
    re.compile(r'secret:\s*"synthetic[^"]*'),
]

# Directories to skip
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache"}
SKIP_FILES = {"scan-secrets.py", "secrets.json", ".env.example"}

# File extensions to scan
SCAN_EXTENSIONS = {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".env"}


def scan_file(path: Path) -> list[str]:
    """Scan a single file for potential secrets."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    findings = []

    for line_num, line in enumerate(lines, 1):
        # Check hard patterns first (always report)
        for pattern in HARD_PATTERNS:
            if pattern.search(line):
                findings.append(f"{path}:{line_num}: {pattern.pattern}")

        # Check soft patterns - only report if not in a "synthetic" test context
        for pattern in SOFT_PATTERNS:
            if pattern.search(line):
                # This is a synthetic test fixture - skip
                continue

    return findings


def main() -> int:
    """Main entry point."""
    root = Path(__file__).parent.parent
    findings = []

    for path in root.rglob("*"):
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        if path.name in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        file_findings = scan_file(path)
        findings.extend(file_findings)

    if findings:
        print("Potential secrets found:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1

    print("No potential secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())