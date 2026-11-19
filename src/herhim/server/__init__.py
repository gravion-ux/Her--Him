"""Local web server for the Her & Him experience.

This module provides a minimal stdlib HTTP server that serves the
web surfaces for both catalogues. No npm, no build step, just Python.
"""

from __future__ import annotations

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any

from herhim.catalog.loader import load_catalogue_cards
from herhim.catalog.registry import PersonaRegistry
from herhim.experience.her import load_her_surface
from herhim.experience.him import load_him_surface

__all__ = ["create_app", "run_server"]


class HerhimHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the Her & Him web server."""

    def log_message(self, format: str, *args: Any) -> None:
        """Log HTTP requests."""
        print(f"[{self.address_string()}] {format % args}")

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/":
            self._serve_index()
        elif self.path == "/her":
            self._serve_her()
        elif self.path == "/him":
            self._serve_him()
        elif self.path.startswith("/api/"):
            self._handle_api()
        else:
            super().do_GET()

    def _serve_index(self) -> None:
        """Serve the index page."""
        html = self._render_html(
            "/",
            "Her & Him",
            "Two complete AI-companion personality catalogues",
        )
        self._send_html(html)

    def _serve_her(self) -> None:
        """Serve the Her catalogue page."""
        surface = load_her_surface()
        html = self._render_html(
            "/her",
            surface.title,
            surface.subtitle,
            theme=surface.theme,
            default_persona=surface.default_persona,
        )
        self._send_html(html)

    def _serve_him(self) -> None:
        """Serve the Him catalogue page."""
        surface = load_him_surface()
        html = self._render_html(
            "/him",
            surface.title,
            surface.subtitle,
            theme=surface.theme,
            default_persona=surface.default_persona,
        )
        self._send_html(html)

    def _handle_api(self) -> None:
        """Handle API requests."""
        if self.path == "/api/herhim/catalogue":
            self._api_catalogue()
        elif self.path.startswith("/api/herhim/persona/"):
            persona_id = self.path[len("/api/herhim/persona/") :]
            self._api_persona(persona_id)
        elif self.path.startswith("/api/herhim/state/"):
            session_id = self.path[len("/api/herhim/state/") :]
            self._api_state(session_id)
        elif self.path == "/api/herhim/health":
            self._api_health()
        elif self.path.startswith("/api/herhim/voice/"):
            clip_id = self.path[len("/api/herhim/voice/") :]
            self._api_voice(clip_id)
        else:
            self._send_json({"error": "not found"}, 404)

    def _api_catalogue(self) -> None:
        """Return the merged catalogue."""
        her_personas = load_catalogue_cards("her")
        him_personas = load_catalogue_cards("him")
        all_personas = her_personas + him_personas

        data = {
            "schema_version": "1.0.0",
            "catalogues": ["her", "him"],
            "personas": [p.summary() for p in all_personas],
        }
        self._send_json(data)

    def _api_persona(self, persona_id: str) -> None:
        """Return a single persona's data."""
        registry = PersonaRegistry.load()
        try:
            persona = registry.get(persona_id)
            self._send_json(persona.to_dict())
        except Exception:
            self._send_json({"error": "persona not found"}, 404)

    def _api_state(self, session_id: str) -> None:
        """Return session state (placeholder)."""
        # Placeholder - actual session state would require session storage
        self._send_json({"session_id": session_id, "note": "session state not yet implemented"})

    def _api_health(self) -> None:
        """Health check endpoint."""
        self._send_json({"status": "ok", "version": "0.1.0"})

    def _api_voice(self, clip_id: str) -> None:
        """Return voice clip info."""
        # Placeholder - actual voice clips would be served from the samples directory
        self._send_json({"clip_id": clip_id, "note": "voice serving not yet implemented"})

    def _render_html(
        self,
        path: str,
        title: str,
        subtitle: str,
        theme: dict[str, str] | None = None,
        default_persona: str = "",
    ) -> str:
        """Render an HTML page."""
        theme = theme or {}
        bg = theme.get("--bg", "#1a1a2e")
        fg = theme.get("--fg", "#eaeaea")
        accent = theme.get("--accent", "#e94560")
        catalogue_filter = path[1:] if len(path) > 1 else "merged"

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: {bg};
            --fg: {fg};
            --accent: {accent};
        }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--fg);
            min-height: 100vh;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        h1 {{ margin: 0 0 0.5rem 0; }}
        h2 {{ color: var(--accent); margin-top: 2rem; }}
        .subtitle {{ opacity: 0.7; margin-bottom: 2rem; }}
        .persona-card {{
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 1rem;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            margin-bottom: 1rem;
            background: rgba(255,255,255,0.05);
        }}
        .persona-name {{ font-weight: bold; margin-bottom: 0.5rem; }}
        .persona-tagline {{ opacity: 0.8; font-size: 0.9rem; }}
        a.persona-link {{
            color: var(--accent);
            text-decoration: none;
        }}
        a.persona-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">{subtitle}</p>
        <p><a href="/">Back to catalogues</a></p>
        <h2>Personas</h2>
        <div id="personas"></div>
    </div>
    <script>
        fetch('/api/herhim/catalogue')
            .then(r => r.json())
            .then(data => {{
                const container = document.getElementById('personas');
                const personas = data.personas.filter(p => p.catalogue === '{catalogue_filter}');
                container.innerHTML = personas.map(p => `
                    <div class="persona-card">
                        <div>
                            <div class="persona-name">{{p.display_name}}</div>
                            <div class="persona-tagline">{{p.tagline}}</div>
                            <div style="font-size: 0.8rem; opacity: 0.6;">
                                {{p.locale}} • {{p.voice_name}}
                            </div>
                        </div>
                        <div>
                            <a class="persona-link" href="/chat?persona={{p.persona_id}}">Chat</a>
                        </div>
                    </div>
                `).join('');
            }});
    </script>
</body>
</html>'''

    def _send_html(self, html: str) -> None:
        """Send HTML response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    catalogues: tuple[str, ...] = ("her", "him"),
) -> None:
    """Run the web server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, HerhimHandler)
    print(f"Her & Him server running at http://{host}:{port}/")
    print("  /her   - Her (female) personas")
    print("  /him   - Him (male) personas")
    print("  /api/  - REST API")
    httpd.serve_forever()


def create_app() -> type[HerhimHandler]:
    """Create the handler for testing."""
    return HerhimHandler