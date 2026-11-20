"""Tests for the local web server and its console script."""

from __future__ import annotations

import json
import runpy
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from herhim import server
from herhim.errors import EXIT_OK
from herhim.server import HerhimHandler, create_app, run_server
from herhim.server import cli as server_cli
from herhim.server.cli import build_parser


@pytest.fixture
def live_server():
    """Serve the handler on an ephemeral port for the duration of a test."""
    httpd = HTTPServer(("127.0.0.1", 0), HerhimHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes, str]:
    """Fetch a URL and return its status, body, and content type."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


def _get_json(url: str) -> tuple[int, dict]:
    """Fetch a URL and decode its JSON body."""
    status, body, _ = _get(url)
    return status, json.loads(body.decode("utf-8"))


# -- the application factory ----------------------------------------------


def test_create_app_returns_the_handler_class() -> None:
    """The factory hands back the class the server is built from."""
    assert create_app() is HerhimHandler


# -- the HTML surfaces ----------------------------------------------------


def test_the_index_page_is_served(live_server: str) -> None:
    """The root page is the catalogue landing page."""
    status, body, content_type = _get(f"{live_server}/")
    assert status == 200
    assert "text/html" in content_type
    assert b"<!DOCTYPE html>" in body
    assert b"Her & Him" in body


def test_the_her_page_is_served(live_server: str) -> None:
    """The "her" page is themed from its catalogue manifest."""
    status, body, content_type = _get(f"{live_server}/her")
    assert status == 200
    assert "text/html" in content_type
    text = body.decode("utf-8")
    assert "Personas" in text
    assert "p.catalogue === 'her'" in text


def test_the_him_page_is_served(live_server: str) -> None:
    """The "him" page is themed from its catalogue manifest."""
    status, body, _ = _get(f"{live_server}/him")
    assert status == 200
    text = body.decode("utf-8")
    assert "p.catalogue === 'him'" in text


def test_the_page_renders_the_persona_template(live_server: str) -> None:
    """The page embeds a JavaScript template literal for each persona."""
    _, body, _ = _get(f"{live_server}/her")
    text = body.decode("utf-8")
    assert "${p.display_name}" in text
    assert "${p.tagline}" in text


def test_an_unknown_path_is_not_served(live_server: str) -> None:
    """A path the server does not know about is a 404."""
    status, _, _ = _get(f"{live_server}/no-such-page.html")
    assert status == 404


# -- the API --------------------------------------------------------------


def test_the_catalogue_api_returns_every_persona(live_server: str) -> None:
    """The catalogue endpoint merges both catalogues."""
    status, data = _get_json(f"{live_server}/api/herhim/catalogue")
    assert status == 200
    assert data["schema_version"] == "1.0.0"
    assert data["catalogues"] == ["her", "him"]
    assert len(data["personas"]) == 52


def test_a_persona_can_be_fetched(live_server: str) -> None:
    """A known persona id returns that persona's card."""
    status, data = _get_json(f"{live_server}/api/herhim/persona/her-001-sassy")
    assert status == 200
    assert data["name"] == "Maya"


def test_an_unknown_persona_is_a_404(live_server: str) -> None:
    """A persona id that does not exist is a 404."""
    status, data = _get_json(f"{live_server}/api/herhim/persona/no-such-persona")
    assert status == 404
    assert data["error"] == "persona not found"


def test_session_state_is_a_placeholder(live_server: str) -> None:
    """The state endpoint echoes the session id back."""
    status, data = _get_json(f"{live_server}/api/herhim/state/abc123")
    assert status == 200
    assert data["session_id"] == "abc123"


def test_the_health_endpoint_reports_ok(live_server: str) -> None:
    """The health endpoint answers with the package version."""
    status, data = _get_json(f"{live_server}/api/herhim/health")
    assert status == 200
    assert data == {"status": "ok", "version": "0.1.0"}


def test_a_voice_clip_is_a_placeholder(live_server: str) -> None:
    """The voice endpoint echoes the clip id back."""
    status, data = _get_json(f"{live_server}/api/herhim/voice/clip-1")
    assert status == 200
    assert data["clip_id"] == "clip-1"


def test_an_unknown_api_route_is_a_404(live_server: str) -> None:
    """An unknown API route is a 404."""
    status, data = _get_json(f"{live_server}/api/herhim/nope")
    assert status == 404
    assert data["error"] == "not found"


# -- the server entry point -----------------------------------------------


def test_run_server_reports_where_it_listens(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The server prints its address and the routes it serves."""
    created = []

    class _FakeServer:
        def __init__(self, address, handler):
            self.address = address
            self.handler = handler
            self.served = False
            created.append(self)

        def serve_forever(self):
            self.served = True

    monkeypatch.setattr(server, "HTTPServer", _FakeServer)
    run_server(host="127.0.0.1", port=1234)
    out = capsys.readouterr().out
    assert "http://127.0.0.1:1234/" in out
    assert "/her" in out
    assert "/api/" in out
    assert created[0].address == ("127.0.0.1", 1234)
    assert created[0].handler is HerhimHandler
    assert created[0].served is True


# -- the server console script --------------------------------------------


def test_the_server_cli_is_named_after_the_console_script() -> None:
    """The parser advertises the installed command name."""
    assert build_parser().prog == "herhim-serve"


def test_the_server_cli_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The console script passes its address to the server."""
    captured: dict[str, object] = {}

    def fake_run_server(host="127.0.0.1", port=8080, catalogues=("her", "him")):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(server_cli, "run_server", fake_run_server)
    assert server_cli.main(["--host", "127.0.0.2", "--port", "9999"]) == EXIT_OK
    assert captured["host"] == "127.0.0.2"
    assert captured["port"] == 9999


def test_the_server_module_can_be_run_as_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the module directly starts the server and exits cleanly."""
    monkeypatch.setattr(sys, "argv", ["herhim-serve", "--port", "0"])
    monkeypatch.setattr(server, "run_server", lambda **kwargs: None)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(server_cli.__file__, run_name="__main__")
    assert excinfo.value.code == EXIT_OK
