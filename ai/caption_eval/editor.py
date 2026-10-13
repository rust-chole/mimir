"""A small local web page for writing the gold transcripts (``python -m ai.caption_eval serve``).

Standard library only, bound to 127.0.0.1. Per case: the audio with 0.75x / 1x speed,
MIMIR's captioned words as clickable chips (click = play from just before that word;
the word being spoken is highlighted), the gold text, the verified names, and Save.
"Finished" removes the TODO line so the case is scored.

Keys in the text box: Ctrl+Space play/pause, Ctrl+Left back 3 s, Ctrl+S save.
"""
from __future__ import annotations

import html
import json
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ai.caption_eval import cases as case_store
from ai.caption_eval import metrics

_STYLE = """
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6b6b70; --line:#dedee3; --chip:#ffffff; --hot:#ffe08a; --bad:#ffd5d5;
        --accent:#2f6fde; }
@media (prefers-color-scheme: dark) { :root { --bg:#16161a; --fg:#ececf0; --muted:#9a9aa3; --line:#34343c;
        --chip:#202027; --hot:#6b5a1a; --bad:#5a2a2a; --accent:#7aa7ff; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:980px;
       padding:16px; }
a { color:var(--accent); } table { border-collapse:collapse; width:100%; }
td, th { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }
.muted { color:var(--muted); } .chips { line-height:2.1; margin:12px 0; }
.chip { background:var(--chip); border:1px solid var(--line); border-radius:6px; cursor:pointer; padding:2px 5px; }
.chip.now { background:var(--hot); } .chip.unsure { border-color:#d08a00; border-style:dashed; }
textarea { box-sizing:border-box; width:100%; min-height:220px; font:16px/1.6 ui-monospace, monospace;
           background:var(--chip); color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:10px; }
input[type=text] { width:100%; box-sizing:border-box; padding:6px; background:var(--chip); color:var(--fg);
                   border:1px solid var(--line); border-radius:6px; }
audio { width:100%; } .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:10px 0; }
button { padding:7px 14px; border-radius:8px; border:1px solid var(--line); background:var(--accent); color:#fff;
         cursor:pointer; } #msg { font-weight:600; }
"""


def _page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_STYLE}</style>"
            f"</head><body>{body}</body></html>").encode("utf-8")


def _index(root: Path) -> bytes:
    cases = case_store.list_cases(root)
    rows = "".join(
        f"<tr><td><a href='/case/{html.escape(c.case_id)}'>{html.escape(c.case_id)}</a></td>"
        f"<td>{c.gold_status()}</td><td>{c.duration:.1f} s</td></tr>" for c in cases)
    done = sum(1 for c in cases if c.gold_status() == "ready")
    body = (f"<h1>Caption gold</h1><p class='muted'>{done}/{len(cases)} finished. Listen, correct the text word for "
            "word, tick Finished, Save.</p><table><tr><th>case</th><th>gold</th><th>length</th></tr>"
            f"{rows}</table>")
    return _page("Caption gold", body)


def _case_page(root: Path, case: case_store.Case) -> bytes:
    cases = [c.case_id for c in case_store.list_cases(root)]
    index = cases.index(case.case_id)
    prev_link = f"<a href='/case/{html.escape(cases[index - 1])}'>&larr; previous</a>" if index > 0 else ""
    next_link = f"<a href='/case/{html.escape(cases[index + 1])}'>next &rarr;</a>" if index + 1 < len(cases) else ""
    words = case.published_words() or []
    chips = "".join(
        f"<span class='chip{' unsure' if w.get('status') == 'uncertain' or w.get('uncertain') else ''}' "
        f"data-s='{float(w.get('start') or 0):.3f}' data-e='{float(w.get('end') or 0):.3f}'>"
        f"{html.escape(str(w.get('word', '')))}</span> " for w in words if w.get("start") is not None)
    body_text = metrics.gold_body(case.gold_text()) if case.gold_path.is_file() else ""
    status = case.gold_status()
    body = f"""
<div class='row'><a href='/'>all cases</a> {prev_link} {next_link}</div>
<h2>{html.escape(case.case_id)}</h2>
<p class='muted'>{case.duration:.1f} s &middot; language {html.escape(case.language)} &middot; gold: <b id='st'>{status}</b>
&middot; dashed chips = words MIMIR itself was unsure about</p>
<audio id='a' controls preload='auto' src='/audio/{html.escape(case.case_id)}'></audio>
<div class='row'>speed <button type='button' onclick='rate(0.75)'>0.75x</button>
<button type='button' onclick='rate(1)'>1x</button>
<span class='muted'>Ctrl+Space play/pause &middot; Ctrl+&larr; back 3 s &middot; Ctrl+S save</span></div>
<div class='chips' id='chips'>{chips or "<span class='muted'>(no captioned words with times for this case)</span>"}</div>
<p class='muted'>Write exactly what is said. Fillers (uh/um), case and punctuation do not matter. Spell names
correctly. Write [inaudible] where even you cannot tell.</p>
<textarea id='t' spellcheck='false'>{html.escape(body_text)}</textarea>
<p>Verified names said in this short (comma separated)</p>
<input type='text' id='n' value='{html.escape(", ".join(case.names))}'>
<div class='row'><label><input type='checkbox' id='d' {'checked' if status == 'ready' else ''}> Finished
(this gold is checked word for word)</label> <button type='button' onclick='save()'>Save</button>
<span id='msg'></span></div>
<script>
const a = document.getElementById('a'), t = document.getElementById('t');
const chips = [...document.querySelectorAll('.chip')];
function rate(r) {{ a.playbackRate = r; }}
chips.forEach(c => c.addEventListener('click', () => {{ a.currentTime = Math.max(0, +c.dataset.s - 0.3); a.play(); }}));
a.addEventListener('timeupdate', () => {{
  for (const c of chips) c.classList.toggle('now', a.currentTime >= +c.dataset.s && a.currentTime < +c.dataset.e);
}});
async function save() {{
  const r = await fetch('/save/{html.escape(case.case_id)}', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{gold:t.value, names:document.getElementById('n').value,
                           done:document.getElementById('d').checked}})}});
  const j = await r.json();
  document.getElementById('msg').textContent = r.ok ? 'saved' : ('error: ' + j.error);
  if (r.ok) document.getElementById('st').textContent = j.status;
}}
document.addEventListener('keydown', e => {{
  if (e.ctrlKey && e.code === 'Space') {{ e.preventDefault(); a.paused ? a.play() : a.pause(); }}
  if (e.ctrlKey && e.key === 'ArrowLeft') {{ e.preventDefault(); a.currentTime = Math.max(0, a.currentTime - 3); }}
  if (e.ctrlKey && (e.key === 's' || e.key === 'S')) {{ e.preventDefault(); save(); }}
}});
</script>"""
    return _page(case.case_id, body)


def save_gold(case: case_store.Case, gold: str, *, names: str, done: bool) -> str:
    """Write gold.txt (TODO line kept until ``done``) and the verified names; returns the new gold status."""
    body = "\n".join(line.rstrip() for line in str(gold).replace("\r\n", "\n").splitlines()
                     if not line.lstrip().startswith("#")).strip()
    header = case_store.GOLD_HEADER if not done else "".join(
        line + "\n" for line in case_store.GOLD_HEADER.splitlines() if not line.startswith(case_store.GOLD_TODO))
    temp = case.gold_path.with_name("gold.txt.tmp")
    temp.write_text(header + body + "\n", encoding="utf-8")
    temp.replace(case.gold_path)
    meta = dict(case.meta)
    meta["verified_names"] = [x.strip() for x in str(names).split(",") if x.strip()]
    case_store._write_json(case.directory / "meta.json", meta)
    case.meta = meta
    return case.gold_status()


class _Handler(BaseHTTPRequestHandler):
    root: Path = case_store.DEFAULT_ROOT

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet server
        return

    def _case(self, raw: str) -> case_store.Case | None:
        case_id = unquote(raw)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
            return None
        found = case_store.list_cases(self.root, [case_id])
        return found[0] if found else None

    def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8",
              headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(HTTPStatus.OK, _index(self.root))
            return
        kind, _, raw = path.lstrip("/").partition("/")
        case = self._case(raw) if raw else None
        if case is None:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        if kind == "case":
            self._send(HTTPStatus.OK, _case_page(self.root, case))
        elif kind == "audio":
            self._audio(case.audio_path)
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _audio(self, path: Path) -> None:
        data = path.read_bytes()
        size = len(data)
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not match:                          # browsers seek in audio only when ranges are served
            self._send(HTTPStatus.OK, data, "audio/wav", {"Accept-Ranges": "bytes"})
            return
        start = int(match.group(1) or 0)
        end = min(size - 1, int(match.group(2)) if match.group(2) else size - 1)
        if start > end:
            self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "audio/wav",
                       {"Content-Range": f"bytes */{size}"})
            return
        self._send(HTTPStatus.PARTIAL_CONTENT, data[start:end + 1], "audio/wav",
                   {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        kind, _, raw = self.path.lstrip("/").partition("/")
        case = self._case(raw) if kind == "save" else None
        if case is None:
            self._send(HTTPStatus.NOT_FOUND, b'{"error":"no such case"}', "application/json")
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 2_000_000)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            status = save_gold(case, str(payload.get("gold", "")), names=str(payload.get("names", "")),
                               done=bool(payload.get("done")))
        except (ValueError, OSError) as error:
            self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(error)[:200]}).encode(), "application/json")
            return
        self._send(HTTPStatus.OK, json.dumps({"status": status}).encode(), "application/json")


def serve(root: str | Path | None = None, *, port: int = 8765, open_browser: bool = True) -> None:
    handler = type("Handler", (_Handler,), {"root": case_store.eval_root(root)})
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Caption gold editor: {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
