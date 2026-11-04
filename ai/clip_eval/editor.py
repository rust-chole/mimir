"""A small local web page for labeling clip moments (``python -m ai.clip_eval serve``).

Standard library only, bound to 127.0.0.1, same look as the caption gold editor. Per video: the source
video (or its ``prepare --proxy`` copy) with the timestamped transcript beside it in 30-second windows
(click a line to jump there), In / Out marks, a grade (great / ok / bad), a short note, the moments and
never-pick ranges so far, and Save. "Finished" makes the labels count; unfinished ones are never scored.

BLIND by construction: the page shows only the video, the transcript and THIS annotator's own labels.
It never reads MIMIR's picks, runs, published shorts, harvested reviews or another annotator's labels.

Keys (outside text fields): Space play/pause, I mark in, O mark out, J / L back / forward 5 s,
Shift+J / Shift+L 30 s, Ctrl+S save.

The same server shows the blind calibration pairs (``/calibrate``): two clips of one video as Clip 1 and
Clip 2 (sides drawn at random, run names never shown) and your verdict.
"""
from __future__ import annotations

import html
import json
import mimetypes
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ai.caption_eval.editor import _STYLE as CAPTION_STYLE
from ai.clip_eval import cases as case_store

_STYLE = CAPTION_STYLE + """
body { max-width:1240px; }
.grid { display:grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap:16px; align-items:start; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
.sticky { position:sticky; top:8px; } video { width:100%; max-height:58vh; background:#000; border-radius:8px; }
.tx { max-height:78vh; overflow:auto; border:1px solid var(--line); border-radius:8px; padding:4px 8px; }
.win { padding:4px 2px; border-bottom:1px solid var(--line); cursor:pointer; }
.win.now { background:var(--hot); } .win .t { color:var(--muted); font:13px ui-monospace, monospace; margin-right:6px; }
.marks b { font:15px ui-monospace, monospace; } select, input.time { padding:5px; background:var(--chip);
  color:var(--fg); border:1px solid var(--line); border-radius:6px; }
input.time { width:92px; font:14px ui-monospace, monospace; } input.note { flex:1; min-width:160px; }
button.alt { background:var(--chip); color:var(--fg); } button.small { padding:3px 8px; }
.row.item { margin:4px 0; } .great { border-left:4px solid #2e9d4f; padding-left:6px; }
.ok { border-left:4px solid #c7a400; padding-left:6px; } .bad { border-left:4px solid #b54545; padding-left:6px; }
.never { border-left:4px solid #7a4cc2; padding-left:6px; } .warn { color:#b07000; } .err { color:#c03030; }
.pair { display:grid; grid-template-columns: 1fr 1fr; gap:16px; } @media (max-width: 900px) { .pair { grid-template-columns: 1fr; } }
pre.clip { white-space:pre-wrap; font:14px/1.5 ui-monospace, monospace; background:var(--chip);
  border:1px solid var(--line); border-radius:8px; padding:8px; max-height:40vh; overflow:auto; }
"""

GUIDE = ("Mark each moment as <b>the Short you would publish</b>: from the first second you would keep "
         "(the setup / hook) to the last (the payoff and the reaction), typically 15-60 s. "
         "<b>great</b> = you would post it; <b>ok</b> = usable, not your first choice; <b>bad</b> = looks like a "
         "highlight but is not (a near miss MIMIR might fall for). <b>Never-pick</b> ranges are parts that must "
         "never become a Short (sponsor reads, private information, dead air, AFK, broken audio). Aim for 2-6 "
         "moments per video.")


def _page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_STYLE}</style>"
            f"</head><body>{body}</body></html>").encode("utf-8")


def _json_script(element_id: str, data: Any) -> str:
    text = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"<script type='application/json' id='{element_id}'>{text}</script>"


def _index(root: Path, annotator: str) -> bytes:
    from ai.clip_eval import calibrate

    cases = case_store.list_cases(root)
    rows = []
    for case in cases:
        labels = case.labels(annotator)
        count = f"{len(labels.moments)} moments, {len(labels.negatives)} never-pick" if labels else "-"
        rows.append(f"<tr><td><a href='/case/{html.escape(case.case_id)}'>{html.escape(case.case_id)}</a></td>"
                    f"<td>{case.split}</td><td>{case_store.clock(case.duration)}</td>"
                    f"<td>{'ok' if case.has_transcript() else 'missing'}</td>"
                    f"<td>{case.label_status(annotator)}</td><td>{count}</td></tr>")
    done = sum(1 for c in cases if c.label_status(annotator) == "ready")
    pairs = calibrate.load_pairs(root)
    labeled = len(calibrate.human_labels(root))
    calib = (f"<p><a href='/calibrate'>Judge calibration</a>: {labeled}/{len(pairs)} pairs answered.</p>"
             if pairs else "")
    body = (f"<h1>Clip labels</h1><p class='muted'>Labeling as <b>{html.escape(annotator)}</b>. {done}/{len(cases)} "
            f"finished. {GUIDE}</p><table><tr><th>video</th><th>split</th><th>length</th><th>transcript</th>"
            f"<th>labels</th><th></th></tr>{''.join(rows)}</table>{calib}")
    return _page("Clip labels", body)


def _case_page(root: Path, case: case_store.Case, annotator: str) -> bytes:
    ids = [c.case_id for c in case_store.list_cases(root)]
    index = ids.index(case.case_id)
    prev_link = f"<a href='/case/{html.escape(ids[index - 1])}'>&larr; previous</a>" if index > 0 else ""
    next_link = f"<a href='/case/{html.escape(ids[index + 1])}'>next &rarr;</a>" if index + 1 < len(ids) else ""
    labels = case.labels(annotator)
    windows = case_store.transcript_windows(case.words(), 30.0, case.duration)
    data = {"case_id": case.case_id, "duration": case.duration,
            "labels": labels.to_dict(case.case_id) if labels else {"moments": [], "negatives": [], "finished": False},
            "windows": [[round(a, 1), round(b, 1), t] for a, b, t in windows]}
    source = "proxy copy" if case.proxy_path else html.escape(case.video_path.name)
    body = f"""
<div class='row'><a href='/'>all videos</a> {prev_link} {next_link}</div>
<h2>{html.escape(case.case_id)}</h2>
<p class='muted'>{case_store.clock(case.duration)} &middot; split {case.split} &middot; {source} &middot; labeling as
<b>{html.escape(annotator)}</b> &middot; labels: <b id='st'>{case.label_status(annotator)}</b></p>
<div class='grid'><div class='sticky'>
<video id='v' controls preload='metadata' src='/video/{html.escape(case.case_id)}'></video>
<div class='row marks'>now <b id='now'>0:00.0</b> &middot; in <b id='mi'>-</b> <button type='button' class='alt small'
 onclick='markIn()'>In (I)</button> out <b id='mo'>-</b> <button type='button' class='alt small'
 onclick='markOut()'>Out (O)</button></div>
<div class='row'><select id='g'><option value='great'>great</option><option value='ok'>ok</option>
<option value='bad'>bad</option></select><input type='text' class='note' id='nt' placeholder='short note: what happens, why it works'>
<button type='button' onclick='add("moment")'>Add moment</button>
<button type='button' class='alt' onclick='add("never")'>Add never-pick</button></div>
<div id='list'></div>
<div class='row'><label><input type='checkbox' id='fin'> Finished (I watched the whole video and these are all
the moments)</label> <button type='button' onclick='save()'>Save (Ctrl+S)</button></div>
<div id='msg'></div>
<p class='muted'>{GUIDE}<br>Keys: Space play/pause &middot; I / O mark in / out &middot; J / L 5 s back / forward
(Shift: 30 s) &middot; click a transcript line to jump.</p>
</div><div><label class='muted'><input type='checkbox' id='follow' checked> follow the video</label>
<div class='tx' id='tx'></div></div></div>
{_json_script('data', data)}
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const v = document.getElementById('v');
let moments = D.labels.moments.map(m => ({{...m}})), negatives = D.labels.negatives.map(m => ({{...m}}));
let mi = null, mo = null, stopAt = null;
document.getElementById('fin').checked = !!D.labels.finished;
function fmt(t) {{ if (t === null || t === undefined || isNaN(t)) return '-'; t = Math.max(0, +t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = (t % 60);
  const ss = s.toFixed(1).padStart(4, '0'); return h ? `${{h}}:${{String(m).padStart(2, '0')}}:${{ss}}` : `${{m}}:${{ss}}`; }}
function parse(x) {{ const p = String(x).trim().replace(',', '.').split(':').map(Number);
  if (!p.length || p.some(isNaN)) return NaN; return p.reduce((a, b) => a * 60 + b, 0); }}
function esc(s) {{ return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function markIn() {{ mi = v.currentTime; document.getElementById('mi').textContent = fmt(mi); }}
function markOut() {{ mo = v.currentTime; document.getElementById('mo').textContent = fmt(mo); }}
function playRange(a, b) {{ v.currentTime = a; stopAt = b; v.play(); }}
function add(kind) {{
  if (mi === null || mo === null || mo <= mi) {{ msg('Mark In before Out first.', 'err'); return; }}
  const note = document.getElementById('nt').value.trim();
  if (kind === 'moment') moments.push({{start: mi, end: mo, grade: document.getElementById('g').value, note}});
  else negatives.push({{start: mi, end: mo, note}});
  mi = mo = null; document.getElementById('mi').textContent = '-'; document.getElementById('mo').textContent = '-';
  document.getElementById('nt').value = ''; render(); msg('added (not saved yet)', 'warn');
}}
function row(item, i, kind) {{
  const cls = kind === 'never' ? 'never' : item.grade;
  const grade = kind === 'never' ? '<b>never-pick</b>' : `<select data-k='${{kind}}' data-i='${{i}}' data-f='grade'>` +
    ['great', 'ok', 'bad'].map(g => `<option ${{g === item.grade ? 'selected' : ''}}>${{g}}</option>`).join('') + '</select>';
  return `<div class='row item ${{cls}}'>${{grade}}
    <input class='time' data-k='${{kind}}' data-i='${{i}}' data-f='start' value='${{fmt(item.start)}}'> -
    <input class='time' data-k='${{kind}}' data-i='${{i}}' data-f='end' value='${{fmt(item.end)}}'>
    <span class='muted'>${{(item.end - item.start).toFixed(1)}} s</span>
    <input class='note' data-k='${{kind}}' data-i='${{i}}' data-f='note' value='${{esc(item.note || '')}}'>
    <button type='button' class='alt small' onclick='playRange(${{item.start}}, ${{item.end}})'>play</button>
    <button type='button' class='alt small' onclick='drop("${{kind}}", ${{i}})'>delete</button></div>`;
}}
function render() {{
  const html = moments.map((m, i) => row(m, i, 'moment')).join('') + negatives.map((m, i) => row(m, i, 'never')).join('');
  const box = document.getElementById('list'); box.innerHTML = html || "<p class='muted'>No moment yet.</p>";
  box.querySelectorAll('[data-f]').forEach(el => el.addEventListener('change', () => {{
    const list = el.dataset.k === 'never' ? negatives : moments, item = list[+el.dataset.i], f = el.dataset.f;
    if (f === 'start' || f === 'end') {{ const t = parse(el.value); if (isNaN(t)) {{ msg('not a time: ' + el.value, 'err'); return; }}
      item[f] = t; }} else item[f] = el.value;
    render(); msg('changed (not saved yet)', 'warn');
  }}));
}}
function drop(kind, i) {{ (kind === 'never' ? negatives : moments).splice(i, 1); render(); msg('deleted (not saved yet)', 'warn'); }}
function msg(text, cls) {{ const m = document.getElementById('msg'); m.className = cls || ''; m.innerHTML = text; }}
async function save() {{
  const r = await fetch('/save/' + encodeURIComponent(D.case_id), {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{moments, negatives, finished: document.getElementById('fin').checked}})}});
  const j = await r.json();
  if (!r.ok) {{ msg('error: ' + esc(j.error || r.status), 'err'); return; }}
  document.getElementById('st').textContent = j.status;
  const lines = [`saved: labels ${{esc(j.status)}}`].concat((j.problems || []).map(p => `<span class='err'>${{esc(p)}}</span>`))
    .concat((j.warnings || []).map(w => `<span class='warn'>check: ${{esc(w)}}</span>`));
  msg(lines.join('<br>'), j.problems && j.problems.length ? 'err' : '');
}}
const tx = document.getElementById('tx');
tx.innerHTML = D.windows.map((w, i) => `<div class='win' data-i='${{i}}' data-s='${{w[0]}}'><span class='t'>${{fmt(w[0])}}</span>${{esc(w[2]) || "<span class='muted'>(no speech)</span>"}}</div>`).join('');
tx.querySelectorAll('.win').forEach(el => el.addEventListener('click', () => {{ v.currentTime = +el.dataset.s; stopAt = null; v.play(); }}));
let lastWin = -1;
v.addEventListener('timeupdate', () => {{
  document.getElementById('now').textContent = fmt(v.currentTime);
  if (stopAt !== null && v.currentTime >= stopAt) {{ v.pause(); stopAt = null; }}
  const i = Math.floor(v.currentTime / 30);
  if (i !== lastWin) {{ const old = tx.querySelector('.win.now'); if (old) old.classList.remove('now');
    const el = tx.querySelector(`.win[data-i='${{i}}']`); if (el) {{ el.classList.add('now');
      if (document.getElementById('follow').checked) el.scrollIntoView({{block: 'center'}}); }} lastWin = i; }}
}});
document.addEventListener('keydown', e => {{
  if (e.ctrlKey && (e.key === 's' || e.key === 'S')) {{ e.preventDefault(); save(); return; }}
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  const k = e.key.toLowerCase();
  if (e.code === 'Space') {{ e.preventDefault(); v.paused ? v.play() : v.pause(); }}
  else if (k === 'i') markIn(); else if (k === 'o') markOut();
  else if (k === 'j') v.currentTime = Math.max(0, v.currentTime - (e.shiftKey ? 30 : 5));
  else if (k === 'l') v.currentTime = v.currentTime + (e.shiftKey ? 30 : 5);
}});
window.addEventListener('beforeunload', e => {{ if (document.getElementById('msg').className === 'warn') {{ e.preventDefault(); e.returnValue = ''; }} }});
render();
</script>"""
    return _page(case.case_id, body)


# ============================================================
# CALIBRATION PAGES (blind)
# ============================================================

def _calibration_index(root: Path) -> bytes:
    from ai.clip_eval import calibrate

    pairs = calibrate.load_pairs(root)
    done = calibrate.human_labels(root)
    rows = "".join(f"<tr><td><a href='/pair/{p['id']}'>pair {i}</a></td><td>{html.escape(p['case_id'])}</td>"
                   f"<td>{'answered' if p['id'] in done else '-'}</td></tr>" for i, p in enumerate(pairs, start=1))
    body = (f"<div class='row'><a href='/'>all videos</a></div><h1>Judge calibration</h1><p class='muted'>"
            f"{len(done)}/{len(pairs)} answered (aim for {calibrate.TARGET_PAIRS}). Each pair shows two clips of one "
            "video; which side came from which run is hidden.</p>"
            f"<table><tr><th>pair</th><th>video</th><th>you</th></tr>{rows}</table>")
    return _page("Judge calibration", body)


def _pair_page(root: Path, pair: dict[str, Any]) -> bytes:
    from ai.clip_eval import calibrate, judge

    case = case_store.list_cases(root, [pair["case_id"]])[0]
    one, two = calibrate.shown(pair)
    answer = calibrate.human_labels(root).get(pair["id"], "")
    pairs = calibrate.load_pairs(root)
    position = [p["id"] for p in pairs].index(pair["id"])
    nxt = pairs[position + 1]["id"] if position + 1 < len(pairs) else ""
    blocks = []
    for number, top in ((1, one), (2, two)):
        view = judge.clip_view(case, top)
        blocks.append(f"<div><h3>Clip {number} <span class='muted'>({view['duration_s']} s)</span></h3>"
                      f"<video id='v{number}' controls preload='metadata' src='/video/{html.escape(case.case_id)}'></video>"
                      f"<div class='row'><button type='button' class='alt' onclick='play({number}, {top[0]}, {top[1]})'>"
                      f"Play clip {number}</button></div><pre class='clip'>{html.escape(view['transcript'])}</pre></div>")
    body = f"""
<div class='row'><a href='/calibrate'>all pairs</a></div>
<h2>Pair {position + 1} of {len(pairs)}</h2>
<p class='muted'>Watch both clips. Which one makes the better Short on its own? Check: a hook in the first 3 s,
it makes sense without the stream, the payoff is inside the clip, it does not cut mid-sentence.</p>
<div class='pair'>{''.join(blocks)}</div>
<div class='row'><button type='button' onclick='answer("1")'>Clip 1 is better</button>
<button type='button' onclick='answer("2")'>Clip 2 is better</button>
<button type='button' class='alt' onclick='answer("tie")'>Tie</button> <span id='msg'>{('your answer: ' + answer) if answer else ''}</span></div>
<script>
const stops = {{}};
function play(n, a, b) {{ const v = document.getElementById('v' + n); v.currentTime = a; stops[n] = b; v.play(); }}
[1, 2].forEach(n => document.getElementById('v' + n).addEventListener('timeupdate', e => {{
  if (stops[n] !== undefined && e.target.currentTime >= stops[n]) {{ e.target.pause(); delete stops[n]; }} }}));
async function answer(a) {{
  const r = await fetch('/answer/{pair['id']}', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{answer: a}})}});
  if (r.ok) {{ {"location.href = '/pair/" + nxt + "';" if nxt else "location.href = '/calibrate';"} }}
  else document.getElementById('msg').textContent = 'error';
}}
</script>"""
    return _page("Judge calibration", body)


# ============================================================
# SERVER
# ============================================================

class _Handler(BaseHTTPRequestHandler):
    root: Path = case_store.DEFAULT_ROOT
    annotator: str = case_store.DEFAULT_ANNOTATOR

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet server
        return

    def _case(self, raw: str) -> case_store.Case | None:
        case_id = unquote(raw)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
            return None
        found = case_store.list_cases(self.root, [case_id])
        return found[0] if found else None

    def _pair(self, raw: str) -> dict[str, Any] | None:
        from ai.clip_eval import calibrate

        pair_id = unquote(raw)
        for pair in calibrate.load_pairs(self.root):
            if pair["id"] == pair_id:
                dev = {c.case_id for c in case_store.list_cases(self.root, split="dev")}
                return pair if pair["case_id"] in dev else None
        return None

    def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8",
              headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(HTTPStatus.OK, _index(self.root, self.annotator))
            return
        if path == "/calibrate":
            self._send(HTTPStatus.OK, _calibration_index(self.root))
            return
        kind, _, raw = path.lstrip("/").partition("/")
        if kind == "pair":
            pair = self._pair(raw)
            if pair is None:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            else:
                self._send(HTTPStatus.OK, _pair_page(self.root, pair))
            return
        case = self._case(raw) if raw else None
        if case is None:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        elif kind == "case":
            self._send(HTTPStatus.OK, _case_page(self.root, case, self.annotator))
        elif kind == "video":
            self._video(case.proxy_path or case.video_path)
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _video(self, path: Path) -> None:
        """Byte ranges streamed from disk (a VOD is far too large to read into memory)."""
        if not path.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"video not found on this machine", "text/plain")
            return
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".mkv":
            content_type = "video/webm"                       # Chromium plays most MKV (VP9/H.264) as WebM
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = min(size - 1, int(match.group(2))) if match.group(2) else size - 1
            else:                                              # suffix range: the last N bytes
                start, end = max(0, size - int(match.group(2))), size - 1
            if start > end or start >= size:
                self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", content_type,
                           {"Content-Range": f"bytes */{size}"})
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            start, end = 0, size - 1
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return                                            # the browser seeks by dropping connections

    def _body(self) -> dict[str, Any]:
        length = min(int(self.headers.get("Content-Length", "0") or 0), 4_000_000)
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        from ai.clip_eval import calibrate

        kind, _, raw = self.path.lstrip("/").partition("/")
        try:
            if kind == "save":
                case = self._case(raw)
                if case is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "no such case"})
                    return
                status, problems, warnings = case_store.save_labels(case, self._body(), annotator=self.annotator)
                self._json(HTTPStatus.OK if not problems else HTTPStatus.UNPROCESSABLE_ENTITY,
                           {"status": status, "problems": problems, "warnings": warnings,
                            **({"error": "; ".join(problems)} if problems else {})})
            elif kind == "answer":
                pair = self._pair(raw)
                if pair is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "no such pair"})
                    return
                calibrate.save_human(pair["id"], str(self._body().get("answer", "")), root=self.root)
                self._json(HTTPStatus.OK, {"status": "saved"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, KeyError, OSError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)[:300]})


def serve(root: str | Path | None = None, *, port: int = 8766, open_browser: bool = True,
          annotator: str | None = None, path: str = "/") -> None:
    handler = type("Handler", (_Handler,), {"root": case_store.eval_root(root),
                                            "annotator": case_store.clean_annotator(annotator)
                                            if annotator else case_store.default_annotator()})
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), handler)
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    print(f"Clip label editor: {url}  (labeling as {handler.annotator}; Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
