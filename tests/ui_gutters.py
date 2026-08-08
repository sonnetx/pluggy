"""
layout regression: the content column must be centred, and the topbar must
sit on the same left/right edges as the cards below it.

this is the bug this test exists for: `.page` is a max-width child of a
column flex container, so without an auto inline margin it hugs the LEFT
edge and every pixel of slack piles up on the right -- a visibly lopsided
page that no amount of per-card padding explains.

measured in real headless chrome via the devtools protocol (no puppeteer:
chrome speaks CDP over a websocket, and we just evaluate
getBoundingClientRect). the pages are served by pluggy's own http server
rather than opened as file:// urls -- they link the stylesheet as
"/theme.css", which under file:// resolves to the filesystem root and
silently loads nothing, so a file:// run measures an UNSTYLED page and
passes no matter how broken the css is.

skipped with a notice when chrome or websocket-client is missing, so ci
stays green on runners without them.

uv run tests/ui_gutters.py
"""

import itertools
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from http.server import ThreadingHTTPServer
from pathlib import Path

from pluggy.synth import server as synth_server

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

# (viewport width, id of an element that must be centred in .content)
VIEWPORTS = (1728, 1440, 1280, 900)
TOLERANCE = 1.0  # px; sub-pixel rounding is fine, a 100px lopsided gap is not

MEASURE_JS = """
(() => {
  const content = document.querySelector('.content');
  const page = document.querySelector('.page');
  const topbar = document.querySelector('.topbar');
  const c = content.getBoundingClientRect();
  const p = page.getBoundingClientRect();
  const crumb = document.querySelector('.crumb').getBoundingClientRect();
  const right = document.querySelector('.topbar-right').getBoundingClientRect();
  const cards = document.querySelector('.cards').getBoundingClientRect();
  const cs = getComputedStyle(page);
  // the page's own padding is part of the gutter, so compare content edges
  const pageLeft = p.left + parseFloat(cs.paddingLeft);
  const pageRight = p.right - parseFloat(cs.paddingRight);
  return JSON.stringify({
    sidebarW: document.querySelector('.sidebar').getBoundingClientRect().width,
    contentLeft: c.left, contentRight: c.right,
    gapLeft: pageLeft - c.left, gapRight: c.right - pageRight,
    cardsLeft: cards.left, cardsRight: cards.right,
    crumbLeft: crumb.left, topbarRight: right.right,
    scrollW: document.documentElement.scrollWidth,
    clientW: document.documentElement.clientWidth,
  });
})()
"""


def find_chrome():
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    return shutil.which("google-chrome") or shutil.which("chromium")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Chrome:
    """the smallest possible CDP client: one websocket, one method."""

    def __init__(self, binary, profile):
        self.port = free_port()
        self.proc = subprocess.Popen(
            [binary, "--headless", f"--remote-debugging-port={self.port}",
             # chrome rejects CDP websockets from an http origin unless told
             # otherwise; the port is loopback-only and dies with this process
             "--remote-allow-origins=*",
             f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
             "--disable-gpu", "--hide-scrollbars", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.ws_url = self._wait_for_target()

    def _wait_for_target(self):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json") as r:
                    for t in json.load(r):
                        if t["type"] == "page":
                            return t["webSocketDebuggerUrl"]
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError("chrome devtools never came up")

    def measure(self, url, width, height=900):
        from websocket import create_connection  # noqa: PLC0415

        ws = create_connection(self.ws_url, timeout=30)
        seq = itertools.count(1)
        try:
            def call(method, params):
                msg_id = next(seq)
                ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
                while True:
                    msg = json.loads(ws.recv())
                    # cdp interleaves unsolicited events with replies
                    if msg.get("id") == msg_id:
                        return msg
            call("Emulation.setDeviceMetricsOverride",
                 {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
            call("Page.enable", {})
            call("Page.navigate", {"url": url})
            time.sleep(0.6)  # no lifecycle plumbing: these are static pages
            res = call("Runtime.evaluate", {"expression": MEASURE_JS, "returnByValue": True})
            return json.loads(res["result"]["result"]["value"])
        finally:
            ws.close()

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)


def check(page_name, m, width):
    problems = []
    if abs(m["gapLeft"] - m["gapRight"]) > TOLERANCE:
        problems.append(
            f"content column is lopsided: {m['gapLeft']:.0f}px of gutter on the "
            f"left, {m['gapRight']:.0f}px on the right"
        )
    # the topbar is full-bleed but its contents must share the cards' edges
    if abs(m["crumbLeft"] - m["cardsLeft"]) > TOLERANCE:
        problems.append(
            f"topbar crumb starts at {m['crumbLeft']:.0f}px but the cards start "
            f"at {m['cardsLeft']:.0f}px"
        )
    if abs(m["topbarRight"] - m["cardsRight"]) > TOLERANCE:
        problems.append(
            f"topbar controls end at {m['topbarRight']:.0f}px but the cards end "
            f"at {m['cardsRight']:.0f}px"
        )
    if m["scrollW"] > m["clientW"] + TOLERANCE:
        problems.append(f"page scrolls horizontally ({m['scrollW']} > {m['clientW']})")
    assert not problems, f"{page_name} @ {width}px:\n  " + "\n  ".join(problems)


def main():
    binary = find_chrome()
    if binary is None:
        print("ui_gutters: no chrome/chromium installed, skipping")
        return
    try:
        import websocket  # noqa: F401
    except ImportError:
        print("ui_gutters: `websocket-client` not installed, skipping "
              "(uv pip install -e '.[dev]')")
        return

    # serve the real pages: "/theme.css" only resolves over http
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), synth_server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    profile = tempfile.mkdtemp(prefix="pluggy_ui_test_")
    chrome = Chrome(binary, profile)
    try:
        for page in ("/", "/train"):
            for width in VIEWPORTS:
                m = chrome.measure(base + page, width)
                # guard against measuring an unstyled page: the shell only
                # exists once theme.css applied
                assert m["sidebarW"] > 0, (
                    f"{page}: stylesheet did not load (sidebar has no width) -- "
                    f"the measurement below would be meaningless"
                )
                check(page, m, width)
            print(f"{page}: gutters equal at {', '.join(str(w) for w in VIEWPORTS)}px")
    finally:
        chrome.close()
        httpd.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
    print("all ui gutter tests passed")


if __name__ == "__main__":
    sys.exit(main())
