"""Capture the real running dashboard with installed Chrome and Tornado.

Start Streamlit first, then run:
    python scripts/capture_dashboard.py --url http://127.0.0.1:8501
No synthetic or drawn dashboard mockups are used.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from tornado.httpclient import AsyncHTTPClient
from tornado.websocket import websocket_connect


async def capture(url, output, port):
    browser = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not browser:
        raise RuntimeError("Install Google Chrome or Chromium to capture the running dashboard.")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="readmitrisk-browser-", ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen([
            browser, "--headless", "--no-sandbox", "--disable-gpu", "--disable-background-networking",
            "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
            f"--user-data-dir={profile}", "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={port}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        socket = None
        try:
            client = AsyncHTTPClient()
            for _ in range(40):
                try:
                    response = await client.fetch(f"http://127.0.0.1:{port}/json")
                    pages = json.loads(response.body)
                    page = next(item for item in pages if item.get("type") == "page")
                    break
                except (OSError, StopIteration):
                    await asyncio.sleep(.25)
            else:
                raise RuntimeError("Chrome did not expose a debugging endpoint.")
            socket = await websocket_connect(page["webSocketDebuggerUrl"], max_message_size=32 * 1024 * 1024)
            message_id = 0

            async def command(method, params=None):
                nonlocal message_id
                message_id += 1
                await socket.write_message(json.dumps({"id": message_id, "method": method, "params": params or {}}))
                while True:
                    message = await socket.read_message()
                    if message is None:
                        raise RuntimeError("Chrome debugging connection closed.")
                    result = json.loads(message)
                    if result.get("id") == message_id:
                        if "error" in result:
                            raise RuntimeError(str(result["error"]))
                        return result.get("result", {})

            async def evaluate(expression):
                result = await command("Runtime.evaluate", {"expression": expression, "returnByValue": True})
                return result.get("result", {}).get("value")

            async def wait_for(expression, seconds=50):
                for _ in range(int(seconds * 2)):
                    if await evaluate(expression):
                        return
                    await asyncio.sleep(.5)
                raise TimeoutError(f"Dashboard did not become ready: {expression}")

            async def screenshot(name):
                await asyncio.sleep(1.5)
                result = await command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                path = output / name
                path.write_bytes(base64.b64decode(result["data"]))
                print(f"Saved {path}", flush=True)

            await command("Page.enable")
            await command("Runtime.enable")
            await command("Emulation.setDeviceMetricsOverride", {"width": 1600, "height": 1120, "deviceScaleFactor": 1, "mobile": False})
            await command("Page.navigate", {"url": url})
            await wait_for("document.body.innerText.includes('Explore the readmission picture') && document.querySelectorAll('[data-testid=stMetric]').length >= 4")
            await wait_for("!document.body.innerText.includes('RUNNING...')")
            await screenshot("dashboard_eda.png")
            shutil.copyfile(output / "dashboard_eda.png", output / "dashboard.png")
            await evaluate("document.querySelectorAll('[role=tab]')[1].click()")
            await asyncio.sleep(.5)
            scored = await evaluate("Array.from(document.querySelectorAll('button')).some(e => e.innerText.includes('Calculate readmission risk'))")
            if scored:
                await evaluate("Array.from(document.querySelectorAll('button')).find(e => e.innerText.includes('Calculate readmission risk')).click()")
                await wait_for("document.body.innerText.includes('Top 3 individual SHAP drivers') || document.body.innerText.includes('Contributions explain the underlying')")
                await evaluate("document.querySelector('[data-testid=stMain]').scrollTo(0, 630)")
                await screenshot("dashboard_risk.png")
            await evaluate("document.querySelectorAll('[role=tab]')[2].click(); document.querySelector('[data-testid=stMain]').scrollTo(0, 0)")
            await screenshot("dashboard_budget.png")
            await evaluate("document.querySelectorAll('[role=tab]')[3].click(); document.querySelector('[data-testid=stMain]').scrollTo(0, 0)")
            await screenshot("dashboard_tiers.png")
        finally:
            if socket is not None:
                socket.close()
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--port", type=int, default=9223)
    args = parser.parse_args()
    asyncio.run(capture(args.url, args.output, args.port))
