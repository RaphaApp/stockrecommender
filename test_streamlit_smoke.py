"""Real headless Streamlit smoke test.

The other test files read the SOURCE. This one actually starts the app, which is
the only way to catch failures that live in execution rather than in text:
import-time errors, Streamlit API misuse, a bad set_page_config, or a widget
that raises before the first frame renders.

Run:  python test_streamlit_smoke.py
Exit: 0 = the server started and answered its health endpoint, 1 = it did not
      (captured stdout/stderr is printed so the failure is diagnosable).
"""
import contextlib
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

APP = pathlib.Path(__file__).with_name("real_estate_app.py")
BOOT_TIMEOUT_S = 90          # cold Streamlit start can be slow on first import
POLL_INTERVAL_S = 0.5


def free_port() -> int:
    """Ask the OS for an unused port rather than guessing one."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(port: int, proc: subprocess.Popen, timeout: float) -> tuple[bool, str]:
    """Poll Streamlit's health endpoint until it answers or the process dies."""
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"process exited early with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip()
                if resp.status == 200:
                    return True, body
                last = f"HTTP {resp.status}"
        except urllib.error.URLError as e:
            last = str(e.reason)
        except Exception as e:                      # connection refused while booting
            last = str(e)
        time.sleep(POLL_INTERVAL_S)
    return False, f"timed out after {timeout}s (last: {last})"


def main() -> int:
    if not APP.exists():
        print(f"SMOKE: FAILED — {APP.name} not found")
        return 1

    port = free_port()
    workdir = tempfile.mkdtemp(prefix="st_smoke_")
    env = {
        **os.environ,
        # Keep the smoke run fully isolated: its own throwaway SQLite file, no
        # telemetry, no browser, no interference with the developer's data.
        "DB_PATH": os.path.join(workdir, "smoke.db"),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
    }
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP),
           "--server.port", str(port),
           "--server.headless", "true",
           "--server.address", "127.0.0.1",
           "--browser.gatherUsageStats", "false"]

    print(f"SMOKE: starting {APP.name} on port {port}")
    proc = subprocess.Popen(cmd, cwd=str(APP.parent), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        ok, detail = wait_for_health(port, proc, BOOT_TIMEOUT_S)
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()

    if ok:
        print(f"SMOKE: clean — health endpoint responded ({detail!r})")
        # Streamlit logs real problems as warnings even when it serves, so a
        # successful boot with tracebacks in the log is still worth surfacing.
        tracebacks = [ln for ln in (out or "").splitlines() if "Traceback" in ln]
        if tracebacks:
            print("  NOTE: tracebacks appeared in the log during startup:")
            for ln in tracebacks[:5]:
                print("   ", ln)
        return 0

    print(f"SMOKE: FAILED — {detail}")
    print("--- captured output " + "-" * 40)
    print((out or "(no output captured)")[-4000:])
    print("-" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
