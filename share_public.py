"""
Start the Streamlit dashboard and expose it publicly with a fixed ngrok domain.

Expected domain:
  https://healt-prediction.com.ngrok-free.app

Required local-only environment variables:
  NGROK_AUTHTOKEN  -> set in tunnel_secrets.bat, never commit it to GitHub
  NGROK_DOMAIN     -> set by run_public_ngrok.bat
Optional:
  DASHBOARD_PASSWORD -> simple password gate already supported by app.py
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

PORT = 8501
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOMAIN = "healt-prediction.com.ngrok-free.app"


def find_ngrok():
    path = shutil.which("ngrok")
    if path:
        return path

    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe"),
        os.path.expandvars(r"%ProgramFiles%\ngrok\ngrok.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\ngrok\ngrok.exe"),
        os.path.join(PROJECT_DIR, "ngrok.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def free_port_windows(port):
    if os.name != "nt":
        return
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, check=False)
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.split()[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, check=False)
        time.sleep(2)
    except Exception:
        pass


def check_streamlit_installed():
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "--version"], check=True, capture_output=True, text=True)
        return True
    except Exception:
        return False


def start_streamlit():
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PUBLIC_APP_URL"] = f"https://{os.getenv('NGROK_DOMAIN', DEFAULT_DOMAIN)}"

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            f"--server.port={PORT}",
            "--server.address=0.0.0.0",
        ],
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stream_process_output(proc, title):
    def reader():
        if not proc.stdout:
            return
        for line in proc.stdout:
            print(f"[{title}] {line}", end="", flush=True)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return thread


def wait_for_streamlit(proc, timeout=60):
    import urllib.request

    url = f"http://127.0.0.1:{PORT}"
    for _ in range(timeout):
        if proc.poll() is not None:
            print("ERROR: Streamlit stopped before the dashboard opened.")
            return False
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(1)
    print("ERROR: Streamlit did not respond in time.")
    return False


def configure_ngrok(ngrok_path):
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not token:
        print("ERROR: NGROK_AUTHTOKEN is missing.")
        print("Open tunnel_secrets.bat and set NGROK_AUTHTOKEN to your token.")
        return False

    result = subprocess.run(
        [ngrok_path, "config", "add-authtoken", token],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("ERROR: ngrok could not save your authtoken.")
        print(result.stdout.strip())
        print(result.stderr.strip())
        return False
    return True


def start_ngrok(ngrok_path):
    domain = os.getenv("NGROK_DOMAIN", DEFAULT_DOMAIN).strip().replace("https://", "").replace("http://", "")
    public_url = f"https://{domain}"

    print(f"Starting ngrok tunnel: {public_url} -> http://localhost:{PORT}")

    proc = subprocess.Popen(
        [ngrok_path, "http", f"--domain={domain}", str(PORT)],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    detected = {"ready": False}
    url_pattern = re.compile(r"https://[^\s]+")

    def reader():
        if not proc.stdout:
            return
        for line in proc.stdout:
            print(f"[ngrok] {line}", end="", flush=True)
            if domain in line or url_pattern.search(line):
                detected["ready"] = True

    threading.Thread(target=reader, daemon=True).start()

    for _ in range(20):
        if proc.poll() is not None:
            print("ERROR: ngrok stopped before creating the tunnel.")
            return None, None
        if detected["ready"]:
            return public_url, proc
        time.sleep(0.5)

    if proc.poll() is None:
        return public_url, proc
    return None, None


def save_public_url(public_url):
    url_file = os.path.join(PROJECT_DIR, "public_url.txt")
    with open(url_file, "w", encoding="utf-8") as f:
        f.write(public_url)

    print()
    print("=" * 56)
    print("  Health Prediction — PUBLIC LINK")
    print(f"  {public_url}")
    print("=" * 56)
    print("  Keep this terminal open. Closing it stops Streamlit/ngrok.")
    print()

    try:
        webbrowser.open(public_url)
    except Exception:
        pass

    if os.name == "nt":
        try:
            subprocess.run(
                ["powershell", "-Command", f"Set-Clipboard -Value '{public_url}'"],
                check=False,
                capture_output=True,
            )
            print("  Link copied to clipboard.")
        except Exception:
            pass

    return url_file


def pause_before_exit():
    try:
        input("\nPress Enter to close...")
    except EOFError:
        pass


def main():
    print()
    print("=" * 56)
    print("  Health Prediction — ngrok public launcher")
    print("=" * 56)
    print(f"  Python: {sys.executable}")
    print()

    if not check_streamlit_installed():
        print("ERROR: Streamlit is not installed in this Python environment.")
        print("Run: pip install -r requirements.txt")
        pause_before_exit()
        sys.exit(1)

    ngrok_path = find_ngrok()
    if not ngrok_path:
        print("ERROR: ngrok was not found.")
        print("Install it with: winget install ngrok.ngrok")
        print("Then close and reopen Command Prompt.")
        pause_before_exit()
        sys.exit(1)

    if not configure_ngrok(ngrok_path):
        pause_before_exit()
        sys.exit(1)

    free_port_windows(PORT)

    url_file = os.path.join(PROJECT_DIR, "public_url.txt")
    if os.path.exists(url_file):
        os.remove(url_file)

    streamlit_proc = start_streamlit()
    stream_process_output(streamlit_proc, "streamlit")
    print("Starting dashboard...")

    if not wait_for_streamlit(streamlit_proc):
        streamlit_proc.terminate()
        pause_before_exit()
        sys.exit(1)

    print(f"Local dashboard: http://localhost:{PORT}")
    print("Creating public ngrok tunnel...")
    print()

    public_url, ngrok_proc = start_ngrok(ngrok_path)
    if not public_url:
        streamlit_proc.terminate()
        pause_before_exit()
        sys.exit(1)

    save_public_url(public_url)

    try:
        while True:
            if streamlit_proc.poll() is not None:
                print("Streamlit stopped.")
                break
            if ngrok_proc and ngrok_proc.poll() is not None:
                print("ngrok stopped.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if streamlit_proc:
            streamlit_proc.terminate()
        if ngrok_proc:
            ngrok_proc.terminate()
        print("\nStopped.")


if __name__ == "__main__":
    main()
