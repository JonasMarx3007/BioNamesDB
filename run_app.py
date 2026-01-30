import os
import sys
import subprocess
from pathlib import Path

def main():
    here = Path(__file__).resolve().parent
    app_path = here / "app.py"

    if not app_path.exists():
        raise FileNotFoundError(f"Could not find {app_path}")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.maxUploadSize=10240",
    ]

    subprocess.run(cmd, cwd=str(here), check=True)

if __name__ == "__main__":
    main()
