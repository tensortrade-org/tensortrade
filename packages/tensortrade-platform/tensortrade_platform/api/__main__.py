"""Allow running: python -m tensortrade_platform.api.server"""

from tensortrade_platform.api.server import create_app

if __name__ == "__main__":
    import os
    from pathlib import Path

    import uvicorn

    # Load .env from project root if present
    env_file = Path(__file__).resolve().parents[4] / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
