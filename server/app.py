"""
Main entry point for the ADCTM OpenEnv Server.
Exposes the mandatory OpenEnv REST API: /reset, /step, and /state.
"""

import os
from fastapi import FastAPI
from core.env import app

def main() -> None:
    """
    Main entrypoint for local execution and Docker.
    """
    import uvicorn
    # Hugging Face Spaces default port is 7860
    port = int(os.getenv("PORT", 7860))
    # Note: app must be referenced by its string package path for uvicorn workers
    # but here we run directly through the app object.
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
