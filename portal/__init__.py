"""
sundog.portal
=============
Developer-facing web portal for deploying and adjusting sundog runner jobs.

Start the portal:

    python -m sundog.portal                        # defaults: host=127.0.0.1, port=7860
    python -m sundog.portal --port 8080 --out /tmp/runs

Then open http://127.0.0.1:7860 in your browser.
"""
from sundog.portal.server import start_server, PortalConfig

__all__ = ["start_server", "PortalConfig"]
