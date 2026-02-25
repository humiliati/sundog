"""Entry point: python -m sundog.portal"""
import argparse
import sys
from sundog.portal.server import start_server

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m sundog.portal",
        description="Sundog developer portal — deploy and adjust runner jobs",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    parser.add_argument("--out", default=".", help="Directory for run output files (default: .)")
    parser.add_argument("--eyesonly-url", default="", help="EyesOnly public/js/ URL for Gone Rogue runs")
    args = parser.parse_args(argv)
    start_server(
        host=args.host,
        port=args.port,
        output_dir=args.out,
        eyesonly_url=args.eyesonly_url,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
