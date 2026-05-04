# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - single entry point. detects OS, kicks off the engine build in the background,
#   and starts the MRI server. PyTorch backend serves immediately. C backend lights
#   up when the build completes. If the kernel for the current OS+arch is missing
#   or the build fails, the dashboard shows it disabled and the Logs tab carries
#   the status.
# run.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "veritate_mri"))

import logs as logmod
import build_runner
from readers import paths

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

def main():
    ap = argparse.ArgumentParser(description="Run the Veritate MRI dashboard. Auto-builds the engine for the current OS in the background.")
    ap.add_argument("--port",    type=int, default=8001)
    ap.add_argument("--threads", type=int, default=0,
                    help="pytorch CPU threads. 0 = auto: physical cores capped at 16.")
    ap.add_argument("--model",   default="auto")
    ap.add_argument("--step",    type=int, default=None)
    ap.add_argument("--skip-build", action="store_true",
                    help="do not auto-build the engine. dashboard still serves PyTorch.")
    args, rest = ap.parse_known_args()

    logmod.info("run", f"detected {paths.current_os()}/{paths.current_arch()}")
    logmod.info("run", f"engine binary path: {paths.engine_binary_path()}")

    if not args.skip_build:
        build_runner.start()
    else:
        logmod.info("run", "build skipped (--skip-build)")

    launch_cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]

    sys.argv = [sys.argv[0],
                "--model",   args.model,
                "--port",    str(args.port),
                "--threads", str(args.threads)]
    if args.step is not None:
        sys.argv += ["--step", str(args.step)]
    sys.argv += rest

    import app as mri_app
    mri_app.app.config["LAUNCH_CMD"] = launch_cmd
    mri_app.main()


if __name__ == "__main__":
    main()
