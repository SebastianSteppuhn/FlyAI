#!/usr/bin/env python3
"""
Run SU2_CFD on run.cfg and log output to su2_out.log.

Use this once per CFD run. Then use plot_wing_drag.py as many times
as you like without rerunning SU2.
"""

import subprocess
import pathlib
from collections import deque

CASE_DIR   = pathlib.Path("")
CFG_FILE   = CASE_DIR / "run.cfg"
SU2_BINARY = "SU2_CFD"          # or absolute path if needed
SU2_LOG    = CASE_DIR / "su2_out.log"


def tail(filename, n=80):
    dq = deque(maxlen=n)
    with open(filename, "r") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


def run_su2():
    print(f"Running {SU2_BINARY} {CFG_FILE} ...")
    with open(SU2_LOG, "w") as log:
        result = subprocess.run(
            [SU2_BINARY, str(CFG_FILE)],
            cwd=CASE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print("\n[ERROR] SU2_CFD returned non-zero exit status.")
        print(f"Return code: {result.returncode}")
        print(f"Full SU2 output is in: {SU2_LOG}")
        print("\n--- Last 80 lines of SU2 output ---")
        for line in tail(SU2_LOG, n=80):
            print(line)
        print("--- End of SU2 output tail ---\n")
        raise SystemExit("SU2_CFD failed.")
    else:
        print("SU2 run finished successfully.")
        print(f"Log written to {SU2_LOG}")


if __name__ == "__main__":
    run_su2()
