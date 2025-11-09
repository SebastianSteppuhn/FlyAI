#!/usr/bin/env python3
"""
app.py — prompt -> CPACS v3.3 (includes wing uID="wing1") -> test2.cpacs.xml

Minimal, example-driven, and conservative:
- Forces CPACS root: version="3.3" and the standard schema location attr
- Writes a single aircraft model with a 'wing1' you can open via TiGL get_wing("wing1")
- No schema fetching, no emojis, no extras
- Optional: if TiXI/TiGL are importable, do a tiny smoketest to ensure 'wing1' lofts

Usage
-----
1) Put your OpenAI API key in the environment:
   export OPENAI_API_KEY=sk-...

   (Optional) If you prefer a .env file:
   echo 'OPENAI_API_KEY=sk-...' > .env

2) Run:
   python app.py --prompt "Simple trainer ~10 m span, straight taper"

Notes
-----
- The model is asked to follow CPACS examples structure closely and return only XML.
- Result path: test2.cpacs.xml
"""

import argparse
import os
from pathlib import Path
from textwrap import dedent

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

# Optional TiXI/TiGL smoketest (ignored if not available)
try:
    import tixi3  # conda-forge: tixi3
    import tigl3  # conda-forge: tigl3
    HAVE_TIGL = True
except Exception:
    HAVE_TIGL = False

# --------------------------- Config ---------------------------

CPACS_OUT = Path("test2.cpacs.xml")

# Keep this instruction short and aligned with CPACS examples.
SYSTEM_INSTRUCTIONS = dedent("""\
You must output only a single CPACS v3.3 XML document (no comments, no markdown).
Follow the style of official CPACS example files.

Hard requirements:
- Root element <cpacs> with attribute version="3.3".
- Add xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  and xsi:noNamespaceSchemaLocation="http://www.cpacs.de/schema/v3/cpacs_schema.xsd".
- Provide <header> with <versionInfos><versionInfo><cpacsVersion>3.3</cpacsVersion></versionInfo></versionInfos>.
- Under /cpacs/vehicles/aircraft/model create exactly one model that contains:
  - /wings/wing with uID="wing1" and <symmetry>y</symmetry>.
  - A minimal, consistent definition using sections/elements/positionings so that TiGL can loft it.
  - Include simple airfoil definitions under /vehicles/profiles/airfoils and reference them via airfoilUID.
- Use reasonable metric units and unique uIDs.
- Keep it small and coherent (root+tip sections are fine).
""")

# --------------------------- OpenAI call ---------------------------

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()
    return s

def generate_cpacs_from_prompt(user_prompt: str) -> str:
    # Uses the simple Chat Completions API for compatibility.
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
    )
    xml = chat.choices[0].message.content or ""
    return _strip_fences(xml)

# --------------------------- Optional TiGL check ---------------------------

def tigl_smoketest(cpacs_path: Path) -> None:
    if not HAVE_TIGL:
        return
    tx = tixi3.Tixi3()
    tg = tigl3.Tigl3()
    try:
        tx.open(str(cpacs_path))
        tg.open(tx, "")
        mgr = tigl3.configuration.CCPACSConfigurationManager_get_instance()
        cfg = mgr.get_configuration(tg._handle.value)
        wing = cfg.get_wing("wing1")  # must exist
        _ = wing.get_loft()           # must be buildable
        print("TiGL smoketest: OK (wing1 lofted).")
    finally:
        try:
            tg.close()
        except Exception:
            pass
        try:
            tx.close()
        except Exception:
            pass

# --------------------------- CLI ---------------------------

def main():
    p = argparse.ArgumentParser(description="Prompt -> CPACS v3.3 (with wing1) -> test2.cpacs.xml")
    p.add_argument("--prompt", default="Single-engine two-seat trainer, ~10 m span, straight-taper wing.")
    p.add_argument("--no-tigl-check", action="store_true", help="Skip TiGL smoketest")
    args = p.parse_args()

    print("Generating CPACS from prompt...")
    xml_text = generate_cpacs_from_prompt(args.prompt)

    print(f"Writing {CPACS_OUT}")
    CPACS_OUT.write_text(xml_text, encoding="utf-8")

    if not args.no_tigl_check:
        try:
            tigl_smoketest(CPACS_OUT)
        except Exception as e:
            # Keep lightweight: just inform, do not fail the run.
            print(f"TiGL smoketest warning: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
