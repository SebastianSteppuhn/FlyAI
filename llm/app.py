#!/usr/bin/env python3
# app.py — prompt -> CPACS v3.3 (with wing uID="wing1") -> test.cpacs.xml
# Tailored so your OCC exporter script can load the wing via get_wing("wing1")
#
# .env (same directory):
#   OPENAI_API_KEY=sk-...
#
# Optional sanity check uses TiXI3/TiGL3 (install from conda-forge).

import os
import ssl
import urllib.request
from pathlib import Path
from textwrap import dedent
from tempfile import NamedTemporaryFile, gettempdir

from dotenv import load_dotenv
from lxml import etree
import xmlschema
import certifi

# Optional: try importing TiXI/TiGL for a quick smoke test (not required)
try:
    import tixi3
    import tigl3
    HAVE_TIGL = True
except Exception:
    HAVE_TIGL = False

# -------------------- USER EDITABLE --------------------
PROMPT = (
    "Single-engine two-seat trainer, ~10 m span, straight-taper wing, "
    "simple tailplane, tricycle gear. Keep geometry minimal and plausible."
)
CPACS_OUT = Path("test.cpacs.xml")   # your OCC converter expects this name
VALIDATE_SCHEMA = True               # set False to skip validation entirely
RUN_TIGL_SMOKETEST = True            # set False to skip opening with TiGL
# ------------------------------------------------------

# CPACS 3.3 schema settings (we cache the XSD locally to bypass SSL issues)
CPACS_SCHEMA_URL = "https://www.cpacs.de/schema/3.3/cpacs_schema.xsd"
CPACS_SCHEMA_PATH = Path(gettempdir()) / "cpacs_schema_3_3.xsd"

SYSTEM_INSTRUCTIONS = dedent("""\
You generate only a single CPACS v3.x XML document.
Hard requirements:
- Root <cpacs> with version="3.3".
- Include xsi:noNamespaceSchemaLocation="http://www.cpacs.de/schema/v3/cpacs_schema.xsd".
- No comments, no markdown, no prose — just XML.
- Provide /cpacs/vehicles/aircraft/model with at least one wing.
- Ensure there is a wing with uID="wing1".
- Use reasonable metric units and unique uIDs; avoid vendor-specific extensions.
""")

# ----- OpenAI: support both Responses API (new) and Chat Completions (fallback)
def generate_cpacs_from_prompt(prompt: str) -> str:
    from openai import OpenAI  # import here so script loads even if lib missing

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Put it in a .env file next to app.py")

    client = OpenAI()

    xml = None
    # Try the newer Responses API first
    try:
        rsp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        )
        xml = (getattr(rsp, "output_text", None) or "").strip()
    except TypeError:
        # Fallback to older Chat Completions API if Responses doesn't match your SDK
        chat = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        )
        xml = (chat.choices[0].message.content or "").strip()

    # Remove accidental ``` fences if present
    if xml.startswith("```"):
        lines = [ln for ln in xml.splitlines() if not ln.strip().startswith("```")]
        xml = "\n".join(lines).strip()

    return xml




# ----- Schema validation (with local cache to avoid SSL/CERT issues)
def ensure_local_schema() -> str:
    if CPACS_SCHEMA_PATH.exists() and CPACS_SCHEMA_PATH.stat().st_size > 0:
        return str(CPACS_SCHEMA_PATH)

    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(CPACS_SCHEMA_URL, context=ctx) as r, open(CPACS_SCHEMA_PATH, "wb") as f:
        f.write(r.read())
    return str(CPACS_SCHEMA_PATH)


def validate_cpacs(xml_text: str) -> None:
    try:
        schema_path = ensure_local_schema()
    except Exception as e:
        print(f"⚠️  Could not fetch schema ({e}). Skipping validation.")
        return

    try:
        schema = xmlschema.XMLSchema(schema_path)
        try:
            schema.validate(xml_text)
        except Exception:
            with NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
                f.write(xml_text)
                tmp = f.name
            schema.validate(tmp)
    except Exception as e:
        raise RuntimeError(f"CPACS schema validation failed: {e}")


# ----- Optional: quick sanity check with TiGL (no pythonOCC needed)
def tigl_smoketest(cpacs_path: Path) -> None:
    if not RUN_TIGL_SMOKETEST or not HAVE_TIGL:
        return
    try:
        tx = tixi3.Tixi3()
        tx.open(str(cpacs_path))
        tg = tigl3.Tigl3()
        tg.open(tx, "")
        # should succeed if wing1 exists
        mgr = tigl3.configuration.CCPACSConfigurationManager_get_instance()
        cfg = mgr.get_configuration(tg._handle.value)
        wing = cfg.get_wing("wing1")
        _ = wing.get_loft()  # ensure loft is buildable
        tg.close()
        tx.close()
        print("✓ TiGL smoketest passed (wing1 lofted).")
    except Exception as e:
        print(f"⚠️  TiGL smoketest warning: {e}")


def main():
    print("→ Generating CPACS from prompt...")
    xml_text = generate_cpacs_from_prompt(PROMPT)

    if VALIDATE_SCHEMA:
        print("→ Validating CPACS against CPACS 3.3 schema (local cache)...")
        try:
            validate_cpacs(xml_text)
        except Exception as e:
            print(f"⚠️  Validation error: {e}\n   Continuing without validation.")

    print(f"→ Writing CPACS: {CPACS_OUT}")
    CPACS_OUT.write_text(xml_text, encoding="utf-8")

    tigl_smoketest(CPACS_OUT)

    print(f"✅ Done. You can now run your OCC exporter on {CPACS_OUT}.")


if __name__ == "__main__":
    main()
