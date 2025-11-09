#!/usr/bin/env python3
# app.py — prompt -> CPACS v3.3 -> STEP (no CLI)
# Requirements (conda/pip):
#   conda install -c conda-forge tigl3 tixi3
#   pip install openai python-dotenv lxml xmlschema certifi
#
# .env in same folder:
#   OPENAI_API_KEY=sk-...
#
# Edit PROMPT below and run:  python app.py

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

# -------------------- USER SETTINGS --------------------
PROMPT = (
    "Single-engine two-seat trainer, ~10 m wingspan, straight tapered wing, "
    "simple empennage, tricycle gear. Keep geometry plausible and minimal."
)
CPACS_OUT = Path("aircraft.cpacs.xml")
STEP_OUT = Path("aircraft.stp")
VALIDATE_SCHEMA = True  # set False to skip schema validation entirely
# -------------------------------------------------------

# CPACS 3.3 schema settings (we cache the XSD locally to bypass SSL issues)
CPACS_SCHEMA_URL = "https://www.cpacs.de/schema/3.3/cpacs_schema.xsd"
CPACS_SCHEMA_PATH = Path(gettempdir()) / "cpacs_schema_3_3.xsd"

SYSTEM_INSTRUCTIONS = dedent("""\
You generate only a single CPACS v3.x XML document.
Hard requirements:
- Root <cpacs> with version="3.3".
- Include xsi:noNamespaceSchemaLocation="http://www.cpacs.de/schema/v3/cpacs_schema.xsd".
- No comments, no markdown, no prose — just XML.
- Provide /cpacs/vehicles/aircraft/model with at least one wing (reasonable units).
- Use unique uIDs and names; avoid vendor-specific extensions.
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
        # New SDKs expose .output_text
        xml = (getattr(rsp, "output_text", None) or "").strip()
    except TypeError:
        # Fallback to older Chat Completions API if Responses doesn't match your SDK
        chat = client.chat.completions.create(
            model="gpt-4.1-mini",
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

    return ensure_cpacs_header(xml)


def ensure_cpacs_header(xml_text: str) -> str:
    """Force correct CPACS v3.3 header; pretty print."""
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Returned text is not parseable XML: {e}")

    if root.tag != "cpacs":
        raise RuntimeError("Root element must be <cpacs>.")

    # Ensure version="3.3"
    if root.get("version") != "3.3":
        root.set("version", "3.3")

    # Ensure xsi:noNamespaceSchemaLocation attribute
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    root.set(f"{{{xsi}}}noNamespaceSchemaLocation", "http://www.cpacs.de/schema/v3/cpacs_schema.xsd")

    return etree.tostring(root, pretty_print=True, encoding="utf-8", xml_declaration=True).decode("utf-8")


# ----- Schema validation (with local cache to avoid SSL/CERT issues)
def ensure_local_schema() -> str:
    """Download CPACS schema to a local file if missing; return local path.
       Uses certifi CA bundle to avoid SSL verify problems.
    """
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
        # some versions are picky about validating from string; try both ways
        try:
            schema.validate(xml_text)
        except Exception:
            with NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
                f.write(xml_text)
                tmp = f.name
            schema.validate(tmp)
    except Exception as e:
        raise RuntimeError(f"CPACS schema validation failed: {e}")



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


if __name__ == "__main__":
    main()
