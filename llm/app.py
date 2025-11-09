"""
One-shot GPT-5 CPACS generator for a VERY SIMPLE aircraft.

- Always uses a fixed header (Basic Wing Model) exactly as requested.
- GPT-5 only generates the part between </header> and </cpacs>.
- We wrap the result in a complete CPACS 3.5 document.
- We only check that the final XML is well-formed (enough to avoid TiXI NOT_WELL_FORMED).

Requirements:
    pip install --upgrade openai
    export OPENAI_API_KEY="sk-..."

Typical usage:

    from app import generate_cpacs_aircraft

    xml_text = generate_cpacs_aircraft("very simple 50 seat regional jet")
    with open("aircraft_cpacs.xml", "w", encoding="utf-8") as f:
        f.write(xml_text)
"""

import os
from xml.etree import ElementTree as ET
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Fixed header you requested (used verbatim)
# ---------------------------------------------------------------------------

CPACS_HEADER = """    <header>
        <name>Basic Wing Model</name>
        <version>1.0.0</version>
        <!-- Deprecated: The <cpacsVersion> is needed to open the file in TiGL. TiGL will soon be adapted to get this information from the versinInfo node. -->
        <cpacsVersion>3.5</cpacsVersion>
        <versionInfos>
            <versionInfo version="1.0.0">
                <creator>DLR-SL</creator>
                <timestamp>2019-12-04T10:30:00</timestamp>
                <description>Create initial data set</description>
                <cpacsVersion>3.5</cpacsVersion>
            </versionInfo>
        </versionInfos>
    </header>
"""


def _strip_code_fences(text: str) -> str:
    """Remove ```...``` if the model wraps the XML in markdown."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _build_system_prompt() -> str:
    """
    System prompt for GPT-5: produce only the CPACS body (between </header> and </cpacs>).

    This is tuned using your TiGL error messages:
    - Use fuselageProfiles + wingAirfoils with proper pointList (x, y, z all present,
      same number of entries).
    - Add transformations for fuselage, wing, sections.
    - Add positionings with names, valid fromSectionUID/toSectionUID.
    - Add elements with airfoilUID, and segments referencing elements correctly.
    """
    return (
        "You are an expert in CPACS 3.5 and TiGL.\n"
        "I will wrap your output inside a CPACS document with this structure:\n"
        "  <?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "  <cpacs xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
        "         xsi:noNamespaceSchemaLocation=\"https://www.cpacs.de/schema/v3_5_0/cpacs_schema.xsd\">\n"
        "      <header> ... fixed, provided by the caller ... </header>\n"
        "      <!-- YOUR OUTPUT GOES HERE -->\n"
        "  </cpacs>\n\n"
        "YOUR TASK:\n"
        "  - Output ONLY the inner CPACS body that goes AFTER </header> and BEFORE </cpacs>.\n"
        "  - Do NOT output the XML declaration, <cpacs>, </cpacs>, or <header>.\n"
        "  - Output ONE well-formed XML fragment, no markdown, no code fences.\n\n"
        "STRUCTURAL REQUIREMENTS (keep it VERY SIMPLE but valid for TiGL):\n"
        "1) Root part of your fragment:\n"
        "   <vehicles>\n"
        "     <profiles>\n"
        "       <fuselageProfiles>...</fuselageProfiles>\n"
        "       <wingAirfoils>...</wingAirfoils>\n"
        "     </profiles>\n"
        "     <aircraft>\n"
        "       <model uID=\"model1\"> ... </model>\n"
        "     </aircraft>\n"
        "   </vehicles>\n\n"
        "2) PROFILES:\n"
        "   - In <fuselageProfiles>, define EXACTLY ONE <fuselageProfile uID=\"fuselageProf1\">.\n"
        "   - In <wingAirfoils>, define EXACTLY ONE <wingAirfoil uID=\"wingAirfoil1\">.\n"
        "   - Each profile must contain a <pointList> with all required coordinates.\n"
        "   - For every point list:\n"
        "       * Provide coordinates for x, y and z as required by CPACS 3.5.\n"
        "       * Make sure the number of x, y and z values is the SAME.\n"
        "       * Use only a FEW points (keep geometry very simple).\n\n"
        "3) AIRCRAFT MODEL (ONLY ONE MODEL):\n"
        "   <aircraft>\n"
        "     <model uID=\"model1\">\n"
        "       <name>VerySimpleModel</name>\n"
        "       <description>Very simple test aircraft model.</description>\n"
        "       <fuselages>...</fuselages>\n"
        "       <wings>...</wings>\n"
        "     </model>\n"
        "   </aircraft>\n\n"
        "4) FUSELAGE (ONE SIMPLE FUSELAGE):\n"
        "   - Define exactly one <fuselage uID=\"fuselage1\">.\n"
        "   - Include a <transformation> at fuselage level (with translation, rotation, scaling).\n"
        "   - Inside the fuselage:\n"
        "       <sections> with at least TWO <section> elements (uID=\"fuseSec1\", \"fuseSec2\").\n"
        "       Each section MUST contain a <transformation>.\n"
        "       Each section MUST contain at least one <element> referencing fuselageProf1.\n"
        "   - Include <positionings> with at least ONE <positioning>:\n"
        "       * A <name> element (required by TiGL).\n"
        "       * fromSectionUID and toSectionUID referencing existing section UIDs.\n"
        "   - Include <segments> that connect the sections using the elements.\n\n"
        "5) WING (ONE SIMPLE MAIN WING):\n"
        "   - In <wings>, define exactly one <wing uID=\"wing1\" symmetry=\"x-z-plane\">.\n"
        "   - Include a <transformation> at wing level (simple translation + rotation).\n"
        "   - Define <sections> with at least TWO <section> elements (uID=\"wingSec1\", \"wingSec2\").\n"
        "       * Each section MUST include a <transformation>.\n"
        "       * Each section MUST contain EXACTLY ONE <element> with:\n"
        "             <airfoilUID>wingAirfoil1</airfoilUID>\n"
        "             a <transformation> (to position the airfoil in that section).\n"
        "   - Define <positionings> with at least ONE <positioning> node:\n"
        "       * A <name> element.\n"
        "       * fromSectionUID and toSectionUID referencing wingSec1 and wingSec2.\n"
        "   - Define <segments> with at least ONE <segment> that connects the two elements:\n"
        "       * fromElementUID and toElementUID referencing the element UIDs from sections.\n\n"
        "6) GENERAL:\n"
        "   - Use very simple numeric values (small integers or one decimal place).\n"
        "   - Make sure EVERY UID that is referenced actually exists.\n"
        "   - Ensure XML is well-formed: matching opening / closing tags, properly nested.\n"
        "   - Do NOT include any comments except those already inside the fixed header (which you do NOT output).\n"
        "   - No CDATA, no markdown, no explanations.\n"
    )


def generate_cpacs_aircraft(
    design_prompt: str,
    *,
    model: str = "gpt-5",
    client: Optional[OpenAI] = None,
) -> str:
    """
    One-shot call to GPT-5 that returns a VERY SIMPLE CPACS 3.5 XML document.

    - Uses the fixed Basic Wing Model header.
    - GPT-5 only generates the body between </header> and </cpacs>.
    - We wrap it into a full CPACS file and check for XML well-formedness.

    Raises RuntimeError if:
      - OPENAI_API_KEY is missing
      - The returned XML is not well-formed
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if client is None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)

    system_prompt = _build_system_prompt()

    user_prompt = (
        "Short description of the desired aircraft (keep it simple):\n"
        f"{design_prompt}\n\n"
        "Now generate the CPACS 3.5 BODY as described in the instructions.\n"
        "Remember: output ONLY the XML fragment that goes after </header> "
        "and before </cpacs>."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=1,  # single deterministic-ish try
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    body_xml = resp.choices[0].message.content or ""
    body_xml = _strip_code_fences(body_xml).strip()

    full_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<cpacs xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '       xsi:noNamespaceSchemaLocation="https://www.cpacs.de/schema/v3_5_0/cpacs_schema.xsd">\n'
        f"{CPACS_HEADER}\n"
        f"{body_xml}\n"
        "</cpacs>\n"
    )

    # Single quick well-formedness check so TiXI doesn't blow up on parse.
    try:
        ET.fromstring(full_xml)
    except ET.ParseError as e:
        snippet = full_xml[:1000]
        raise RuntimeError(
            f"Generated CPACS XML is not well-formed: {e}\n\n"
            f"First 1000 characters of XML:\n{snippet}"
        ) from e

    return full_xml


# Example usage if you run this file directly
def main() -> None:
    design_prompt = "Very simple vtol without fuselage and gears"
    xml_text = generate_cpacs_aircraft(design_prompt)

    out_file = "aircraft_cpacs.xml"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(xml_text)

    print(f"Wrote {out_file} (XML is well-formed).")


if __name__ == "__main__":
    main()
