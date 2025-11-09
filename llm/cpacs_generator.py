"""
cpacs_generator.py

Library-style helper to generate a CPACS 3.5 aircraft XML document
from a natural-language prompt using the OpenAI API, and validate it
against the official CPACS schema.

Usage from your own main/app:

    from cpacs_generator import generate_cpacs_aircraft

    def main():
        prompt = "Single-aisle, twin-engine airliner for 180 passengers, 3000 nm range"
        xml_text, is_valid, errors = generate_cpacs_aircraft(prompt)

        if is_valid:
            with open("aircraft_cpacs.xml", "w", encoding="utf-8") as f:
                f.write(xml_text)
        else:
            print("CPACS XML is invalid:")
            for e in errors:
                print(" -", e)

    if __name__ == "__main__":
        main()
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

from xml.etree import ElementTree as ET
import xmlschema
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# Official CPACS 3.5 schema URL
CPACS_SCHEMA_URL = "https://www.cpacs.de/schema/v3_5_0/cpacs_schema.xsd"


def strip_code_fences(text: str) -> str:
    """Remove ```...``` fences if the model wraps the XML in Markdown."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    # Drop first line (``` or ```xml)
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # Drop trailing ``` if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def build_system_prompt() -> str:
    """
    System prompt to force the model to emit CPACS 3.5 compliant aircraft XML.
    """
    return (
        "You are an expert in CPACS (Common Parametric Aircraft Configuration Schema) "
        "version 3.5. You generate *only* CPACS XML instances for fixed-wing aircraft.\n\n"
        "Requirements:\n"
        "1) Output must be a single, well-formed XML document (UTF-8).\n"
        "2) Start with the XML declaration:\n"
        '   <?xml version="1.0" encoding="UTF-8"?>\n'
        "3) The root element must be exactly:\n"
        '   <cpacs xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '          xsi:noNamespaceSchemaLocation="https://www.cpacs.de/schema/v3_5_0/cpacs_schema.xsd">\n'
        "   and must be closed with </cpacs>.\n"
        "4) The document MUST be valid against the official CPACS 3.5 XML schema at:\n"
        "   https://www.cpacs.de/schema/v3_5_0/cpacs_schema.xsd\n"
        "   (respect element order, required elements/attributes, and data types).\n"
        "5) Include a <header> element consistent with CPACS conventions "
        "(name, description, creator/author, and other allowed metadata).\n"
        "6) Within <vehicles>, define exactly one aircraft with at least one model, e.g.\n"
        "   <vehicles>\n"
        "     <aircraft>\n"
        "       <model> ... </model>\n"
        "     </aircraft>\n"
        "   </vehicles>\n"
        "   following the structure defined in the CPACS schema.\n"
        "7) Use reasonable UIDs and references, following CPACS naming conventions; "
        "ensure every required UID is unique and all references point to existing UIDs.\n"
        "8) Do NOT include any comments, explanations, markdown formatting, or CDATA; "
        "only the raw XML instance.\n"
    )


def create_client() -> OpenAI:
    """
    Create an OpenAI client from the OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before using generate_cpacs_aircraft()."
        )
    return OpenAI(api_key=api_key)


def call_openai_for_cpacs_xml(
    client: OpenAI,
    design_prompt: str,
    previous_xml: Optional[str] = None,
    validation_errors: Optional[List[str]] = None,
    model: str = "gpt-4.1",
) -> str:
    """
    Ask the model to generate (or fix) CPACS XML.

    If previous_xml + validation_errors are provided, we ask it to correct the XML.
    """
    system_prompt = build_system_prompt()

    if previous_xml is None:
        user_content = (
            "User design prompt describing the desired aircraft:\n"
            f"{design_prompt}\n\n"
            "Generate a *new* CPACS 3.5 XML document for this aircraft. "
            "Remember: output only the raw XML instance, no markdown."
        )
    else:
        errors_text = "\n".join(validation_errors or [])
        user_content = (
            "The following CPACS XML you previously generated did NOT validate "
            "against the CPACS 3.5 schema:\n\n"
            "----- INVALID XML BEGIN -----\n"
            f"{previous_xml}\n"
            "----- INVALID XML END -----\n\n"
            "The schema validation reported these errors:\n"
            f"{errors_text}\n\n"
            "Please return a corrected CPACS 3.5 XML instance that fixes these issues. "
            "Do not explain anything; output only the XML."
        )

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    xml = response.choices[0].message.content or ""
    xml = strip_code_fences(xml)
    return xml.strip()


def validate_cpacs_xml(xml_text: str, schema: xmlschema.XMLSchema) -> Tuple[bool, List[str]]:
    """
    Validate the given XML text against the CPACS schema.

    Returns:
        (is_valid, errors)
    """
    errors: List[str] = []

    # First: well-formedness check.
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        errors.append(f"XML not well-formed: {e}")
        return False, errors

    # Second: schema validity (version-agnostic: no XMLSchemaValidationError import).
    for err in schema.iter_errors(root):
        errors.append(str(err))

    return (len(errors) == 0), errors


def generate_cpacs_aircraft(
    design_prompt: str,
    *,
    model: str = "gpt-4.1",
    max_attempts: int = 3,
    client: Optional[OpenAI] = None,
) -> Tuple[str, bool, List[str]]:
    """
    High-level function you can call from your own main/app.

    Args:
        design_prompt: Natural-language description of the aircraft.
        model: OpenAI model name (default: "gpt-4.1").
        max_attempts: Maximum attempts to fix validation errors.
        client: Optionally pass an existing OpenAI client instance.

    Returns:
        (xml_text, is_valid, errors)
        - xml_text: The last XML returned by the model (valid or not).
        - is_valid: True if it validates against the CPACS 3.5 schema.
        - errors: List of validation error strings (empty if valid).
    """
    if client is None:
        client = create_client()

    # Load CPACS schema once
    schema = xmlschema.XMLSchema(CPACS_SCHEMA_URL)

    current_xml: Optional[str] = None
    validation_errors: List[str] | None = None
    is_valid = False

    for _ in range(max_attempts):
        current_xml = call_openai_for_cpacs_xml(
            client=client,
            design_prompt=design_prompt,
            previous_xml=current_xml,
            validation_errors=validation_errors,
            model=model,
        )

        is_valid, validation_errors = validate_cpacs_xml(current_xml, schema)

        if is_valid:
            break

    # current_xml should always be set if OpenAI responded
    if current_xml is None:
        raise RuntimeError("No XML was generated by the model.")

    return current_xml, is_valid, (validation_errors or [])


# Optional: example usage if you *do* run this file directly.
def _example_main() -> None:
    prompt = "Single-aisle, twin-engine turbofan airliner for ~180 passengers, 3000 nm range."
    xml_text, is_valid, errors = generate_cpacs_aircraft(prompt)

    print("Is valid CPACS?:", is_valid)
    if not is_valid:
        print("Errors:")
        for e in errors:
            print(" -", e)

    with open("aircraft_cpacs.xml", "w", encoding="utf-8") as f:
        f.write(xml_text)


if __name__ == "__main__":
    _example_main()
