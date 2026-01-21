"""
This module implements a decorator that takes the documentation
from a sympy function, for example ``sympy.printing.latex.latex``,
and merge it with the documentation of another function,
for example ``sympy_equation.printing.extended_latex``.

This assures that the documentation of all options from the first 
function (``latex``) are visible on the documentation of the second
function (``extended_latex``), without the need to hard code them.
This allows future improvement in the docstring of ``latex`` to be 
reflected on ``extended_latex``.
"""

import re
import inspect
from collections import OrderedDict


SECTION_HEADER_RE = re.compile(
    r"""
    (?P<title>^[A-Z][A-Za-z0-9 _-]*)       
    \n
    (?P<underline>[-=]{3,})               
    \n
    (?P<body>.*?)(?=^[A-Z][A-Za-z0-9 _-]*\n[-=]{3,}\n|\Z)
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE
)

IMPORT_LINE = ">>> from sympy_equation import extended_latex"


def parse_sections_and_lead(doc):
    if not doc:
        return "", OrderedDict()
    sections = OrderedDict()
    matches = list(SECTION_HEADER_RE.finditer(doc))

    if not matches:
        return doc.strip(), sections

    lead = doc[:matches[0].start()].strip()

    for m in matches:
        title = m.group("title").strip()
        body = m.group("body").rstrip()
        sections[title] = body
    return lead, sections


def normalize_concat(a, b):
    a = a.rstrip()
    b = b.lstrip()
    if not a: return b
    if not b: return a
    return a + "\n" + b


def normalize_examples(body, *, is_local):
    """
    For Examples:
    - Upstream examples (is_local=False) get the import added.
    - Local examples (is_local=True) DO NOT get the import added.
    - All existing import lines are removed first.
    """
    if ">>>" not in body:
        return body

    # Remove all existing import lines to avoid duplicates
    cleaned = "\n".join(
        line for line in body.splitlines()
        if not line.strip().startswith(IMPORT_LINE)
    ).lstrip()

    if is_local:
        # Do NOT add the import again for the local Examples section
        return cleaned

    # Upstream: add import at top
    return f"{IMPORT_LINE}\n{cleaned}"


def inject_import_before_first_doctest(body):
    if ">>>" not in body:
        return body
    if IMPORT_LINE in body:
        return body

    parts = body.split(">>>", 1)
    before = parts[0].rstrip()
    after = parts[1]

    if before:
        return before + "\n\n" + IMPORT_LINE + "\n>>>"+ after
    else:
        return IMPORT_LINE + "\n>>>" + after


def extend_doc(latex_func, replace_call="latex", new_call="extended_latex"):
    def decorator(func):
        upstream_doc = inspect.getdoc(latex_func) or ""
        local_doc = inspect.getdoc(func) or ""

        up_lead, up_sections = parse_sections_and_lead(upstream_doc)
        lo_lead, lo_sections = parse_sections_and_lead(local_doc)

        leading = normalize_concat(up_lead, lo_lead)
        merged = OrderedDict()

        # --- Process upstream sections ---
        for title, body in up_sections.items():
            body = body.replace(f"{replace_call}(", f"{new_call}(")
            key = title.lower()

            if key == "examples":
                body = normalize_examples(body, is_local=False)
            elif key == "notes":
                body = inject_import_before_first_doctest(body)

            merged[title] = body

        # --- Process local sections ---
        for title, body in lo_sections.items():
            key = title.lower()

            if key == "examples":
                body = normalize_examples(body, is_local=True)
            elif key == "notes":
                body = inject_import_before_first_doctest(body)

            if title in merged:
                # Add a clean blank line between upstream and local Examples
                merged[title] = merged[title].rstrip() + "\n\n" + body.lstrip()
            else:
                merged[title] = body

        # --- Assemble final docstring ---
        result = ""
        if leading.strip():
            result += leading.strip() + "\n\n"

        for title, body in merged.items():
            underline = "=" * len(title)

            if title.lower() == "examples":
                result += f"{title}\n{underline}\n\n{body.rstrip()}\n\n"
            else:
                result += f"{title}\n{underline}\n{body.rstrip()}\n\n"

        func.__doc__ = result.rstrip() + "\n"
        return func

    return decorator
