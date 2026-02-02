
"""
This module contains functions to modify the generation of the Sphinx
documentation. They are meant to be private and will be modified/deleted
as the code progresses. Use it at your own risk.
"""
import param
from param.parameterized import label_formatter
import inspect
import re
import textwrap


param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

# Parameter attributes which are never shown
IGNORED_ATTRS = [
    'precedence', 'check_on_set', 'instantiate', 'pickle_default_value',
    'watchers', 'compute_default_fn', 'doc', 'owner', 'per_instance',
    'is_instance', 'name', 'time_fn', 'time_dependent', 'allow_refs',
    'nested_refs', 'rx', 'label', 'softbounds', 'step'
]

# Default parameter attribute values (value not shown if it matches defaults)
DEFAULT_VALUES = {'allow_None': False, 'readonly': False, 'constant': False}


def should_skip_this_member(pobj, name, value, label):
    try:
        is_default = bool(DEFAULT_VALUES.get(name) == value)
    except Exception:
        is_default = False

    skip = (
        name.startswith('_') or
        name in IGNORED_ATTRS or
        inspect.ismethod(value) or
        inspect.isfunction(value) or
        value is None or
        is_default or
        (name == 'label' and pobj.label != label)
    )
    return skip


def process_List(child, pobj, ptype, label, members):
    lines = [f"{child} : {ptype}"]
    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if name == "class_":
            skip = True
        if name == "bounds":
            if (
                (value == (0, None))
                or (value == (None, None))
                or (value is None)
            ):
                skip = True
            elif (value[1] is None) and value[0]:
                name = "min_length"
                value = str(value[0])
            elif (value[0] is None) and value[1]:
                name = "max_length"
                value = str(value[1])
            else:
                name = "length"
                if value[0] != value[1]:
                    value = f"{value[0]} <= len({child}) <= {value[1]}"
                else:
                    value = str(value[0])
        if name == "item_type":
            if value is None:
                skip = True
            elif "'" in str(value):
                values = str(value).split(",")
                values = [s.split("'")[1] for s in values]
                if len(values) == 1:
                    value = values[0]
                else:
                    value = "(%s)" % ", ".join(values)
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def process_ClassSelector(child, pobj, ptype, label, members):
    types = pobj.class_
    if isinstance(types, type):
        types = str(types).split("'")[1]
    elif isinstance(types, tuple):
        types = str(types).split(",")
        for i, v in enumerate(types):
            types[i] = v.split("'")[1]
        types = "(%s)" % ", ".join(types)
    lines = [f"{child} : {types}"]

    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if name == "class_":
            skip = True
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def process_Selector(child, pobj, ptype, label, members):
    lines = [f"{child} : {ptype}"]

    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if name == "default":
            skip = False
        if name == "names":
            skip = True
        if name == "objects":
            name = "options"
        if name == "default" and isinstance(value, str) and ("'" not in value):
            value = f"'{value}'"
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def process_Number(child, pobj, ptype, label, members):
    lines = [f"{child} : {ptype}"]
    constant = pobj.constant
    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if constant and name == "default":
            skip = True
        if name == "constant":
            name = "read only"
        if name == "inclusive_bounds":
            skip = True
        if name == "bounds":
            inclusive_bounds = pobj.inclusive_bounds

            if (value == (None, None)) or (value is None):
                skip = True
            elif all(v is not None for v in value):
                symbol_1 = "<=" if inclusive_bounds[0] else "<"
                symbol_2 = "<=" if inclusive_bounds[1] else "<"
                value = f"{value[0]} {symbol_1} {child} {symbol_2} {value[1]}"
            elif value[0] is None:
                symbol = "<=" if inclusive_bounds[0] else "<"
                value = f"{child} {symbol} {value[1]}"
            else:
                symbol = ">=" if inclusive_bounds[1] else ">"
                value = f"{child} {symbol} {value[0]}"
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def process_Parameter(child, pobj, ptype, label, members):
    lines = [f"{child} : {ptype}"]
    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if name == "class_":
            skip = True
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


ptype_func_map = {
    "List":             process_List,
    "Selector":         process_Selector,
    "ClassSelector":    process_ClassSelector,
    "Number":           process_Number,
    "Integer":          process_Number,
}


# def


def param_formatter(app, what, name, obj, options, lines, is_sphinx=True):
    if what == 'module':
        lines = ["start"]

    if what == 'class' and isinstance(obj, param.parameterized.ParameterizedMetaclass):

        parameters = ['name']
        lines.extend([
            "Attributes" if is_sphinx else "Parameters",
            "==========" if is_sphinx else "=========="
        ])

        params = [p for p in obj.param if p not in parameters]
        for child in params:
            if child in ["print_level", "name"]:
                continue
            if child[0] == "_":
                continue
            if (
                hasattr(obj, "_params_to_document")
                and child not in obj._params_to_document
            ):
                continue

            pobj = obj.param[child]
            label = label_formatter(pobj.name)
            doc = pobj.doc or ""
            members = inspect.getmembers(pobj)
            ptype = pobj.__class__.__name__

            func = ptype_func_map.get(ptype, process_Parameter)
            p_lines = func(child, pobj, ptype, label, members)
            if is_sphinx:
                doc = '   %s' % doc
                p_lines.extend([
                    '',
                    doc,
                    ''
                ])
            else:
                doc = textwrap.indent(textwrap.dedent(doc), "   ")
                if doc and doc[0] == "\n":
                    doc = doc[1:]
                p_lines.append(doc)

            lines.extend(p_lines)


# NOTE: these are the numpydoc sections that I'm aware of. They will be used
# as keys in some dictionary. 'general' is not a real section: the value
# associated to this key will be lines of docstring starting from line 0
# up to the first section which will be found.
# The final docstring will be build according to the order of these keys.
numpydoc_sections_to_look_for = [
    "general", "Parameters", "Attributes", "Methods",
    "Returns", "Yields", "Raises", "Warns", "Warnings",
    "Examples", "Notes", "References", "See Also",
]


def split_docstring(docstring: str) -> dict:
    """
    Split the docstring of a function or class into sections, like
    Parameters, Returns, etc.

    Returns
    =======
    d : dict
        A dictionary having the form {
        "general": docstring_before_sections,
        "Parameters": docstring_for_parameters,
        "Returns": docstring_for_returns,
    }
    """
    docstring = textwrap.dedent(docstring).strip()

    # Regular expression to find section headers
    section_header_re = re.compile(r'^(?P<header>\w[\w ]*)\n[-=]{3,}$', re.MULTILINE)

    sections = {}
    current_section = "general"
    matches = list(section_header_re.finditer(docstring))

    if matches:
        # Process the general part (before first section)
        first_match = matches[0]
        general_text = docstring[:first_match.start()].strip()
        sections[current_section] = general_text if general_text else ""

        # Process each section
        for i, match in enumerate(matches):
            header = match.group("header").strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(docstring)
            content = docstring[start:end].strip()
            sections[header] = content
    else:
        # No sections found, everything is general
        sections[current_section] = docstring

    return sections


def add_parameters_to_docstring(printer_cls):
    def decorator(func):
        original_docstring = textwrap.dedent(func.__doc__)
        if original_docstring is None:
            return func

        sections = split_docstring(original_docstring)
        print("sections.keys", sections.keys())

        docstring_sections = split_docstring(original_docstring)
        lines = []
        param_formatter(None, "class", None, printer_cls, None, lines, False)
        parameters_section = "\n".join(lines)

        # rearrange the sections and add colors
        final_docstring = docstring_sections.get("general", "")
        final_docstring += "\n\n" + parameters_section

        for section in docstring_sections:
            print(section)
            if section != "general":
                final_docstring += "\n\n" + section + "\n" + "=" * len(section)
                final_docstring += "\n" + docstring_sections[section]

        func.__doc__ = final_docstring
        return func
    return decorator
