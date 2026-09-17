"""Static integrity checks for real_estate_app.py.

Python compilation does NOT catch these: a duplicate Streamlit widget key raises
only at runtime when that branch renders, and a duplicate dict key is silently
legal — the later value just wins. Both have bitten this app, so they get a
standing test rather than a one-off inspection.

Run:  python test_static_integrity.py
Exit code 0 = clean, 1 = problems found (suitable for CI).
"""
import ast
import collections
import pathlib
import re
import sys

APP = pathlib.Path(__file__).with_name("real_estate_app.py")
WIDGET_KEY_RE = r'key\s*=\s*"([A-Za-z0-9_]+)"'
TR_KEY_RE = r'tr\(\s*"([a-z0-9_]+)"\s*[,)]'


def load() -> str:
    return APP.read_text(encoding="utf-8")


def _widget_key_sites(tree: ast.AST):
    """Every call passing key="..." , with its AST node."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    yield kw.value.value, node


def _branch_path(tree: ast.AST, target: ast.AST) -> tuple:
    """Identify which if/elif/else arm a node sits in.

    Two widgets sharing a key are only a bug if both can render in the SAME run.
    Arms of one if/else are mutually exclusive, so a shared key there is legal —
    flagging it would push authors to invent pointless distinct keys."""
    path = []
    def walk(node, trail):
        for child in ast.iter_child_nodes(node):
            if child is target:
                path.extend(trail)
                return True
            if isinstance(node, ast.If):
                arm = "body" if child in node.body else (
                    "orelse" if child in node.orelse else None)
                if arm and walk(child, trail + [(id(node), arm)]):
                    return True
                if arm:
                    continue
            if walk(child, trail):
                return True
        return False
    walk(tree, [])
    return tuple(path)


def duplicate_widget_keys(src: str) -> dict:
    """Streamlit raises DuplicateWidgetID when two widgets share a key.

    AST-based so mutually exclusive branches are not reported: keys are only a
    conflict when two sites could execute in the same run. A plain textual count
    over the whole file cannot tell those apart."""
    tree = ast.parse(src)
    sites = collections.defaultdict(list)
    for key, node in _widget_key_sites(tree):
        sites[key].append(node)
    conflicts = {}
    for key, nodes in sites.items():
        if len(nodes) < 2:
            continue
        paths = [_branch_path(tree, n) for n in nodes]
        # Mutually exclusive iff some common if-statement puts them on
        # different arms.
        def exclusive(a, b):
            da, db = dict(a), dict(b)
            return any(da[k] != db[k] for k in set(da) & set(db))
        if not all(exclusive(paths[i], paths[j])
                   for i in range(len(paths)) for j in range(i + 1, len(paths))):
            conflicts[key] = len(nodes)
    return conflicts


def duplicate_dict_keys(src: str) -> list:
    """Literal dict keys repeated in one display — silently legal, usually a bug."""
    found = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Dict):
            lits = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            dup = {k: n for k, n in collections.Counter(lits).items() if n > 1}
            if dup:
                found.append((node.lineno, dup))
    return found


def duplicate_panels(src: str) -> dict:
    """Panels that must appear exactly once in the analyze path."""
    try:
        ana = src[src.index("def render_manual_analyzer"):]
    except ValueError:
        return {"render_manual_analyzer": 0}
    counts = {
        "financing panel": ana.count("financing_analysis("),
        "size panel": len(re.findall(r"tr\(['\"]size_header['\"]\)", ana)),
        "size N/A branch": len(re.findall(r"tr\(['\"]size_na['\"]\)", ana)),
    }
    return {k: v for k, v in counts.items() if v != 1}


def unresolved_tr_keys(src: str) -> list:
    """Every literal tr("key") must exist in the English table.

    Parsed with AST, not regex: a non-greedy regex over the TRANSLATIONS literal
    stops at the first closing brace it meets and silently reports most of the
    table as undefined (which it did on the first run of this file)."""
    defined = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TRANSLATIONS" for t in node.targets):
            if isinstance(node.value, ast.Dict):
                for lang_dict in node.value.values:
                    if isinstance(lang_dict, ast.Dict):
                        defined |= {k.value for k in lang_dict.keys
                                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    used = set(re.findall(TR_KEY_RE, src))
    return sorted(used - defined) if defined else []


def deprecated_width_kwarg(src: str) -> int:
    """use_container_width passed as an actual keyword argument.

    A regex also matches the word inside docstrings that merely explain the ban,
    so this counts real call-site keywords only."""
    return sum(1 for node in ast.walk(ast.parse(src))
               if isinstance(node, ast.Call)
               for kw in node.keywords
               if kw.arg == "use_container_width")


def main() -> int:
    src = load()
    problems = []

    dup_keys = duplicate_widget_keys(src)
    if dup_keys:
        problems.append(f"duplicate Streamlit widget keys: {dup_keys}")

    dup_dicts = duplicate_dict_keys(src)
    if dup_dicts:
        problems.append(f"duplicate dict keys: {dup_dicts}")

    dup_panels = duplicate_panels(src)
    if dup_panels:
        problems.append(f"panels not appearing exactly once: {dup_panels}")

    missing = unresolved_tr_keys(src)
    if missing:
        problems.append(f"tr() keys with no definition: {missing}")

    n_dep = deprecated_width_kwarg(src)
    if n_dep:
        problems.append(f"use_container_width passed as a kwarg {n_dep}x; use width='stretch'")

    if problems:
        print("STATIC INTEGRITY: FAILED")
        for p in problems:
            print("  -", p)
        return 1
    print("STATIC INTEGRITY: clean")
    print(f"  widget keys checked:  {len(re.findall(WIDGET_KEY_RE, src))}")
    print(f"  tr() call sites:      {len(set(re.findall(TR_KEY_RE, src)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
