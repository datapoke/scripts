#!/usr/bin/env python3
"""
Newspaper-order the methods of each Python class, so a reader meets an
entrypoint first and every helper below the code that calls it.

A port of newspack-nodes' `scripts/reorder-node-methods.php`, which holds
the orderings; `test-reorder-python` ports its fixtures.

A PHP class body is declarations; a Python class body is code that runs top
to bottom when the class is defined. So, unlike the PHP, this never moves a
field: a field's order is load-bearing (dataclass and NamedTuple fields,
Enum members and their `_generate_next_value_`, a `del` or rebinding of a
method, a default that names a module constant the field would shadow).
Every statement that is not a `def` is a barrier, and so is a method whose
decorator may register it in order (a validator, a route): methods reorder
only within each run of consecutive methods between barriers. For the same
reason the PHP's `--sort-fields` and `# @ordered` are not ported.

Member bodies are NEVER edited: each member moves as whole lines, and its
leading comments and decorators travel with it, as do comment lines
indented into its body after its last statement. Blank-line separators stay
in place, so spacing survives the reorder. Files are read and written as
bytes in their declared encoding, so line endings survive. Four invariants
are checked before any write: the multiset of member texts is unchanged,
the byte histogram of the whole file is unchanged, the output compiles, and
every class body holds the same statements. A mismatch aborts that file and
fails the run. A class whose methods name each other while the class is
being defined (a default, a decorator), or that defines a method twice, is
left alone and reported.

Node detection follows the JS twin: a class named `Node` or `*Node`, or
extending one, takes the node policy.

Two ordering policies, applied to each run:

  NODE — `__new__`, `__init__`, `arguments`, `fill`, `fire_cb`, `fire` (the
    node lifecycle in the order it runs), then the call-graph middle in
    topological order, then the methods no self-dispatch edge touches in
    source order, then `node_schema` last.

  GENERIC (every other class) — `__new__`, `__init__`, then every remaining
    method in topological order.

Topological order means a method is emitted only once EVERY caller of it
already is, so no caller ever prints below something it calls. A cycle
stalls the sort, and the members left over fall through in source order.

An edge is a self-dispatch within the run: `self.m`, `cls.m` or
`ClassName.m`, called or passed as a reference, anywhere in the method,
lambdas and nested `def`s included. Only module-level classes are
reordered; module functions and classes nested in functions keep their
order. A file under `tests/` or `__tests__/`, or named `test_*.py` or
`*_test.py`, is skipped whole.

Usage:

  reorder-python.py [--check|--write] <file.py> [...]

With no flag it reports what it would change and exits 0. `--check` reports
the same and exits 1. `--write` applies.
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys
import tempfile
import tokenize
from collections import Counter

PRIORITY = {
    "__new__": 0,
    "__init__": 1,
    "arguments": 2,
    "fill": 3,
    "fire_cb": 4,
    "fire": 5,
    "node_schema": 1000,
}
PREFIX = 6  # ranks below this are the node policy's fixed prefix
MIDDLE = 500
CONSTRUCTORS = ("__new__", "__init__")
# Decorators that only wrap the function; any other may register it in order.
INERT_DECORATORS = {
    "abstractmethod",
    "cache",
    "cached_property",
    "classmethod",
    "final",
    "lru_cache",
    "override",
    "property",
    "staticmethod",
}
LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


class LeftAlone(Exception):
    """A class that cannot be reordered safely; the message says why."""


class Member:
    """One statement of a class body: its lines, and what ordering needs."""

    def __init__(self, stmt: ast.stmt, first: int, last: int):
        self.stmt = stmt
        self.first = first  # 0-based line of its first decorator or keyword
        self.last = last  # 0-based last line, absorbed body comments included
        self.method = isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        self.name = stmt.name if self.method else ""
        self.movable = self.method and all(
            decorator_tail(d) in INERT_DECORATORS for d in stmt.decorator_list
        )


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    files = [a for a in argv if not a.startswith("--")]
    if not files or flags - {"--check", "--write"}:
        print(
            "usage: reorder-python.py [--check|--write] <file.py> [...]",
            file=sys.stderr,
        )
        return 1
    failed = False
    for f in files:
        if is_test_path(f):
            continue
        try:
            with open(f, "rb") as fh:
                data = fh.read()
            encoding = tokenize.detect_encoding(io.BytesIO(data).readline)[0]
            src = data.decode(encoding)
            out, notes, skipped = reorder(src)
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            print(f"✗ {f}: {e}", file=sys.stderr)
            failed = True
            continue
        for note in skipped:
            print(f"· {f}  {note}")
        if out == src:
            continue
        new = out.encode(encoding)
        if not invariants_hold(data, new, src, out):
            print(f"✗ {f}: INVARIANT VIOLATION — aborted", file=sys.stderr)
            failed = True
            continue
        if "--write" in flags and not write_atomic(f, new):
            print(f"✗ {f}: write failed", file=sys.stderr)
            failed = True
            continue
        print(f"~ {f}  " + "; ".join(notes))
        if "--check" in flags:
            failed = True
    return 1 if failed else 0


def is_test_path(f: str) -> bool:
    """Test code is left alone: its methods form no call graph worth
    ordering, and a double mirrors the order of the class it stands in for."""
    norm = "/" + f.replace("\\", "/")
    base = os.path.basename(norm)
    return (
        "/tests/" in norm
        or "/__tests__/" in norm
        or (base.startswith("test_") and base.endswith(".py"))
        or base.endswith("_test.py")
    )


def reorder(src: str):
    """Rewrite every module-level class body into convention order.

    Returns the new source, one note per class reordered, and one note per
    class left alone. Classes are rewritten bottom to top, which keeps the
    line numbers of the ones still to come valid. A class already in order,
    or with no run of two methods, is skipped silently.
    """
    # A last line with no terminator can't move up; lend it one, then take
    # it back from whatever line ends the file afterwards.
    lent = "" if not src or src.endswith(("\n", "\r")) else newline_of(src)
    src += lent
    tree = ast.parse(src)
    lines = split_lines(src)
    deferred = has_future_annotations(tree)
    headers = header_ends(src)
    notes: list[str] = []
    skipped: list[str] = []
    classes = [s for s in tree.body if isinstance(s, ast.ClassDef)]
    for cls in reversed(classes):
        try:
            result = reorder_class(cls, lines, headers, deferred)
        except LeftAlone as e:
            skipped.append(f"{cls.name}: left alone ({e})")
            continue
        if result is None:
            continue
        start, end, region, note = result
        lines[start : end + 1] = region
        notes.append(f"{cls.name} reordered: {note}")
    out = "".join(lines)
    out = out[: len(out) - len(lent)]
    return out, list(reversed(notes)), list(reversed(skipped))


def newline_of(src: str) -> str:
    """The file's first line terminator, or \n when it has none."""
    m = re.search(r"\r\n|\r|\n", src)
    return m.group() if m else "\n"


def split_lines(src: str) -> list[str]:
    """Lines as the parser counts them: \\r\\n, \\r or \\n, nothing else."""
    return LINE_RE.findall(src)


def reorder_class(cls, lines, headers, deferred):
    """The new lines for one class's member region, or None to leave it."""
    body = list(cls.body)
    if body and is_docstring(body[0]):
        region_start = body[0].end_lineno  # 0-based line after the docstring
        body = body[1:]
    else:
        region_start = headers[cls.lineno] + 1
    if len(body) < 2:
        return None
    members = scan_members(body, lines, region_start)
    check_definition_order(members, deferred)

    # Each chunk runs from the end of the member before it, so leading
    # comments travel; its leading blank lines stay behind as the separator.
    seps: list[list[str]] = []
    contents: list[list[str]] = []
    prev = region_start - 1
    for m in members:
        chunk = lines[prev + 1 : m.last + 1]
        k = 0
        while k < len(chunk) and not chunk[k].strip():
            k += 1
        seps.append(chunk[:k])
        contents.append(chunk[k:])
        prev = m.last

    policy = order_methods_node if is_node_class(cls) else order_methods_generic
    final: list[int] = []
    for run in runs(members):
        if len(run) < 2:
            final += run
            continue
        methods = [members[p] for p in run]
        final += [run[k] for k in policy(cls, methods)]
    if final == list(range(len(members))):
        return None
    region = [line for p, q in enumerate(final) for line in seps[p] + contents[q]]
    note = ", ".join(members[p].name or "(field)" for p in final)
    return region_start, members[-1].last, region, note


def runs(members: list[Member]) -> list[list[int]]:
    """Member positions, grouped: each run of movable methods together, and
    every barrier alone."""
    out: list[list[int]] = []
    for p, m in enumerate(members):
        if m.movable and out and members[out[-1][-1]].movable:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def header_ends(src: str) -> dict[int, int]:
    """For each `class` keyword's 1-based line, the 0-based line of the `:`
    that closes its header, from one tokenize pass over the file."""
    out: dict[int, int] = {}
    start = None
    depth = 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string == "class" and start is None:
            start, depth = tok.start[0], 0
        elif start is not None and tok.type == tokenize.OP:
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth -= 1
            elif tok.string == ":" and depth == 0:
                out[start] = tok.start[0] - 1
                start = None
    return out


def scan_members(body: list[ast.stmt], lines: list[str], region_start: int):
    members: list[Member] = []
    for i, stmt in enumerate(body):
        first = first_line(stmt) - 1
        last = stmt.end_lineno - 1
        nxt = first_line(body[i + 1]) - 1 if i + 1 < len(body) else None
        if first < region_start or (members and first <= members[-1].last):
            raise LeftAlone("two members share a line")
        if nxt is not None and nxt <= last:
            raise LeftAlone("two members share a line")
        last = absorb_body_comments(lines, stmt, last, nxt)
        members.append(Member(stmt, first, last))
    return members


def first_line(stmt: ast.stmt) -> int:
    """1-based line of a statement's first decorator, or of the statement."""
    return min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list", [])])


def absorb_body_comments(lines, stmt, last: int, stop) -> int:
    """Extend a member over the comment lines indented into its body after
    its last statement; otherwise they would ride with the next member."""
    indent = stmt.col_offset
    k = last + 1
    limit = len(lines) if stop is None else stop
    while k < limit:
        text = lines[k]
        if not text.strip():
            k += 1
            continue
        if text.lstrip().startswith("#") and len(text) - len(text.lstrip()) > indent:
            last = k
            k += 1
            continue
        break
    return last


def decorator_tail(expr: ast.expr) -> str:
    """`functools.lru_cache(maxsize=8)` → `lru_cache`; `x.setter` → `setter`."""
    if isinstance(expr, ast.Call):
        expr = expr.func
    return dotted_tail(expr)


def dotted_tail(expr: ast.expr) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return ""


def has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(s, ast.ImportFrom)
        and s.module == "__future__"
        and any(a.name == "annotations" for a in s.names)
        for s in tree.body
    )


def check_definition_order(members: list[Member], deferred: bool) -> None:
    """Refuse a class whose methods name each other while the body runs, or
    that defines a method twice: moving either would change what a name
    means or raise a NameError."""
    method_names = [m.name for m in members if m.method]
    dupes = sorted({n for n in method_names if method_names.count(n) > 1})
    if dupes:
        raise LeftAlone("definition-time reference: " + ", ".join(dupes) + " defined twice")
    methods = set(method_names)
    for m in members:
        if not m.method:
            continue
        hit = sorted(definition_time_names(m.stmt, deferred) & methods)
        if hit:
            raise LeftAlone(f"definition-time reference: {m.name} names {', '.join(hit)}")


def definition_time_names(fn: ast.FunctionDef, deferred: bool) -> set[str]:
    """The names a `def` reads while the class body runs."""
    args = fn.args
    roots: list[ast.AST] = list(fn.decorator_list)
    roots += args.defaults + [d for d in args.kw_defaults if d is not None]
    if not deferred:
        every = args.posonlyargs + args.args + args.kwonlyargs
        every += [a for a in (args.vararg, args.kwarg) if a is not None]
        roots += [a.annotation for a in every if a.annotation is not None]
        roots += [fn.returns] if fn.returns is not None else []
    return {
        n.id
        for root in roots
        for n in ast.walk(root)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def is_node_class(cls: ast.ClassDef) -> bool:
    names = [cls.name] + [dotted_tail(b) for b in cls.bases]
    return any(n == "Node" or n.endswith("Node") for n in names if n)


def callees(cls: ast.ClassDef, methods: list[Member], i: int) -> list[str]:
    """Method i's self-dispatched callees among `methods`, in
    first-appearance order."""
    fn = methods[i].stmt
    receivers = {cls.name}
    is_static = any(decorator_tail(d) == "staticmethod" for d in fn.decorator_list)
    positional = fn.args.posonlyargs + fn.args.args
    if positional and not is_static:
        receivers.add(positional[0].arg)
    names = {m.name for m in methods} - {fn.name}
    seen: dict[str, tuple[int, int]] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in receivers
            and node.attr in names
        ):
            at = (node.lineno, node.col_offset)
            if node.attr not in seen or at < seen[node.attr]:
                seen[node.attr] = at
    return sorted(seen, key=seen.get)


def order_methods_node(cls: ast.ClassDef, methods: list[Member]) -> list[int]:
    """NODE policy: the fixed prefix, the call-graph middle, the standalone
    methods, then `node_schema`."""
    index = {m.name: i for i, m in enumerate(methods)}
    prefix, middle, suffix = [], [], []
    for i, m in enumerate(methods):
        r = PRIORITY.get(m.name, MIDDLE)
        if r < PREFIX:
            prefix.append((r, i))
        elif r == PRIORITY["node_schema"]:
            suffix.append(i)
        else:
            middle.append(i)
    prefix_idx = [i for _, i in sorted(prefix)]
    middle_set = set(middle)

    # Prefix entrypoints are pre-placed, but still pull middle helpers in.
    edges: dict[int, list[int]] = {}
    indeg = {i: 0 for i in middle}
    called_by_prefix: set[int] = set()
    for i in prefix_idx + middle:
        cs = [index[n] for n in callees(cls, methods, i) if index[n] in middle_set]
        edges[i] = cs
        for j in cs:
            if i in middle_set:
                indeg[j] += 1  # only middle callers gate
            else:
                called_by_prefix.add(j)

    connected = {
        i for i in middle if edges[i] or indeg[i] > 0 or i in called_by_prefix
    }
    placed: list[int] = []
    visited: set[int] = set()
    remaining = [i for i in middle if i in connected]
    while True:
        avail = sorted(i for i in remaining if i not in visited and indeg[i] == 0)
        if not avail:
            break
        i = avail[0]
        visited.add(i)
        placed.append(i)
        for j in edges[i]:
            indeg[j] -= 1
    placed += [i for i in remaining if i not in visited]  # cycle
    standalone = [i for i in middle if i not in connected]
    return prefix_idx + placed + standalone + suffix


def order_methods_generic(cls: ast.ClassDef, methods: list[Member]) -> list[int]:
    """GENERIC policy: the constructors, then every remaining method in
    topological order.

    Ties prefer, in order: the method a caller just freed, so a chain stays
    together; then a public method that calls something, then any other
    public method; then source order.
    """
    index = {m.name: i for i, m in enumerate(methods)}
    ctor = sorted(
        (i for i, m in enumerate(methods) if m.name in CONSTRUCTORS),
        key=lambda i: CONSTRUCTORS.index(methods[i].name),
    )
    rest = [i for i in range(len(methods)) if i not in ctor]
    rest_set = set(rest)

    # The constructors gate nothing, so their callees sink below other callers.
    edges: dict[int, list[int]] = {}
    indeg = {i: 0 for i in rest}
    for i in ctor + rest:
        cs = [index[n] for n in callees(cls, methods, i) if index[n] in rest_set]
        edges[i] = cs
        if i in rest_set:
            for j in cs:
                indeg[j] += 1

    def rank(i: int) -> int:
        public = is_public(methods[i].name)
        return 0 if public and edges[i] else (1 if public else 2)

    placed: list[int] = []
    visited: set[int] = set()
    freed = {i: 0 for i in rest}
    tick = 0
    while True:
        avail = [i for i in rest if i not in visited and indeg[i] == 0]
        if not avail:
            break
        i = min(avail, key=lambda a: (-freed[a], rank(a), a))
        visited.add(i)
        placed.append(i)
        for j in edges[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                tick += 1
                freed[j] = tick
    placed += [i for i in rest if i not in visited]  # cycle
    return ctor + placed


def is_public(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def invariants_hold(data: bytes, new: bytes, src: str, out: str) -> bool:
    """Member texts, byte histogram, compilability and class contents."""
    try:
        compile(new, "<reordered>", "exec")
    except SyntaxError:
        return False
    return (
        fingerprint(src) == fingerprint(out)
        and Counter(data) == Counter(new)
        and shape(src) == shape(out)
    )


def fingerprint(src: str) -> list[str]:
    """Sorted texts of every member of every module-level class, each
    without its final terminator, which a file's last line may lack."""
    lines = split_lines(src)
    texts = []
    for cls in (s for s in ast.parse(src).body if isinstance(s, ast.ClassDef)):
        for stmt in cls.body:
            text = "".join(lines[first_line(stmt) - 1 : stmt.end_lineno])
            texts.append(re.sub(r"(?:\r\n|\r|\n)\Z", "", text))
    return sorted(texts)


def shape(src: str) -> list:
    """The module's statements, each class body as a multiset."""
    out = []
    for stmt in ast.parse(src).body:
        if isinstance(stmt, ast.ClassDef):
            body = sorted(ast.dump(s) for s in stmt.body)
            head = ast.dump(
                ast.ClassDef(
                    name=stmt.name,
                    bases=stmt.bases,
                    keywords=stmt.keywords,
                    body=[],
                    decorator_list=stmt.decorator_list,
                )
            )
            out.append((head, body))
        else:
            out.append(ast.dump(stmt))
    return out


def write_atomic(path: str, data: bytes) -> bool:
    """Write through a temp file beside the real file, then rename it into
    place; a symlink keeps pointing at the rewritten target."""
    path = os.path.realpath(path)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".reorder")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        os.replace(tmp, path)
    except OSError:
        if tmp is not None:
            os.unlink(tmp)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
