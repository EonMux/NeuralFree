"""Generates the data half of llama.cpp's Hugging Face converter as NeuralFree resources (src/main/resources/hf).

Usage: run only by the updateHfTables Gradle task (gradle/hf-tables.gradle):

    ./gradlew updateHfTables -PllamaCpp=<llama.cpp checkout>

which calls

    python3 -I gradle/hfgen/hfgen.py --llama-cpp <checkout> --out <directory> --commit <short sha>

The generator parses llama.cpp's Python sources with the ast module and never imports them, so it needs only the
standard library of Python 3.9 or newer (one that parses the checkout's syntax) and no llama.cpp dependency. The
output is deterministic: UTF-8, "\\n" line ends, rows in the source order stated per file. Any source construct the
generator does not recognise stops it with an error that names the file and line, so a changed converter is noticed
instead of being misread. Python source text in the tables (conditions, values, statements) is ast.unparse output:
its spelling can differ between Python versions, its meaning cannot.

Every TSV starts with the line
    # <file> - Generated from llama.cpp <commit> by the updateHfTables Gradle task; do not edit. Rows: <n>
and a "#" line with the TAB-separated column names. Fields are TAB-separated, "-" marks an empty field, and list
fields are comma-separated. Files (LC = the llama.cpp checkout, CV = LC/conversion):

archs.tsv             hf_name, kind, class, source, gguf_arch, modify, vocab_writers. One row per TEXT_MODEL_MAP name
                      then per MMPROJ_MODEL_MAP name (CV/__init__.py order). class is the @ModelBase.register class of
                      that kind (a class is MMPROJ when its model_arch resolves to MODEL_ARCH.MMPROJ), "-" when no
                      class of that kind registers the name. gguf_arch is MODEL_ARCH_NAMES of the class's model_arch
                      resolved through the C3 MRO; the conditional assignments of dynamic_archs.tsv may replace it at
                      run time. When that class attribute never takes effect, gguf_arch is instead the architecture
                      that an __init__ the converter runs (following super().__init__ and Class.__init__(self))
                      assigns on every path, or "-" when that choice, or the nested implementation class the class
                      delegates to, depends on the checkpoint (dynamic_archs.tsv lists the choices). modify is "own",
                      the ancestor that defines modify_tensors, or "-" for the ModelBase/TextModel/MmprojModel
                      default. vocab_writers lists, in call order, the _set_vocab_* writers that the effective
                      set_vocab reaches (following super(), Class.set_vocab(self), self.impl.set_vocab() and helper
                      methods), "<Class>.<method>" for every reached method that writes tokenizer keys itself, and
                      "target_model.set_vocab" where a draft model runs its target model's set_vocab; "-" for MMPROJ
                      rows.
dynamic_archs.tsv     class, source, condition, gguf_arch. Every "self.model_arch = ..." in a method of CV/*.py, and
                      every nested implementation class of a delegating class (mistral3.py), in (file, line) order.
                      condition is the converter's Python condition: the enclosing if/elif/else tests joined by "and",
                      written as in the source (walrus included; plain names are locals of the method), "-" when
                      unconditional. A conditional expression yields one row per outcome.
tensor_patterns.tsv   seq, scope, tensor, hf_template. gguf-py tensor_mapping.py mappings_cfg (scope global) then
                      block_mappings_cfg (scope block), in dict iteration order, one row per pattern; seq counts from
                      0. A tensor listed with no pattern has one row with hf_template "-": it still maps its own GGUF
                      name (TensorNameMap adds that identity key for every listed tensor of the arch).
tensor_names.tsv      tensor, gguf_template. constants.py TENSOR_NAMES in order.
arch_tensors.tsv      gguf_arch, tensor. constants.py MODEL_TENSORS in order (it lists a few tensors that have no
                      TENSOR_NAMES entry; TensorNameMap never maps those).
arch_block_overrides.tsv
                      gguf_arch, tensor, hf_template, seq. tensor_mapping.py arch_block_mappings_cfg in order; seq
                      counts from 0 per arch. An override replaces the block pattern list of its tensor in place, and
                      a tensor absent from block_mappings_cfg is appended in override order.
gguf_keys.tsv         writer_method, key_template, value_type. One row per (GGUFWriter.add_* method, key) the method
                      writes, in method order then first-write order. key_template is the Keys constant with its
                      placeholders ({arch}, {name}, ...) kept. value_type is the GGUF type (UINT8..FLOAT64, BOOL,
                      STRING), ARRAY:<element type>, or ANY when it depends on the value (several writer calls of
                      different types, or an element type neither the annotation nor the collected items fix).
                      Generic writers that take the key as an argument (add_uint32, add_key_value, _add_u64_array,
                      ...) and tensor writers have no row; their types are read from their add_key_value calls.
hparam_aliases.tsv    gguf_key_template, hf_aliases, value_type, rule. The config reads of TextModel.__init__ and
                      TextModel.set_gguf_parameters (CV/base.py) in source order. hf_aliases are config keys in
                      priority order (the first one present wins, as find_hparam). Keys read from the flattened config
                      are bare; keys read from a value bound by a "local" row are prefixed with its name: an alias
                      whose first dotted part names an earlier local row reads that local, and the bare name is the
                      local itself. rule is "<head>[; when <condition>][; value <expression>][; after <statement>]"
                      with head one of:
                        required        the first present alias is written; none present is an error
                        optional        written when an alias is present
                        always          written whenever the condition holds
                        default=<v>     an absent alias reads as v
                        local <name>    binds a value that later rows read (conditions name it)
                        mirror <path>   TextModel.__init__ copies the value into that rope_parameters path
                        reject          the converter raises the exception given as value when the condition holds
                      gguf_key_template and value_type are "-" for local, mirror and reject rows. Conditions, values
                      and statements are the converter's Python source, with walrus targets shown by name.
chkhsh.tsv            sha256, pre_name, repo. The chkhsh branches of TextModel.get_vocab_base_pre in source order;
                      repo is the Hugging Face repository of the branch's "# ref:" comment. The branches are
                      sequential ifs, so of two rows with the same sha256 the later one decides the pre-tokenizer.
chktxt.txt            The exact chktxt string of get_vocab_base_pre, UTF-8, no added newline.
filters.tsv           kind, value, replacement, source. rename_suffix / rename_contains (ModelBase.filter_tensors: a
                      name ending in / containing value has every occurrence of value replaced, "-" meaning the empty
                      string), strip_prefix (TextModel.filter_tensors removes value from the front before matching
                      the drop rows; the tensor keeps its name), drop_prefix / drop_contains (TextModel.filter_tensors
                      drops the tensor), skip_contains (MmprojModel.filter_tensors drops the tensor), skip_suffix
                      (prepare_tensors skips the tensor), forced_f32 (prepare_tensors stores tensors of this
                      MODEL_TENSOR as F32).
templates/*.jinja     Byte copies of every LC/models/templates file the converter names in CV/*.py.
vocab/*.gguf          Byte copies of every LC/models/ggml-vocab-<name>.gguf that a _set_vocab_builtin call names.
.gitignore, .gitattributes
                      Version-control settings, not resources (Gradle leaves both out of the jar): keep the vocabulary
                      GGUFs under version control despite the repository-wide "*.gguf" ignore rule, and keep every
                      file byte-exact (no end-of-line conversion), so the manifest hashes hold.
manifest.tsv          file, rows_or_bytes, sha256. Every other resource file (not the two settings files), sorted by
                      path; rows_or_bytes is "<n> rows" for a TSV and "<n> bytes" otherwise. Written last.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NONE = "-"
MIN_PYTHON = (3, 9)
BASE_CLASSES = ("ModelBase", "TextModel", "MmprojModel")
GGUF_SCALARS = {"int": "INT32", "float": "FLOAT32", "bool": "BOOL", "str": "STRING", "bytes": "STRING",
                "bytearray": "STRING", "TokenType": "INT32"}
GITIGNORE = ("# The built-in vocabularies are checked-in resources, unlike other GGUF files.\n"
             "!/vocab/*.gguf\n")
GITATTRIBUTES = ("# Every file is kept byte-exact: manifest.tsv records its sha256.\n"
                 "* -text\n")


class GenError(Exception):
    """A source construct the generator does not understand, or an output that breaks the table format."""


def fail(where: str, message: str) -> None:
    raise GenError(f"{where}: {message}")


# ----------------------------------------------------------------------------------------------------------------
# Source access and AST helpers


@dataclass(frozen=True)
class Module:
    path: str
    rel: str
    text: str
    tree: ast.Module

    def at(self, node: ast.AST) -> str:
        return f"{self.rel}:{getattr(node, 'lineno', 0)}"

    def line(self, number: int) -> str:
        return self.text.splitlines()[number - 1]


def load_module(root: str, rel: str) -> Module:
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        raise GenError(f"{path}: missing; is --llama-cpp a llama.cpp checkout?")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        raise GenError(f"{rel}: cannot read: {e}") from e
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as e:
        raise GenError(f"{rel}:{e.lineno}: cannot parse: {e.msg}") from e
    return Module(path, rel, text, tree)


def one_line(node: ast.AST, walrus: bool = False) -> str:
    """Python source of a node on one line; walrus targets are shown by name unless walrus is set."""
    class Unwalrus(ast.NodeTransformer):
        def visit_NamedExpr(self, n: ast.NamedExpr) -> ast.AST:
            return ast.copy_location(ast.Name(id=n.target.id, ctx=ast.Load()), n)
    copy = ast.parse(ast.unparse(node), mode="exec")
    if not walrus:
        copy = Unwalrus().visit(copy)
    return " ".join(part.strip() for part in ast.unparse(copy).splitlines())


def negate(test: ast.AST, walrus: bool = False) -> str:
    text = one_line(test, walrus)
    atomic = isinstance(test, (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.Constant)) or \
        (isinstance(test, ast.NamedExpr) and not walrus)
    return f"not {text}" if atomic else f"not ({text})"


def condition_text(test: ast.AST, walrus: bool = False) -> str:
    """A condition that stays correct when joined to others with 'and'."""
    text = one_line(test, walrus)
    loose = (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)) \
        or isinstance(test, (ast.IfExp, ast.Lambda)) \
        or (isinstance(test, ast.NamedExpr) and walrus and not text.startswith("("))
    return f"({text})" if loose else text


def dotted(node: ast.AST) -> Optional[str]:
    """'a.b.c' for a Name/Attribute chain, else None."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def member(node: ast.AST, owner: str) -> Optional[str]:
    """X for an expression 'owner.X' (owner may be dotted, e.g. 'gguf.MODEL_ARCH'), else None."""
    d = dotted(node)
    if d is None:
        return None
    for prefix in (owner + ".", owner.split(".")[-1] + "."):
        if d.startswith(prefix) and "." not in d[len(prefix):]:
            return d[len(prefix):]
    return None


def string(node: ast.AST, where: str) -> str:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        fail(where, f"expected a string literal, got {one_line(node)}")
    return node.value


def literal_dict(node: ast.AST, where: str) -> List[Tuple[ast.AST, ast.AST]]:
    """Items of a dict display with Python semantics: a repeated key keeps its first position and its last value."""
    if not isinstance(node, ast.Dict):
        fail(where, "expected a dict display")
    order: List[str] = []
    items: Dict[str, Tuple[ast.AST, ast.AST]] = {}
    for k, v in zip(node.keys, node.values):
        if k is None:
            fail(where, "dict unpacking (**) is not supported")
        key = ast.dump(k)
        if key not in items:
            order.append(key)
        items[key] = (k, v)
    return [items[k] for k in order]


def module_statements(body: List[ast.stmt]) -> Iterable[ast.stmt]:
    """Every module-level statement, including those nested in if/try/with blocks, but not function or class bodies."""
    for stmt in body:
        yield stmt
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for part in ("body", "orelse", "finalbody"):
                yield from module_statements(getattr(stmt, part, []) or [])
            for handler in getattr(stmt, "handlers", []):
                yield from module_statements(handler.body)


def module_assign(mod: Module, name: str) -> ast.AST:
    """The value of the single top-level assignment to name; any other module-level write to name is an error, so
    a table that the module extends after its display is noticed."""
    def writes(node: ast.AST) -> bool:
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return isinstance(node, ast.Name) and node.id == name

    value: Optional[ast.AST] = None
    for node in module_statements(mod.tree.body):
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
        if any(writes(t) for t in targets):
            if value is not None or not (isinstance(node, (ast.Assign, ast.AnnAssign)) and len(targets) == 1
                                         and isinstance(targets[0], ast.Name) and node.value is not None):
                fail(mod.at(node), f"{name} is changed after its top-level assignment")
            value = node.value
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) and writes(node.value.func.value):
            fail(mod.at(node), f"{name} is changed by {one_line(node.value)}")
    if value is None:
        raise GenError(f"{mod.rel}: no top-level assignment to {name}")
    return value


def class_def(body: Iterable[ast.stmt], name: str, where: str) -> ast.ClassDef:
    for node in body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise GenError(f"{where}: class {name} not found")


def method(cls: ast.ClassDef, name: str, where: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise GenError(f"{where}: {cls.name}.{name} not found")


def self_attr(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


# ----------------------------------------------------------------------------------------------------------------
# gguf-py constants and tensor mapping


@dataclass
class Constants:
    keys: Dict[str, str]                 # "Keys.LLM.CONTEXT_LENGTH" -> "{arch}.context_length"
    arch_names: Dict[str, str]           # "LLAMA" -> "llama", in MODEL_ARCH_NAMES order
    tensor_names: List[Tuple[str, str]]  # TENSOR_NAMES in order
    model_tensors: List[Tuple[str, List[str]]]


def read_constants(mod: Module) -> Constants:
    keys: Dict[str, str] = {}

    def collect(cls: ast.ClassDef, prefix: str) -> None:
        for node in cls.body:
            if isinstance(node, ast.ClassDef):
                collect(node, f"{prefix}.{node.name}")
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    fail(mod.at(node), "unsupported Keys assignment")
                keys[f"{prefix}.{node.targets[0].id}"] = string(node.value, mod.at(node))
            elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                fail(mod.at(node), "unsupported statement in class Keys")

    collect(class_def(mod.tree.body, "Keys", mod.rel), "Keys")
    arch_names: Dict[str, str] = {}
    node = module_assign(mod, "MODEL_ARCH_NAMES")
    for k, v in literal_dict(node, mod.at(node)):
        arch = member(k, "MODEL_ARCH") or fail(mod.at(k), "expected MODEL_ARCH.X")
        arch_names[arch] = string(v, mod.at(v))
    tensor_names: List[Tuple[str, str]] = []
    node = module_assign(mod, "TENSOR_NAMES")
    for k, v in literal_dict(node, mod.at(node)):
        tensor = member(k, "MODEL_TENSOR") or fail(mod.at(k), "expected MODEL_TENSOR.X")
        tensor_names.append((tensor, string(v, mod.at(v))))
    model_tensors: List[Tuple[str, List[str]]] = []
    node = module_assign(mod, "MODEL_TENSORS")
    for k, v in literal_dict(node, mod.at(node)):
        arch = member(k, "MODEL_ARCH") or fail(mod.at(k), "expected MODEL_ARCH.X")
        if not isinstance(v, ast.List):
            fail(mod.at(v), f"MODEL_TENSORS[{arch}] is not a list display")
        model_tensors.append((arch, [member(e, "MODEL_TENSOR") or fail(mod.at(e), "expected MODEL_TENSOR.X")
                                     for e in v.elts]))
    return Constants(keys, arch_names, tensor_names, model_tensors)


def arch_name(consts: Constants, arch: str, where: str) -> str:
    if arch not in consts.arch_names:
        fail(where, f"MODEL_ARCH.{arch} has no MODEL_ARCH_NAMES entry")
    return consts.arch_names[arch]


def pattern_table(mod: Module, node: ast.AST) -> List[Tuple[str, List[str]]]:
    rows: List[Tuple[str, List[str]]] = []
    for k, v in literal_dict(node, mod.at(node)):
        tensor = member(k, "MODEL_TENSOR") or fail(mod.at(k), "expected MODEL_TENSOR.X")
        if not isinstance(v, ast.Tuple):
            fail(mod.at(v), f"patterns of {tensor} are not a tuple display")
        rows.append((tensor, [string(e, mod.at(e)) for e in v.elts]))
    return rows


def read_tensor_mapping(mod: Module) -> Tuple[List[list], List[list]]:
    cls = class_def(mod.tree.body, "TensorNameMap", mod.rel)
    tables: Dict[str, ast.AST] = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            tables[node.target.id] = node.value
    for name in ("mappings_cfg", "block_mappings_cfg", "arch_block_mappings_cfg"):
        if name not in tables:
            fail(mod.rel, f"TensorNameMap.{name} not found")
    patterns: List[list] = []
    for scope, name in (("global", "mappings_cfg"), ("block", "block_mappings_cfg")):
        for tensor, templates in pattern_table(mod, tables[name]):
            for template in templates or [NONE]:
                patterns.append([str(len(patterns)), scope, tensor, template])
    overrides: List[list] = []
    node = tables["arch_block_mappings_cfg"]
    for k, v in literal_dict(node, mod.at(node)):
        arch = member(k, "MODEL_ARCH") or fail(mod.at(k), "expected MODEL_ARCH.X")
        seq = 0
        for tensor, templates in pattern_table(mod, v):
            for template in templates or [NONE]:
                overrides.append([arch, tensor, template, str(seq)])
                seq += 1
    return patterns, overrides


# ----------------------------------------------------------------------------------------------------------------
# GGUFWriter keys


def annotation_elements(ann: Optional[ast.AST]) -> List[Optional[str]]:
    """GGUF element types of every sequence alternative of a parameter annotation (None when not fixed)."""
    if ann is None:
        return [None]
    alternatives: List[ast.AST] = []

    def split(n: ast.AST) -> None:
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
            split(n.left)
            split(n.right)
        else:
            alternatives.append(n)

    split(ann)
    out: List[Optional[str]] = []
    for alt in alternatives:
        name = dotted(alt)
        if name in ("bytes", "bytearray"):
            out.append("UINT8")
        elif isinstance(alt, ast.Subscript) and dotted(alt.value) in ("Sequence", "list", "List", "tuple", "Tuple"):
            inner = alt.slice
            out.append(GGUF_SCALARS.get(dotted(inner) or ""))
    return out or [None]


def last_assignment(fn: ast.FunctionDef, name: str, before: int) -> Optional[ast.AST]:
    """The value most recently assigned to a local name before a line (a for target yields its iterable)."""
    found: List[Tuple[int, ast.AST]] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            found.append((n.lineno, n.value))
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == name:
            found.append((n.lineno, n.iter))
    found = [f for f in found if f[0] < before]
    return max(found, key=lambda f: f[0])[1] if found else None


def str_valued(node: ast.AST, fn: ast.FunctionDef, params: Dict[str, Optional[ast.AST]]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("join", "format"):
        return isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)
    if isinstance(node, ast.Name):
        if node.id in params:
            return params[node.id] is not None and dotted(params[node.id]) == "str"
        value = last_assignment(fn, node.id, node.lineno)
        return value is not None and str_valued(value, fn, params)
    return False


def array_element(value: ast.AST, fn: ast.FunctionDef) -> Optional[str]:
    """GGUF element type of an add_array value, from its annotation or the items added to it; None when not fixed."""
    params = {a.arg: a.annotation for a in fn.args.args + fn.args.kwonlyargs}
    if isinstance(value, ast.Call) and dotted(value.func) == "list" and len(value.args) == 1:
        value = value.args[0]
    if not isinstance(value, ast.Name):
        return None
    if value.id in params:
        types = set(annotation_elements(params[value.id]))
        return types.pop() if len(types) == 1 else None
    collected = last_assignment(fn, value.id, value.lineno)
    if isinstance(collected, ast.Call) and dotted(collected.func) in ("set", "list") and not collected.args:
        types = set()
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("add", "append")
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == value.id and len(n.args) == 1):
                types.add("STRING" if str_valued(n.args[0], fn, params) else None)
        return types.pop() if len(types) == 1 else None
    return None


def key_template(node: ast.AST, consts: Constants, fn: ast.FunctionDef, where: str) -> Optional[str]:
    """The Keys template a writer call names, or None when the key is a parameter of the writer method."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        node = node.func.value
    if isinstance(node, ast.Name):
        if node.id in {a.arg for a in fn.args.args}:
            return None
        value = last_assignment(fn, node.id, node.lineno)
        if value is None:
            fail(where, f"cannot resolve key variable {node.id}")
        return key_template(value, consts, fn, where)
    d = dotted(node)
    if d is not None and d in consts.keys:
        return consts.keys[d]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    fail(where, f"cannot resolve key expression {one_line(node)}")
    return None


def value_type(vtype: ast.AST, sub: Optional[ast.AST]) -> str:
    """GGUF type named by add_key_value's type arguments ('ARRAY' when the element type comes from the value)."""
    vt = member(vtype, "GGUFValueType")
    if vt is None:
        return "ANY"
    if vt != "ARRAY":
        return vt
    st = member(sub, "GGUFValueType") if sub is not None else None
    return f"ARRAY:{st}" if st else "ARRAY"


def generic_writers(cls: ast.ClassDef) -> Dict[str, str]:
    """GGUFWriter methods that write a key given as their first parameter, with the type they write it as."""
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    generic: Dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name, fn in methods.items():
            if name in generic or len(fn.args.args) < 2:
                continue
            key_param = fn.args.args[1].arg
            types = set()
            for n in ast.walk(fn):
                if not (isinstance(n, ast.Call) and n.args and dotted(n.args[0]) == key_param):
                    continue
                callee = self_attr(n.func)
                if callee == "add_key_value" and len(n.args) >= 3:
                    types.add(value_type(n.args[2], n.args[3] if len(n.args) > 3 else None))
                elif callee in generic:
                    types.add(generic[callee])
            if name == "add_key_value":
                types = {"ANY"}
            if types:
                generic[name] = types.pop() if len(types) == 1 else "ANY"
                changed = True
    return generic


def writer_keys(mod: Module, consts: Constants) -> List[list]:
    cls = class_def(mod.tree.body, "GGUFWriter", mod.rel)
    generic = generic_writers(cls)
    rows: List[list] = []
    for fn in cls.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("add_")):
            continue
        writes: Dict[str, List[str]] = {}
        order: List[str] = []
        keyless = fn.name in generic

        def record(key: Optional[str], vtype: str) -> None:
            if key is None:
                return
            if key not in writes:
                writes[key] = []
                order.append(key)
            writes[key].append(vtype)

        for n in sorted((x for x in ast.walk(fn) if isinstance(x, (ast.Call, ast.Assign))),
                        key=lambda x: (x.lineno, x.col_offset)):
            callee = self_attr(n.func) if isinstance(n, ast.Call) else None
            if callee in generic and len(n.args) >= 2:
                vtype = generic[callee]
                if callee == "add_key_value":
                    vtype = value_type(n.args[2], n.args[3] if len(n.args) > 3 else None) if len(n.args) > 2 \
                        else "ANY"
                if vtype == "ARRAY":
                    elem = array_element(n.args[1], fn)
                    vtype = f"ARRAY:{elem}" if elem else "ANY"
                record(key_template(n.args[0], consts, fn, mod.at(n)), vtype)
            elif (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript)
                  and isinstance(n.value, ast.Call) and dotted(n.value.func) == "GGUFValue"):
                vt = value_type(n.value.args[1], None) if len(n.value.args) > 1 else "ANY"
                record(key_template(n.targets[0].slice, consts, fn, mod.at(n)), vt)
        if not order and not keyless and fn.name not in ("add_tensor", "add_tensor_info"):
            fail(mod.at(fn), f"{fn.name} writes no key the generator recognises")
        for key in order:
            types = set(writes[key])
            rows.append([fn.name, key, types.pop() if len(types) == 1 else "ANY"])
    return rows


def writer_method_keys(rows: List[list]) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for method_name, key, vtype in rows:
        out.setdefault(method_name, []).append((key, vtype))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Converter classes


@dataclass
class ConvClass:
    name: str                  # qualified for nested classes: "Outer.Inner"
    module: Module
    node: ast.ClassDef
    bases: List[str]
    registered: List[str] = field(default_factory=list)
    nested: List["ConvClass"] = field(default_factory=list)
    mro: List[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        return f"{os.path.basename(self.module.rel)}:{self.node.lineno}"

    def methods(self) -> Dict[str, ast.FunctionDef]:
        return {n.name: n for n in self.node.body if isinstance(n, ast.FunctionDef)}

    def attr(self, name: str) -> Optional[ast.AST]:
        for n in self.node.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                return n
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name \
                    and n.value is not None:
                return n
        return None


class Classes:
    """Every class of LC/conversion/*.py with its C3 method resolution order."""

    def __init__(self, modules: List[Module]):
        self.by_name: Dict[str, ConvClass] = {}
        for mod in modules:
            for node in mod.tree.body:
                if isinstance(node, ast.ClassDef):
                    c = self._make(node, mod, node.name)
                    if node.name in self.by_name:
                        fail(mod.at(node), f"class {node.name} is also defined in {self.by_name[node.name].source}")
                    self.by_name[node.name] = c
                    for inner in node.body:
                        if isinstance(inner, ast.ClassDef):
                            nc = self._make(inner, mod, f"{node.name}.{inner.name}")
                            nc.bases = [f"{node.name}.{b}" if b in {i.name for i in node.body
                                                                       if isinstance(i, ast.ClassDef)} else b
                                        for b in nc.bases]
                            c.nested.append(nc)
                            self.by_name[nc.name] = nc
        for c in list(self.by_name.values()):
            c.mro = self._linearize(c.name, [])

    @staticmethod
    def _make(node: ast.ClassDef, mod: Module, name: str) -> ConvClass:
        bases = [(dotted(b) or fail(mod.at(node), f"unsupported base {one_line(b)}")).split(".")[-1]
                 for b in node.bases]
        registered: List[str] = []
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and dotted(d.func) == "ModelBase.register":
                registered.extend(string(a, mod.at(d)) for a in d.args)
        return ConvClass(name, mod, node, bases, registered)

    def _linearize(self, name: str, stack: List[str]) -> List[str]:
        if name in stack:
            raise GenError(f"class hierarchy cycle through {name}")
        c = self.by_name.get(name)
        if c is None:
            return [name]
        if c.mro:
            return c.mro
        seqs = [self._linearize(b, stack + [name]) for b in c.bases] + [list(c.bases)]
        out = [name]
        seqs = [list(s) for s in seqs if s]
        while seqs:
            for s in seqs:
                head = s[0]
                if not any(head in other[1:] for other in seqs):
                    break
            else:
                raise GenError(f"{c.source}: no consistent C3 method resolution order for {name}")
            out.append(head)
            seqs = [[x for x in s if x != head] for s in seqs]
            seqs = [s for s in seqs if s]
        return out

    def owner(self, name: str, method_name: str, after: Optional[str] = None) -> Optional[ConvClass]:
        mro = self.by_name[name].mro
        start = mro.index(after) + 1 if after is not None else 0
        for n in mro[start:]:
            c = self.by_name.get(n)
            if c is not None and method_name in c.methods():
                return c
        return None

    def model_arch(self, name: str) -> Tuple[Optional[str], Optional[ConvClass]]:
        for n in self.by_name[name].mro:
            c = self.by_name.get(n)
            node = c.attr("model_arch") if c is not None else None
            if node is not None:
                return member(node.value, "gguf.MODEL_ARCH") or fail(c.module.at(node), "model_arch is not "
                                                                       "gguf.MODEL_ARCH.X"), c
        return None, None

    def delegates(self, name: str) -> bool:
        """True for a class that picks a nested implementation class at run time (its own arch is unused)."""
        return any(n.attr("model_arch") is not None for n in self.by_name[name].nested)


def read_model_maps(mod: Module) -> Dict[str, List[Tuple[str, str]]]:
    maps: Dict[str, List[Tuple[str, str]]] = {}
    for kind, var in (("TEXT", "TEXT_MODEL_MAP"), ("MMPROJ", "MMPROJ_MODEL_MAP")):
        node = module_assign(mod, var)
        maps[kind] = [(string(k, mod.at(k)), string(v, mod.at(v))) for k, v in literal_dict(node, mod.at(node))]
    return maps


def writes_tokenizer_keys(fn: ast.FunctionDef) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and dotted(n.func.value) == "self.gguf_writer":
                return True
            if any(dotted(a) == "self.gguf_writer" for a in n.args):
                return True
    return False


def vocab_writers(classes: Classes, cls: str) -> List[str]:
    found: List[str] = []
    visiting: set = set()

    def add(item: str) -> None:
        if item not in found:
            found.append(item)

    def visit(start: str, method_name: str, lookup: Optional[str] = None, after: Optional[str] = None) -> None:
        """Follows method_name of an instance of start, resolved through the MRO of lookup (default start) after
        the class after; self calls inside it dispatch on start."""
        owner = classes.owner(lookup or start, method_name, after)
        if owner is None:
            fail(classes.by_name[start].source, f"{lookup or start} has no {method_name} in its MRO")
        key = (start, owner.name, method_name)
        if key in visiting:
            return
        visiting.add(key)
        fn = owner.methods()[method_name]
        if writes_tokenizer_keys(fn):
            add(f"{owner.name}.{method_name}")
        assigned = {t.id: n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
        calls = sorted((n for n in ast.walk(fn) if isinstance(n, ast.Call)),
                       key=lambda n: (n.lineno, n.col_offset))
        for call in calls:
            f = call.func
            if not isinstance(f, ast.Attribute):
                continue
            target = f.value
            if isinstance(target, ast.Name) and target.id == "self":
                if f.attr.startswith("_set_vocab_"):
                    add(f.attr)
                elif classes.owner(start, f.attr) is not None and f.attr != method_name:
                    visit(start, f.attr)
            elif isinstance(target, ast.Call) and dotted(target.func) == "super":
                if owner.name not in classes.by_name[start].mro:
                    fail(owner.module.at(call), f"super() in {owner.name} runs on {start}, not a subclass of it")
                visit(start, f.attr, after=owner.name)
            elif dotted(target) == "self.impl" and f.attr == method_name:
                for inner in classes.by_name[owner.name.split(".")[0]].nested:
                    visit(inner.name, method_name)
            elif isinstance(target, ast.Name) and f.attr == method_name and call.args \
                    and dotted(call.args[0]) == "self":
                if target.id in classes.by_name:
                    visit(start, method_name, lookup=target.id)
                elif isinstance(assigned.get(target.id), ast.Call) \
                        and dotted(assigned[target.id].func) == "get_model_class":
                    add(f"target_model.{method_name}")
                else:
                    fail(owner.module.at(call), f"cannot resolve the class of {one_line(call)}")

    visit(cls, "set_vocab")
    return found


def archs_table(maps: Dict[str, List[Tuple[str, str]]], classes: Classes, consts: Constants) -> List[list]:
    registered: Dict[str, Dict[str, ConvClass]] = {"TEXT": {}, "MMPROJ": {}}
    for c in classes.by_name.values():
        if not c.registered:
            continue
        arch, _ = classes.model_arch(c.name)
        kind = "MMPROJ" if arch == "MMPROJ" else "TEXT"
        for hf_name in c.registered:
            if hf_name in registered[kind] and registered[kind][hf_name] is not c:
                fail(c.source, f"{hf_name} is registered by {registered[kind][hf_name].name} and {c.name}")
            registered[kind][hf_name] = c
    rows: List[list] = []
    for kind in ("TEXT", "MMPROJ"):
        for hf_name, module_name in maps[kind]:
            c = registered[kind].get(hf_name)
            if c is None:
                rows.append([hf_name, kind, NONE, NONE, NONE, NONE, NONE])
                continue
            if os.path.splitext(os.path.basename(c.module.rel))[0] != module_name:
                fail(c.source, f"{hf_name} maps to module {module_name} but {c.name} is in {c.module.rel}")
            arch, arch_owner = classes.model_arch(c.name)
            if arch is None:
                fail(c.source, f"{c.name} has no model_arch")
            gguf_arch = runtime_arch(classes, c.name, consts) or arch_name(consts, arch, arch_owner.source)
            mt = classes.owner(c.name, "modify_tensors")
            modify = NONE if mt is None or mt.name in BASE_CLASSES else ("own" if mt.name == c.name else mt.name)
            vocab = ",".join(vocab_writers(classes, c.name)) if kind == "TEXT" else NONE
            rows.append([hf_name, kind, c.name, c.source, gguf_arch, modify, vocab or NONE])
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Runtime architecture choices


def resolve_arch_value(value: ast.AST, fn: ast.FunctionDef, before: int, where: str) -> List[Tuple[Optional[str], str]]:
    """(condition or None, MODEL_ARCH member) outcomes of an expression assigned to model_arch."""
    arch = member(value, "gguf.MODEL_ARCH")
    if arch is not None:
        return [(None, arch)]
    if isinstance(value, ast.IfExp):
        outcomes: List[Tuple[Optional[str], str]] = []
        for test, branch in ((condition_text(value.test, True), value.body), (negate(value.test, True), value.orelse)):
            for inner, arch in resolve_arch_value(branch, fn, before, where):
                outcomes.append((test if inner is None else f"{test} and {inner}", arch))
        return outcomes
    if isinstance(value, ast.Name):
        prior = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.lineno < before
                 and any(isinstance(t, ast.Name) and t.id == value.id for t in n.targets)]
        if len(prior) > 1:
            fail(where, f"{value.id} is assigned more than once before it is stored in model_arch")
        if prior:
            return resolve_arch_value(prior[0].value, fn, prior[0].lineno, where)
    fail(where, f"cannot resolve the architecture of {one_line(value)}")
    return []


def always_assigns_arch(body: List[ast.stmt]) -> bool:
    """True when every path through a body that does not raise assigns self.model_arch."""
    for stmt in body:
        if isinstance(stmt, ast.Assign) and any(self_attr(t) == "model_arch" for t in stmt.targets):
            return True
        if isinstance(stmt, ast.Raise):
            return True
        if isinstance(stmt, ast.If) and always_assigns_arch(stmt.body) and always_assigns_arch(stmt.orelse):
            return True
        if isinstance(stmt, ast.With) and always_assigns_arch(stmt.body):
            return True
    return False


def init_chain(classes: Classes, name: str) -> List[ConvClass]:
    """The classes whose __init__ runs when the converter instantiates a class: the __init__ resolved through its
    MRO, then every super().__init__ and Class.__init__(self, ...) that __init__ calls, transitively."""
    chain: List[ConvClass] = []
    mro = classes.by_name[name].mro

    def visit(owner: Optional[ConvClass]) -> None:
        if owner is None or owner in chain:
            return
        chain.append(owner)
        fn = owner.methods()["__init__"]
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "__init__"):
                continue
            target = call.func.value
            if isinstance(target, ast.Call) and dotted(target.func) == "super":
                visit(classes.owner(name, "__init__", owner.name))
            elif isinstance(target, ast.Name) and call.args and dotted(call.args[0]) == "self":
                if target.id not in mro:
                    fail(owner.module.at(call), f"{target.id}.__init__ is not a base of {name}")
                visit(classes.owner(target.id, "__init__"))

    visit(classes.owner(name, "__init__"))
    return chain


def runtime_arch(classes: Classes, name: str, consts: Constants) -> Optional[str]:
    """gguf_arch of a class whose model_arch class attribute never takes effect, because the converter picks a
    nested implementation class or an __init__ it runs assigns self.model_arch on every path: "-" when that choice
    depends on the checkpoint, else the one architecture assigned. None when the class attribute stays the default
    (conditional assignments, listed in dynamic_archs.tsv, may still replace it)."""
    chain = init_chain(classes, name)
    if classes.delegates(name) or any(classes.delegates(c.name) for c in chain):
        return NONE
    assigned = {arch for c in chain if always_assigns_arch(c.methods()["__init__"].body)
                for _, _, arch in arch_assignments(c, c.methods()["__init__"], consts)}
    if not assigned:
        return None
    return assigned.pop() if len(assigned) == 1 else NONE


def conditional_statements(body: List[ast.stmt], path: List[str]) -> Iterable[Tuple[ast.stmt, List[str]]]:
    """Every statement of a body with the conditions of its enclosing if/elif/else chain."""
    for stmt in body:
        yield stmt, path
        if isinstance(stmt, ast.If):
            yield from conditional_statements(stmt.body, path + [condition_text(stmt.test, True)])
            yield from conditional_statements(stmt.orelse, path + [negate(stmt.test, True)])
        elif isinstance(stmt, (ast.For, ast.While, ast.With, ast.Try)):
            for part in ("body", "orelse", "finalbody"):
                yield from conditional_statements(getattr(stmt, part, []) or [], path)
            for handler in getattr(stmt, "handlers", []):
                yield from conditional_statements(handler.body, path)


def join_conditions(parts: List[Optional[str]]) -> str:
    kept = [p for p in parts if p]
    return " and ".join(kept) if kept else NONE


def arch_assignments(c: ConvClass, fn: ast.FunctionDef, consts: Constants) -> List[Tuple[int, str, str]]:
    """(line, condition, gguf_arch) of every outcome of every self.model_arch assignment in a method."""
    out: List[Tuple[int, str, str]] = []
    for stmt, path in conditional_statements(fn.body, []):
        if not (isinstance(stmt, ast.Assign) and any(self_attr(t) == "model_arch" for t in stmt.targets)):
            continue
        where = c.module.at(stmt)
        for cond, arch in resolve_arch_value(stmt.value, fn, stmt.lineno, where):
            out.append((stmt.lineno, join_conditions(path + [cond]), arch_name(consts, arch, where)))
    return out


def dynamic_archs(classes: Classes, consts: Constants) -> List[Tuple[str, int, list]]:
    rows: List[Tuple[str, int, list]] = []
    for c in classes.by_name.values():
        for fn in c.methods().values():
            for line, condition, arch in arch_assignments(c, fn, consts):
                rows.append((c.module.rel, line, [c.name, f"{os.path.basename(c.module.rel)}:{line}", condition, arch]))
        if not classes.delegates(c.name):
            continue
        for inner in c.nested:
            node = inner.attr("model_arch")
            if node is None:
                continue
            where = inner.module.at(node)
            arch = member(node.value, "gguf.MODEL_ARCH") or fail(where, "model_arch is not gguf.MODEL_ARCH.X")
            short = inner.name.split(".")[-1]
            sites = []
            for fn in c.methods().values():
                for stmt, path in conditional_statements(fn.body, []):
                    if isinstance(stmt, ast.Assign) and any(self_attr(t) == "impl" for t in stmt.targets) \
                            and isinstance(stmt.value, ast.Call) \
                            and dotted(stmt.value.func) in (short, f"{c.name}.{short}"):
                        sites.append(path)
            if not sites:
                fail(where, f"{inner.name} declares model_arch but {c.name} never selects it as self.impl")
            for path in sites:
                rows.append((inner.module.rel, node.lineno,
                             [c.name, f"{os.path.basename(inner.module.rel)}:{node.lineno}", join_conditions(path),
                              arch_name(consts, arch, where)]))
    rows.sort(key=lambda r: (r[0], r[1], r[2][2]))
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Generic hparams of TextModel


@dataclass(eq=False)
class Binding:
    """What a local name or self attribute holds: config aliases (a read) or a Python value."""
    aliases: List[str]
    head: str             # required | optional | always | default=<v>
    value: Optional[str]  # source of a value that is not a config read
    line: int
    conds: List[ast.AST]
    after: List[str] = field(default_factory=list)
    ambiguous: bool = False


class HparamWalker:
    """Follows TextModel.__init__ and set_gguf_parameters and records one row per config-driven write."""

    def __init__(self, mod: Module, writer: Dict[str, List[Tuple[str, str]]]):
        self.mod = mod
        self.writer = writer
        self.env: Dict[str, Binding] = {}
        self.rows: List[Tuple[int, int, list]] = []
        self.locals: Dict[int, Tuple[str, Binding]] = {}

    @staticmethod
    def bound_name(node: ast.AST) -> Optional[str]:
        d = dotted(node.target) if isinstance(node, ast.NamedExpr) else dotted(node)
        if d is None:
            return None
        return d[len("self."):] if d.startswith("self.") else d

    def binding(self, node: ast.AST) -> Optional[Binding]:
        name = self.bound_name(node)
        b = self.env.get(name) if name is not None else None
        if b is not None and b.ambiguous:
            fail(self.mod.at(node), f"{name} is bound differently on the branches of an if")
        return b

    def refer(self, b: Binding, name: str) -> None:
        if b.aliases:
            self.locals[id(b)] = (name, b)

    # reads ---------------------------------------------------------------------------------------------------

    def prefix(self, node: ast.AST) -> Optional[str]:
        """Alias prefix of a dictionary expression: "" for the config, "<name>." for a bound dictionary."""
        if dotted(node) == "self.hparams":
            return ""
        b = self.binding(node) if isinstance(node, (ast.Name, ast.Attribute)) else None
        if b is not None and b.aliases:
            name = self.bound_name(node)
            self.refer(b, name)
            return name + "."
        read = self.read(node) if isinstance(node, ast.Call) else None
        if read is not None and len(read.aliases) == 1:
            return read.aliases[0] + "."
        return None

    def read(self, node: ast.AST) -> Optional[Binding]:
        """The config keys an expression reads in priority order, or None when it is not a config read."""
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Call) and dotted(node.func) == "self.find_hparam":
            if not node.args or not isinstance(node.args[0], (ast.List, ast.Tuple)):
                fail(self.mod.at(node), "find_hparam without a literal key list")
            optional = any(k.arg == "optional" and isinstance(k.value, ast.Constant) and k.value.value is True
                           for k in node.keywords)
            return Binding([string(e, self.mod.at(e)) for e in node.args[0].elts],
                           "optional" if optional else "required", None, line, [])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and node.args:
            prefix = self.prefix(node.func.value)
            if prefix is None:
                return None
            aliases = [prefix + string(node.args[0], self.mod.at(node))]
            if len(node.args) == 1:
                return Binding(aliases, "optional", None, line, [])
            fallback = node.args[1]
            if isinstance(fallback, (ast.Constant, ast.Dict)):
                return Binding(aliases, f"default={one_line(fallback)}", None, line, [])
            sub = self.read(fallback) if isinstance(fallback, ast.Call) else None
            if sub is not None:
                return Binding(aliases + sub.aliases, sub.head, None, line, [])
            whole = self.prefix(fallback)
            if whole:
                return Binding(aliases + [whole.rstrip(".")], "required", None, line, [])
            fail(self.mod.at(node), f"unsupported default {one_line(fallback)}")
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            prefix = self.prefix(node.value)
            if prefix is not None:
                return Binding([prefix + string(node.slice, self.mod.at(node))], "required", None, line, [])
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and len(node.values) == 2 \
                and isinstance(node.values[1], (ast.Dict, ast.Constant)):
            sub = self.read(node.values[0])
            if sub is not None:
                return Binding(sub.aliases, f"default={one_line(node.values[1])}", None, line, [])
        return None

    def inline_reads(self, node: ast.AST) -> List[str]:
        """Aliases that an expression reads directly, in source order; walrus bindings have their own local rows."""
        bound = {id(x) for w in ast.walk(node) if isinstance(w, ast.NamedExpr) for x in ast.walk(w.value)}
        out: List[str] = []
        for n in sorted((x for x in ast.walk(node) if isinstance(x, ast.Call) and id(x) not in bound),
                        key=lambda x: (x.lineno, x.col_offset)):
            r = self.read(n)
            if r is not None:
                out.extend(a for a in r.aliases if a not in out)
        return out

    def referenced(self, node: ast.AST) -> List[Tuple[str, Binding]]:
        out: List[Tuple[str, Binding]] = []
        for n in ast.walk(node):
            if isinstance(n, (ast.Name, ast.Attribute, ast.NamedExpr)):
                name = self.bound_name(n)
                b = self.binding(n) if name in self.env else None
                if b is not None and b.aliases and all(b is not x for _, x in out):
                    out.append((name, b))
        return out

    # statements ----------------------------------------------------------------------------------------------

    def bind(self, name: str, value: ast.AST, line: int, conds: List[ast.AST]) -> None:
        r = self.read(value)
        if r is None:
            self.env[name] = Binding([], "always", one_line(value), line, list(conds))
        else:
            self.env[name] = Binding(r.aliases, r.head, None, line, list(conds))

    def walk(self, body: List[ast.stmt], conds: List[ast.AST]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.If):
                for n in ast.walk(stmt.test):
                    if isinstance(n, ast.NamedExpr):
                        self.bind(n.target.id, n.value, stmt.lineno, conds)
                before = dict(self.env)
                self.walk(stmt.body, conds + [stmt.test])
                taken = self.env
                self.env = dict(before)
                self.walk(stmt.orelse, conds + [ast.UnaryOp(op=ast.Not(), operand=stmt.test)])
                for name in set(taken) | set(self.env):
                    if taken.get(name) is not self.env.get(name):
                        b = taken.get(name) or self.env.get(name)
                        self.env[name] = Binding(b.aliases, b.head, b.value, b.line, b.conds, b.after, True)
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                self.assign(stmt, conds)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self.call(stmt, conds)
            elif isinstance(stmt, (ast.While, ast.For)):
                for _, b in self.referenced(stmt):
                    b.after.append(one_line(stmt))
            elif isinstance(stmt, ast.Raise) and stmt.exc is not None:
                self.emit(stmt.lineno, NONE, [], NONE, "reject", conds, None, one_line(stmt.exc), [])
            elif not isinstance(stmt, ast.Pass):
                fail(self.mod.at(stmt), f"unsupported statement {one_line(stmt)}")

    def assign(self, stmt: ast.Assign, conds: List[ast.AST]) -> None:
        target = stmt.targets[0]
        if dotted(target) == "self.hparams":
            return
        if isinstance(target, ast.Subscript) and dotted(target.value) == "self.rope_parameters":
            self.mirror(stmt, conds)
            return
        name = self.bound_name(target) if isinstance(target, (ast.Name, ast.Attribute)) else None
        if name is None:
            fail(self.mod.at(stmt), f"unsupported assignment {one_line(stmt)}")
        self.bind(name, stmt.value, stmt.lineno, conds)

    def mirror(self, stmt: ast.Assign, conds: List[ast.AST]) -> None:
        base = "rope_parameters." + string(stmt.targets[0].slice, self.mod.at(stmt))
        pairs = [(f"{base}.{string(k, self.mod.at(stmt))}", v) for k, v in zip(stmt.value.keys, stmt.value.values)] \
            if isinstance(stmt.value, ast.Dict) else [(base, stmt.value)]
        for target, value in pairs:
            b = self.binding(value) if isinstance(value, ast.Name) else None
            if b is None or not b.aliases:
                fail(self.mod.at(stmt), f"unsupported rope_parameters store {one_line(stmt)}")
            self.emit(stmt.lineno, NONE, b.aliases, NONE, f"mirror {target}", conds, value.id, None, b.after)

    def call(self, stmt: ast.Expr, conds: List[ast.AST]) -> None:
        call = stmt.value
        if not (isinstance(call.func, ast.Attribute) and dotted(call.func.value) == "self.gguf_writer"):
            return
        keys = self.writer.get(call.func.attr)
        if not keys or len(keys) != 1:
            fail(self.mod.at(stmt), f"{call.func.attr} does not write exactly one key")
        if len(call.args) != 1 or call.keywords:
            fail(self.mod.at(stmt), f"unsupported writer call {one_line(call)}")
        key, vtype = keys[0]
        arg = call.args[0]
        b = self.binding(arg) if isinstance(arg, (ast.Name, ast.Attribute)) else None
        if b is not None and b.value is not None:
            self.emit(stmt.lineno, key, [], vtype, "always", conds, None, b.value, [])
            return
        if b is None:
            inline = self.read(arg)
            if inline is not None:
                self.emit(stmt.lineno, key, inline.aliases, vtype, inline.head, conds, None, None, [])
                return
            used = self.referenced(arg)
            if len(used) == 1:
                name, b = used[0]
                self.emit(stmt.lineno, key, b.aliases, vtype, b.head, conds, name, one_line(arg), b.after)
                return
            self.emit(stmt.lineno, key, [], vtype, "always", conds, None, one_line(arg), [])
            return
        self.emit(stmt.lineno, key, b.aliases, vtype, b.head, conds, self.bound_name(arg), None, b.after)

    def condition(self, cond: ast.AST, value_name: Optional[str]) -> List[str]:
        """Condition parts of one enclosing test, without the presence test of the written value."""
        if isinstance(cond, ast.UnaryOp) and isinstance(cond.op, ast.Not):
            return [negate(cond.operand)]
        parts = cond.values if isinstance(cond, ast.BoolOp) and isinstance(cond.op, ast.And) else [cond]
        out = []
        for p in parts:
            presence = isinstance(p, ast.Compare) and len(p.ops) == 1 and isinstance(p.ops[0], ast.IsNot) \
                and isinstance(p.comparators[0], ast.Constant) and p.comparators[0].value is None \
                and value_name is not None and self.bound_name(p.left) == value_name
            if not presence:
                out.append(condition_text(p))
        return out

    def emit(self, line: int, key: str, aliases: List[str], vtype: str, head: str, conds: List[ast.AST],
             value_name: Optional[str], value: Optional[str], after: List[str]) -> None:
        aliases = list(aliases)
        parts: List[str] = []
        depends = bool(aliases)
        for cond in conds:
            parts.extend(self.condition(cond, value_name))
            for name, b in self.referenced(cond):
                if name != value_name:
                    self.refer(b, name)
                    depends = True
            if head == "always":
                aliases.extend(a for a in self.inline_reads(cond) if a not in aliases)
        if value is not None:
            for name, b in self.referenced(ast.parse(value, mode="eval")):
                if name != value_name:
                    self.refer(b, name)
                    depends = True
        if not (depends or aliases):
            return
        rule = head + (f"; when {' and '.join(parts)}" if parts else "") + (f"; value {value}" if value else "") \
            + (f"; after {' ; '.join(after)}" if after else "")
        self.rows.append((line, 1, [key, ",".join(aliases) or NONE, vtype, rule]))

    def finish(self) -> List[list]:
        for name, b in self.locals.values():
            parts = [p for c in b.conds for p in self.condition(c, None)]
            head = f"local {name}" + (f"; {b.head}" if b.head.startswith("default=") else "")
            rule = head + (f"; when {' and '.join(parts)}" if parts else "")
            self.rows.append((b.line, 0, [NONE, ",".join(b.aliases), NONE, rule]))
        self.rows.sort(key=lambda r: (r[0], r[1]))
        return [r[2] for r in self.rows]


def hparam_aliases(mod: Module, writer: Dict[str, List[Tuple[str, str]]]) -> List[list]:
    text_model = class_def(mod.tree.body, "TextModel", mod.rel)
    walker = HparamWalker(mod, writer)
    walker.walk(method(text_model, "__init__", mod.rel).body, [])
    walker.walk(method(text_model, "set_gguf_parameters", mod.rel).body, [])
    return walker.finish()


# ----------------------------------------------------------------------------------------------------------------
# Pre-tokenizer hashes and tensor filters


def find_function(mod: Module, name: str) -> ast.FunctionDef:
    for n in ast.walk(mod.tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise GenError(f"{mod.rel}: function {name} not found")


def chkhsh_table(base: Module, update: Module) -> Tuple[List[list], str]:
    fn = find_function(base, "get_vocab_base_pre")
    chktxt = None
    rows: List[list] = []
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and dotted(stmt.targets[0]) == "chktxt":
            chktxt = string(stmt.value, base.at(stmt))
        if not (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare) and dotted(stmt.test.left) == "chkhsh"):
            continue
        digest = string(stmt.test.comparators[0], base.at(stmt))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(base.at(stmt), f"'{digest}' is not a sha256 hex digest")
        res = [s for s in stmt.body if isinstance(s, ast.Assign) and dotted(s.targets[0]) == "res"]
        if len(res) != 1:
            fail(base.at(stmt), "a chkhsh branch must assign res exactly once")
        name = string(res[0].value, base.at(res[0]))
        repo = NONE
        for number in range(stmt.lineno + 1, res[0].lineno):
            m = re.match(r"\s*#\s*ref:\s*(\S+)", base.line(number))
            if m:
                repo = re.sub(r"^https://huggingface\.co/", "", m.group(1)).rstrip("/")
        rows.append([digest, name, repo])
    if chktxt is None:
        fail(base.at(fn), "get_vocab_base_pre assigns no chktxt literal")
    update_text = string(module_assign(update, "CHK_TXT"), update.rel)
    if update_text != chktxt:
        fail(update.rel, "CHK_TXT differs from the chktxt of get_vocab_base_pre")
    return rows, chktxt


def str_tuple(node: ast.AST, where: str) -> List[Tuple[str, int]]:
    if isinstance(node, ast.Tuple):
        return [(string(e, where), e.lineno) for e in node.elts]
    return [(string(node, where), node.lineno)]


def name_test(node: ast.AST, where: str) -> Tuple[str, List[Tuple[str, int]]]:
    """('suffix'|'prefix'|'contains', values) for name.endswith(...), name.startswith(...) or '...' in name."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and dotted(node.func.value) == "name" \
            and node.func.attr in ("endswith", "startswith") and len(node.args) == 1:
        return ("suffix" if node.func.attr == "endswith" else "prefix"), str_tuple(node.args[0], where)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.In) \
            and dotted(node.comparators[0]) == "name":
        return "contains", [(string(node.left, where), node.left.lineno)]
    fail(where, f"unsupported name test {one_line(node)}")
    return "", []


def filters_table(base: Module) -> List[list]:
    rows: List[list] = []
    file_name = os.path.basename(base.rel)

    def statements(cls_name: str) -> List[ast.stmt]:
        """The name tests of a filter_tensors method, without its 'name, gen = item' unpacking and final return."""
        fn = method(class_def(base.tree.body, cls_name, base.rel), "filter_tensors", base.rel)
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if not (body and isinstance(body[0], ast.Assign) and dotted(body[0].value) == "item"
                and isinstance(body[-1], ast.Return)):
            fail(base.at(fn), f"{cls_name}.filter_tensors does not start with 'name, gen = item' and end with a return")
        return body[1:-1]

    for stmt in statements("ModelBase"):
        if not isinstance(stmt, ast.If) or len(stmt.body) != 1:
            fail(base.at(stmt), "unsupported ModelBase.filter_tensors statement")
        how, values = name_test(stmt.test, base.at(stmt))
        repl = stmt.body[0]
        if not (isinstance(repl, ast.Assign) and isinstance(repl.value, ast.Call)
                and isinstance(repl.value.func, ast.Attribute) and repl.value.func.attr == "replace"):
            fail(base.at(repl), "expected name = name.replace(...)")
        old = string(repl.value.args[0], base.at(repl))
        new = string(repl.value.args[1], base.at(repl))
        if [v for v, _ in values] != [old] or how not in ("suffix", "contains"):
            fail(base.at(stmt), "the rename test and the replaced text differ")
        rows.append([f"rename_{how}", old, new or NONE, f"{file_name}:{stmt.lineno}"])

    for stmt in statements("TextModel"):
        if not isinstance(stmt, ast.If):
            fail(base.at(stmt), "unsupported TextModel.filter_tensors statement")
        body = stmt.body[0]
        if isinstance(body, ast.Assign) and dotted(body.targets[0]) == "name":
            how, values = name_test(stmt.test, base.at(stmt))
            if how != "prefix" or len(values) != 1 or not isinstance(body.value, ast.Subscript):
                fail(base.at(stmt), "expected a prefix strip")
            rows.append(["strip_prefix", values[0][0], NONE, f"{file_name}:{stmt.lineno}"])
        elif isinstance(body, ast.Return) and isinstance(body.value, ast.Constant) and body.value.value is None:
            tests = stmt.test.values if isinstance(stmt.test, ast.BoolOp) and isinstance(stmt.test.op, ast.Or) \
                else [stmt.test]
            for t in tests:
                how, values = name_test(t, base.at(stmt))
                if how == "suffix":
                    fail(base.at(t), "unexpected suffix drop")
                rows.extend([f"drop_{how}", v, NONE, f"{file_name}:{line}"] for v, line in values)
        else:
            fail(base.at(stmt), "unsupported TextModel.filter_tensors branch")

    for stmt in statements("MmprojModel"):
        if not (isinstance(stmt, ast.If) and isinstance(stmt.body[0], ast.Return)):
            fail(base.at(stmt), "unsupported MmprojModel.filter_tensors statement")
        how, values = name_test(stmt.test, base.at(stmt))
        rows.extend([f"skip_{how}", v, NONE, f"{file_name}:{line}"] for v, line in values)

    prepare = method(class_def(base.tree.body, "ModelBase", base.rel), "prepare_tensors", base.rel)
    skips = [n for n in ast.walk(prepare) if isinstance(n, ast.If) and len(n.body) == 1
             and isinstance(n.body[0], ast.Continue) and isinstance(n.test, ast.Call)
             and isinstance(n.test.func, ast.Attribute) and dotted(n.test.func.value) == "name"]
    if len(skips) != 1:
        fail(base.at(prepare), f"expected one tensor-name skip test in prepare_tensors, found {len(skips)}")
    how, values = name_test(skips[0].test, base.at(skips[0]))
    rows.extend([f"skip_{how}", v, NONE, f"{file_name}:{line}"] for v, line in values)

    forced = []
    for n in ast.walk(prepare):
        if isinstance(n, ast.If) and any(isinstance(s, ast.Assign) and dotted(s.targets[0]) == "data_qtype"
                                         and member(s.value, "gguf.GGMLQuantizationType") == "F32" for s in n.body):
            tuples = [t for t in ast.walk(n.test) if isinstance(t, ast.Tuple)
                      and t.elts and all(member(e, "gguf.MODEL_TENSOR") for e in t.elts)]
            forced.extend(e for t in tuples for e in t.elts)
    if not forced:
        fail(base.at(prepare), "no forced-F32 MODEL_TENSOR tuple found in prepare_tensors")
    rows.extend(["forced_f32", member(e, "gguf.MODEL_TENSOR"), NONE, f"{file_name}:{e.lineno}"] for e in forced)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Copied assets


def asset_sources(root: str, modules: List[Module]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    templates_dir = os.path.join(root, "models", "templates")
    templates: Dict[str, str] = {}
    vocabs: Dict[str, str] = {}
    for mod in modules:
        for n in ast.walk(mod.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".jinja") \
                    and os.path.isfile(os.path.join(templates_dir, n.value)):
                templates.setdefault(n.value, os.path.join(templates_dir, n.value))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_set_vocab_builtin":
                name = string(n.args[0], mod.at(n))
                path = os.path.join(root, "models", f"ggml-vocab-{name}.gguf")
                if not os.path.isfile(path):
                    fail(mod.at(n), f"{path} is missing")
                vocabs.setdefault(f"ggml-vocab-{name}.gguf", path)
    for file_name, path in vocabs.items():
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                fail(path, "not a GGUF file")
    return sorted(templates.items()), sorted(vocabs.items())


# ----------------------------------------------------------------------------------------------------------------
# Output


@dataclass(frozen=True)
class Table:
    file: str
    columns: Tuple[str, ...]
    rows: Sequence[Sequence[str]]
    lists: Tuple[str, ...] = ()

    def render(self, commit: str) -> str:
        lines = [f"# {self.file} - Generated from llama.cpp {commit} by the updateHfTables Gradle task; do not edit. "
                 f"Rows: {len(self.rows)}", "# " + "\t".join(self.columns)]
        for i, row in enumerate(self.rows):
            where = f"{self.file} row {i + 1}"
            if len(row) != len(self.columns):
                fail(where, f"{len(row)} fields for {len(self.columns)} columns")
            for col, value in zip(self.columns, row):
                if not isinstance(value, str) or value == "" or re.search(r"[\t\r\n]", value):
                    fail(where, f"column {col} holds an empty value or a TAB/CR/LF: {value!r}")
                if col in self.lists and value != NONE and any(item == "" or item != item.strip()
                                                                for item in value.split(",")):
                    fail(where, f"list column {col} has an empty or padded item: {value!r}")
            if row[0].startswith("#"):
                fail(where, "a row must not start with '#'")
            lines.append("\t".join(row))
        return "\n".join(lines) + "\n"


def write_file(out: str, rel: str, data: bytes) -> None:
    path = os.path.join(out, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def build(root: str, out: str, commit: str) -> Dict[str, str]:
    conv_dir = os.path.join(root, "conversion")
    if not os.path.isdir(conv_dir):
        raise GenError(f"{conv_dir}: missing; is --llama-cpp a llama.cpp checkout?")
    modules = [load_module(root, f"conversion/{f}") for f in sorted(os.listdir(conv_dir)) if f.endswith(".py")]
    by_rel = {m.rel: m for m in modules}
    base = by_rel.get("conversion/base.py") or fail(conv_dir, "base.py missing")
    consts = read_constants(load_module(root, "gguf-py/gguf/constants.py"))
    patterns, overrides = read_tensor_mapping(load_module(root, "gguf-py/gguf/tensor_mapping.py"))
    keys = writer_keys(load_module(root, "gguf-py/gguf/gguf_writer.py"), consts)
    classes = Classes([m for m in modules if not m.rel.endswith("__init__.py")])
    maps = read_model_maps(by_rel["conversion/__init__.py"])
    chk_rows, chktxt = chkhsh_table(base, load_module(root, "convert_hf_to_gguf_update.py"))
    templates, vocabs = asset_sources(root, modules)

    known_tensors = {t for t, _ in consts.tensor_names}
    unnamed = sorted({r[2] for r in patterns} - known_tensors) + sorted({r[1] for r in overrides} - known_tensors)
    if unnamed:
        fail("gguf-py/gguf/tensor_mapping.py", f"patterns for MODEL_TENSORs without a TENSOR_NAMES entry: {unnamed}")

    tables = [
        Table("archs.tsv", ("hf_name", "kind", "class", "source", "gguf_arch", "modify", "vocab_writers"),
              archs_table(maps, classes, consts), ("vocab_writers",)),
        Table("dynamic_archs.tsv", ("class", "source", "condition", "gguf_arch"),
              [r[2] for r in dynamic_archs(classes, consts)]),
        Table("tensor_patterns.tsv", ("seq", "scope", "tensor", "hf_template"), patterns),
        Table("tensor_names.tsv", ("tensor", "gguf_template"), [list(r) for r in consts.tensor_names]),
        Table("arch_tensors.tsv", ("gguf_arch", "tensor"),
              [[arch_name(consts, a, "constants.py MODEL_TENSORS"), t] for a, ts in consts.model_tensors for t in ts]),
        Table("arch_block_overrides.tsv", ("gguf_arch", "tensor", "hf_template", "seq"),
              [[arch_name(consts, r[0], "tensor_mapping.py arch_block_mappings_cfg")] + r[1:] for r in overrides]),
        Table("gguf_keys.tsv", ("writer_method", "key_template", "value_type"), keys),
        Table("hparam_aliases.tsv", ("gguf_key_template", "hf_aliases", "value_type", "rule"),
              hparam_aliases(base, writer_method_keys(keys)), ("hf_aliases",)),
        Table("chkhsh.tsv", ("sha256", "pre_name", "repo"), chk_rows),
        Table("filters.tsv", ("kind", "value", "replacement", "source"), filters_table(base)),
    ]
    files: Dict[str, bytes] = {t.file: t.render(commit).encode("utf-8") for t in tables}
    files["chktxt.txt"] = chktxt.encode("utf-8")
    for file_name, path in templates:
        with open(path, "rb") as f:
            files[f"templates/{file_name}"] = f.read()
    for file_name, path in vocabs:
        with open(path, "rb") as f:
            files[f"vocab/{file_name}"] = f.read()

    rows_of = {t.file: len(t.rows) for t in tables}
    manifest = [[rel, f"{rows_of[rel]} rows" if rel in rows_of else f"{len(data)} bytes",
                 hashlib.sha256(data).hexdigest()] for rel, data in sorted(files.items())]
    files["manifest.tsv"] = Table("manifest.tsv", ("file", "rows_or_bytes", "sha256"), manifest) \
        .render(commit).encode("utf-8")
    files[".gitignore"] = GITIGNORE.encode("utf-8")
    files[".gitattributes"] = GITATTRIBUTES.encode("utf-8")
    for rel, data in sorted(files.items()):
        write_file(out, rel, data)
    summary = {t.file: f"{len(t.rows)} rows" for t in tables}
    summary["templates"] = f"{len(templates)} files"
    summary["vocab"] = f"{len(vocabs)} files"
    summary["chkhsh pre names"] = f"{len({r[1] for r in chk_rows})} distinct"
    return summary


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generates NeuralFree's llama.cpp converter tables.")
    parser.add_argument("--llama-cpp", required=True, help="llama.cpp checkout to read")
    parser.add_argument("--out", required=True, help="empty or absent output directory")
    parser.add_argument("--commit", required=True, help="short commit id of the checkout, recorded in every header")
    args = parser.parse_args(argv)
    if sys.version_info < MIN_PYTHON:
        print(f"hfgen: needs Python {'.'.join(map(str, MIN_PYTHON))} or newer, found {sys.version.split()[0]}",
              file=sys.stderr)
        return 2
    if not re.fullmatch(r"[0-9a-f]{7,40}", args.commit):
        print(f"hfgen: --commit '{args.commit}' is not a git commit id", file=sys.stderr)
        return 2
    if os.path.exists(args.out) and os.listdir(args.out):
        print(f"hfgen: --out '{args.out}' is not empty", file=sys.stderr)
        return 2
    try:
        summary = build(os.path.abspath(args.llama_cpp), args.out, args.commit)
    except GenError as e:
        shutil.rmtree(args.out, ignore_errors=True)
        print(f"hfgen: {e}", file=sys.stderr)
        return 1
    for name, count in summary.items():
        print(f"{name}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
