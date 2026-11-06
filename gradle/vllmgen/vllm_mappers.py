"""Generates src/main/resources/vllm/weights_mappers.tsv from a vLLM checkout (standard library only).

Every architecture of vLLM's model registry (vllm/model_executor/models/registry.py) maps to a model class; this
script resolves the class's effective ``hf_to_vllm_mapper`` (its own, else the first base class's, through the
module's imports; the NVIDIA implementation of a platform-split model) by evaluating the class attribute with
Python's ``ast``: ``WeightsMapper(...)`` literals, ``A.hf_to_vllm_mapper | WeightsMapper(...)`` merges,
``dataclasses.replace``, ``dict.fromkeys``, module constants and module functions of straight-line code. vLLM is
never imported.

Output: one row per mapper entry, in vLLM's application order (WeightsMapper._map_name_with_shard), columns
separated by tabs:

    architecture   the registry name (config.json ``architectures[0]``)
    kind           regex | substr | stacked | prefix | suffix, or unresolved when the mapper is not static data
    match          the Python regex, substring, prefix or suffix (for unresolved: the expression not evaluated)
    replacement    the new text (a Python re.sub template for regex); <drop> when vLLM drops the weight (None)
    shard          the shard id of a stacked entry; empty otherwise

Usage: python3 -I gradle/vllmgen/vllm_mappers.py --vllm <checkout> --commit <short commit> --out <tsv>
"""
import argparse
import ast
import os
import sys

FIELDS = (('orig_to_new_regex', 'regex'), ('orig_to_new_substr', 'substr'), ('orig_to_new_stacked', 'stacked'),
          ('orig_to_new_prefix', 'prefix'), ('orig_to_new_suffix', 'suffix'))
DROP = '<drop>'


class Unresolved(Exception):
    """An expression this evaluator does not reduce to static mapper data."""


class Regex:
    """A ``re.compile`` pattern key."""

    def __init__(self, pattern):
        self.pattern = pattern


def empty():
    return {kind: {} for _, kind in FIELDS}


def merge(a, b):
    out = empty()
    for _, kind in FIELDS:
        out[kind].update(a[kind])
        out[kind].update(b[kind])
    return out


class Module:
    """One parsed module: its classes, functions, constants and imported names (first platform branch wins,
    NVIDIA implementations preferred)."""

    def __init__(self, name, path, repo):
        self.name = name
        self.classes, self.functions, self.constants, self.imports = {}, {}, {}, {}
        tree = ast.parse(open(path, encoding='utf-8').read(), path)
        package = os.path.relpath(os.path.dirname(path), repo).replace(os.sep, '.')
        for node in self._top(tree.body):
            if isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.constants.setdefault(node.targets[0].id, node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self.constants.setdefault(node.target.id, node.value)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    parts = package.split('.')
                    parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                    base = '.'.join(parts) + ('.' + node.module if node.module else '')
                for alias in node.names:
                    key = alias.asname or alias.name
                    if key not in self.imports or ('nvidia' in base and 'nvidia' not in self.imports[key][0]):
                        self.imports[key] = (base, alias.name)

    @staticmethod
    def _top(body):
        for node in body:
            if isinstance(node, ast.If):
                yield from Module._top(node.body)
                yield from Module._top(node.orelse)
            elif isinstance(node, ast.Try):
                yield from Module._top(node.body)
            else:
                yield node


class Resolver:
    def __init__(self, repo):
        self.repo = repo
        self.models = os.path.join(repo, 'vllm', 'model_executor', 'models')
        self.cache = {}

    def module(self, name):
        if name not in self.cache:
            rel = name.replace('.', os.sep) if name.startswith('vllm.') else os.path.join(
                'vllm', 'model_executor', 'models', name)
            path = None
            for candidate in (os.path.join(self.repo, rel + '.py'), os.path.join(self.repo, rel, '__init__.py')):
                if os.path.exists(candidate):
                    path = candidate
                    break
            self.cache[name] = Module(name, path, self.repo) if path else None
        return self.cache[name]

    def lookup(self, mod, name, kind, depth=0):
        m = self.module(mod)
        if m is None or depth > 10:
            return None
        table = {'class': m.classes, 'function': m.functions, 'constant': m.constants}[kind]
        if name in table:
            return m, table[name]
        if name in m.imports:
            base, real = m.imports[name]
            return self.lookup(base, real, kind, depth + 1)
        return None

    def mapper_of(self, m, cls):
        for stmt in cls.body:
            target = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = ast.unparse(stmt.targets[0])
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                target = ast.unparse(stmt.target)
            if target == 'hf_to_vllm_mapper':
                return self.evaluate(stmt.value, m, {})
        for base in cls.bases:
            name = ast.unparse(base).split('[')[0]
            found = self.lookup(m.name, name, 'class')
            if found is not None:
                result = self.mapper_of(*found)
                if result is not None:
                    return result
        return None

    def evaluate(self, node, m, scope):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.evaluate(e, m, scope) for e in node.elts)
        if isinstance(node, ast.Dict):
            out = {}
            for k, v in zip(node.keys, node.values):
                if k is None:
                    out.update(self.evaluate(v, m, scope))
                else:
                    out[self.evaluate(k, m, scope)] = self.evaluate(v, m, scope)
            return out
        if isinstance(node, ast.Name):
            if node.id in scope:
                return scope[node.id]
            found = self.lookup(m.name, node.id, 'constant')
            if found is not None:
                return self.evaluate(found[1], found[0], {})
            raise Unresolved(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left, right = self.evaluate(node.left, m, scope), self.evaluate(node.right, m, scope)
            if isinstance(left, dict) and 'regex' in left:
                return merge(left, right)
            out = dict(left)
            out.update(right)
            return out
        if isinstance(node, ast.Attribute) and node.attr == 'hf_to_vllm_mapper':
            found = self.lookup(m.name, ast.unparse(node.value), 'class')
            if found is None:
                raise Unresolved(ast.unparse(node))
            return self.mapper_of(*found)
        if isinstance(node, ast.Call):
            return self.call(node, m, scope)
        if isinstance(node, ast.DictComp) and len(node.generators) == 1 and not node.generators[0].ifs:
            gen = node.generators[0]
            out = {}
            for item in self.evaluate(gen.iter, m, scope):
                inner = dict(scope)
                self.bind(gen.target, item, inner)
                out[self.evaluate(node.key, m, inner)] = self.evaluate(node.value, m, inner)
            return out
        if isinstance(node, ast.JoinedStr):
            return ''.join(str(self.evaluate(v, m, scope)) if isinstance(v, ast.Constant)
                           else self.formatted(v, m, scope) for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.evaluate(node.left, m, scope) + self.evaluate(node.right, m, scope)
        raise Unresolved(ast.unparse(node))

    def formatted(self, node, m, scope):
        if not isinstance(node, ast.FormattedValue) or node.conversion != -1 or node.format_spec is not None:
            raise Unresolved(ast.unparse(node))
        return str(self.evaluate(node.value, m, scope))

    def bind(self, target, value, scope):
        if isinstance(target, ast.Name):
            scope[target.id] = value
        elif isinstance(target, ast.Tuple) and len(target.elts) == len(value):
            for t, v in zip(target.elts, value):
                self.bind(t, v, scope)
        else:
            raise Unresolved(ast.unparse(target))

    def call(self, node, m, scope):
        fn = ast.unparse(node.func)
        if fn.endswith('WeightsMapper'):
            out = empty()
            for kw in node.keywords:
                if kw.arg == 'weights_transforms':
                    continue
                if kw.arg == 'orig_to_new_renaming':
                    value = self.evaluate(kw.value, m, scope)
                    if value:
                        raise Unresolved('orig_to_new_renaming')
                    continue
                kind = dict(FIELDS).get(kw.arg)
                if kind is None:
                    raise Unresolved('WeightsMapper(' + str(kw.arg) + '=...)')
                out[kind] = dict(self.evaluate(kw.value, m, scope))
            return out
        if fn in ('replace', 'dataclasses.replace'):
            out = dict(self.evaluate(node.args[0], m, scope))
            for kw in node.keywords:
                out[dict(FIELDS)[kw.arg]] = dict(self.evaluate(kw.value, m, scope))
            return out
        if fn in ('re.compile', 'regex.compile'):
            return Regex(self.evaluate(node.args[0], m, scope))
        if fn == 'dict.fromkeys':
            return dict.fromkeys(self.evaluate(node.args[0], m, scope),
                                 self.evaluate(node.args[1], m, scope) if len(node.args) > 1 else None)
        if fn == 'range' and len(node.args) == 1:
            return tuple(range(self.evaluate(node.args[0], m, scope)))
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'items' and not node.args:
            return tuple(self.evaluate(node.func.value, m, scope).items())
        if fn == 'dict' and len(node.args) == 1 and not node.keywords:
            return dict(self.evaluate(node.args[0], m, scope))
        found = self.lookup(m.name, fn, 'function')
        if found is not None:
            return self.run(found[1], found[0], [self.evaluate(a, m, scope) for a in node.args])
        raise Unresolved(ast.unparse(node))

    def run(self, fn, m, args):
        """Runs a module function of assignments, if/else on its arguments and a return."""
        scope = {}
        params = fn.args.args
        defaults = fn.args.defaults
        for i, p in enumerate(params):
            if i < len(args):
                scope[p.arg] = args[i]
            elif i >= len(params) - len(defaults):
                scope[p.arg] = self.evaluate(defaults[i - (len(params) - len(defaults))], m, {})
        return self.block(fn.body, m, scope)

    def block(self, body, m, scope):
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                scope[stmt.targets[0].id] = self.evaluate(stmt.value, m, scope)
            elif isinstance(stmt, ast.If):
                result = self.block(stmt.body if self.test(stmt.test, m, scope) else stmt.orelse, m, scope)
                if result is not None:
                    return result
            elif isinstance(stmt, ast.Return):
                return self.evaluate(stmt.value, m, scope)
            else:
                raise Unresolved(ast.unparse(stmt).splitlines()[0])
        return None

    def test(self, node, m, scope):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = self.evaluate(node.left, m, scope)
            right = self.evaluate(node.comparators[0], m, scope)
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            if isinstance(op, ast.In):
                return left in right
            if isinstance(op, ast.NotIn):
                return left not in right
        raise Unresolved(ast.unparse(node))


def registry(resolver):
    tree = ast.parse(open(os.path.join(resolver.models, 'registry.py'), encoding='utf-8').read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if (isinstance(k, ast.Constant) and isinstance(v, ast.Tuple) and len(v.elts) == 2
                        and all(isinstance(e, ast.Constant) for e in v.elts)):
                    out.setdefault(k.value, (v.elts[0].value, v.elts[1].value))
    return out


def text(value):
    if value is None:
        return DROP
    if not isinstance(value, str) or any(c in value for c in '\t\r\n') or value == DROP:
        raise Unresolved('a mapper value that is not a plain string: ' + repr(value))
    return value


def rows(arch, mapper):
    out = []
    for _, kind in FIELDS:
        for key, value in mapper[kind].items():
            match = key.pattern if isinstance(key, Regex) else key
            if kind == 'stacked':
                new, shard = value
                out.append((arch, kind, text(match), text(new), str(shard)))
            else:
                out.append((arch, kind, text(match), text(value), ''))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    resolver = Resolver(os.path.abspath(args.vllm))
    table, unresolved = [], []
    for arch, (mod, cls) in sorted(registry(resolver).items()):
        found = resolver.lookup(mod, cls, 'class')
        if found is None:
            continue
        try:
            mapper = resolver.mapper_of(*found)
            if mapper is not None:
                table.extend(rows(arch, mapper))
        except Unresolved as e:
            table.append((arch, 'unresolved', str(e).replace('\t', ' ').replace('\n', ' '), '', ''))
            unresolved.append(arch)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# weights_mappers.tsv - Generated from vLLM %s by gradle/vllmgen/vllm_mappers.py; do not edit.'
                ' Rows: %d\n' % (args.commit, len(table)))
        f.write('# architecture\tkind\tmatch\treplacement\tshard\n')
        for row in table:
            f.write('\t'.join(row) + '\n')
    archs = len({r[0] for r in table})
    print('%d rows for %d architectures; unresolved: %s' % (len(table), archs, ', '.join(unresolved) or 'none'))


if __name__ == '__main__':
    sys.exit(main())
