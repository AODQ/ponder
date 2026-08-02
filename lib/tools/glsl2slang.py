#!/usr/bin/env python3
"""GLSL -> Slang converter for the ponder shader port.

Handles the mechanical 90%: signatures, builtin renames, includes, struct
visibility. Anything it cannot do confidently it leaves alone and reports,
so the remainder is hand-fixed rather than silently wrong.
"""
import json, re, sys, pathlib

TYPES = r'(?:void|bool|int|uint|float|f32|f64|i32|u32|u64|vec2|vec3|vec4|ivec2|ivec3|ivec4|uvec2|uvec3|uvec4|mat2|mat3|mat4|f32v2|f32v3|f32v4|i32v2|i32v3|i32v4|u32v2|u32v3|u32v4|f32m44|f32m3x3|f32m4x4|[A-Z][A-Za-z0-9_]*)'

BUILTIN = [
    (r'\bmix\(', 'lerp('),
    (r'\bfract\(', 'frac('),
    (r'\binversesqrt\(', 'rsqrt('),
    (r'\buintBitsToFloat\(', 'asfloat('),
    (r'\bfloatBitsToUint\(', 'asuint('),
    (r'\bintBitsToFloat\(', 'asfloat('),
    (r'\bfloatBitsToInt\(', 'asint('),
    (r'\bdFdx\(', 'ddx('),
    (r'\bdFdy\(', 'ddy('),
    (r'\bnonuniformEXT\(', 'NonUniformResourceIndex('),
    (r'\bbvec2\b', 'bool2'),
    (r'\bbvec3\b', 'bool3'),
    (r'\bbvec4\b', 'bool4'),
]

# glsl vector/matrix spellings -> the f32v* aliases used everywhere here
TYPEMAP = [
    (r'\bvec2\b', 'f32v2'), (r'\bvec3\b', 'f32v3'), (r'\bvec4\b', 'f32v4'),
    (r'\bivec2\b', 'i32v2'), (r'\bivec3\b', 'i32v3'), (r'\bivec4\b', 'i32v4'),
    (r'\buvec2\b', 'u32v2'), (r'\buvec3\b', 'u32v3'), (r'\buvec4\b', 'u32v4'),
    (r'\bmat3\b', 'f32m3x3'), (r'\bmat4\b', 'f32m4x4'),
    (r'\bmat4x3\b', 'f32m4x3'), (r'\bmat3x4\b', 'f32m3x4'),
    (r'\bfloat\b', 'f32'), (r'\buint\b', 'u32'), (r'\bint\b', 'i32'),
]

def map_outside_comments(text, apply):
    """runs a textual rewrite on code only.

    the doc comments carry latex (`\\int`) and glsl names that a blind
    substitution would rewrite into nonsense
    """
    out = []
    i = 0
    n = len(text)
    start = 0
    while i < n:
        if text[i] == '/' and i + 1 < n and text[i + 1] in '/*':
            out.append(apply(text[start:i]))
            if text[i + 1] == '/':
                j = text.find('\n', i)
                j = n if j < 0 else j
            else:
                j = text.find('*/', i + 2)
                j = n if j < 0 else j + 2
            out.append(text[i:j])
            i = start = j
            continue
        i += 1
    out.append(apply(text[start:]))
    return ''.join(out)


def convert_signatures(text):
    """`Ret name(const T a, out T b) {` -> `public func name(a: T, out b: T) -> Ret {`"""
    out = []
    i = 0
    pat = re.compile(r'^(' + TYPES + r')\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.M)
    pos = 0
    for m in pat.finditer(text):
        # find the matching close paren
        depth = 1
        j = m.end()
        while j < len(text) and depth:
            if text[j] == '(': depth += 1
            elif text[j] == ')': depth -= 1
            j += 1
        after = text[j:j+40]
        if not re.match(r'\s*\{', after):
            continue  # a call or prototype, not a definition
        ret, name = m.group(1), m.group(2)
        params = text[m.end():j-1]
        # strip comments per line first: a comment may itself contain a comma
        # and would otherwise split the parameter list wrongly
        notes = {}
        stripped = []
        for ln in params.split('\n'):
            cm = re.search(r'//(.*)$', ln)
            if cm:
                notes[len(stripped)] = cm.group(1).strip()
                ln = ln[:cm.start()]
            stripped.append(ln)
        params = '\n'.join(stripped)
        newparams = []
        ok = True
        idx = -1
        for p in [x.strip() for x in params.split(',')]:
            idx += 1
            if not p:
                continue
            note = notes.get(idx + (1 if idx else 0))
            note = None
            # a buffer reference has already become `T *`; slang rejects
            # `const` on a pointer parameter, so the qualifier is dropped
            pm = re.match(r'^(const\s+|out\s+|inout\s+)?(' + TYPES + r'\s*\*?)\s+([a-zA-Z_][a-zA-Z0-9_]*)$', p.replace('\n',' ').strip())
            if not pm:
                ok = False
                break
            qual = (pm.group(1) or '').strip()
            ty, nm = pm.group(2), pm.group(3)
            ty = ty.strip()
            prefix = '' if qual in ('', 'const') else qual + ' '
            newparams.append(f'\t{prefix}{nm}: {ty},')
        if not ok:
            continue
        body = '\n'.join(newparams)
        rep = f'public func {name}(\n{body}\n)'
        if ret != 'void':
            rep += f' -> {ret}'
        out.append((m.start(), j, rep))
    for start, end, rep in reversed(out):
        text = text[:start] + rep + text[end:]
    return text

def fix_bool_mix(text):
    """rewrite `lerp(a, b, cond)` to `select(cond, b, a)` where cond is a
    bool declared in the same file. runs after the mix->lerp rename, so it
    matches lerp"""
    bools = set(re.findall(r'\bbool[234]?\s+(\w+)', text))
    if not bools:
        return text
    names = '|'.join(sorted(bools))
    # the selector may be negated and parenthesised, e.g. `(!(raise))`
    sel = r'((?:[!(\s]*(?:' + names + r')[)\s]*))'
    return re.sub(
        r'\blerp\(\s*([^,()]+(?:\([^()]*\))?[^,()]*),\s*'
        r'([^,()]+(?:\([^()]*\))?[^,()]*),\s*' + sel + r'\)',
        r'select(\3, \2, \1)', text)


_MATRIX_NAMES = None
MATRIX_TYPE = r'(?:mat[234](?:x[234])?|f32m44|f32m3x3|f32m4x4)'


def matrix_names():
    """every identifier declared with a matrix type anywhere in the shader
    tree, locals and struct fields alike.

    the operand order of a multiply has to flip for matrices and must not
    flip for anything else, and getting it wrong is silently wrong rather
    than a compile error -- so the set is collected rather than guessed at
    from naming conventions.
    """
    global _MATRIX_NAMES
    if _MATRIX_NAMES is not None:
        return _MATRIX_NAMES
    names = set()
    roots = [pathlib.Path('lib/ponder/shaders'), pathlib.Path('lib/mor/include/mor'),
             pathlib.Path('app/viewer/shaders'), pathlib.Path('app/observer/shaders'),
             pathlib.Path('unit-test/shaders')]
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob('*.glsl')) + list(root.rglob('*.h')) + \
                 list(root.rglob('*.comp')):
            try:
                text = f.read_text()
            except Exception:
                continue
            names |= set(re.findall(
                r'\b' + MATRIX_TYPE + r'\s+(\w+)', text))
    _MATRIX_NAMES = names
    return names


def is_matrix_operand(name):
    if name is None:
        return False
    # a constructor call is as much a matrix operand as a declared name
    if re.match(r'^(?:' + MATRIX_TYPE + r'|f32m[234]x[234])\s*\(', name):
        return True
    return name.split('.')[-1] in matrix_names()


def _operand_before(text, pos):
    """the dotted name or call ending at pos-1, or None.

    a matrix operand is often a constructor -- `f32m3x3(objectToWorld)` --
    so the scan has to step back over a balanced call, not just a name
    """
    j = pos
    while j > 0 and text[j - 1] in ' \t\n':
        j -= 1
    k = j
    if k > 0 and text[k - 1] == ')':
        depth = 0
        i = k - 1
        while i >= 0:
            if text[i] == ')':
                depth += 1
            elif text[i] == '(':
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        if i < 0:
            return (None, pos)
        k = i
        while k > 0 and (text[k - 1].isalnum() or text[k - 1] in '_.'):
            k -= 1
        return (text[k:j], k) if k < j else (None, pos)
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] in '_.'):
        k -= 1
    return (text[k:j], k) if k < j else (None, pos)


def _operand_after(text, pos):
    """a dotted name, or a call/parenthesised expression, starting at pos"""
    m = pos
    while m < len(text) and text[m] in ' \t\n':
        m += 1
    n = m
    while n < len(text) and (text[n].isalnum() or text[n] in '_.'):
        n += 1
    if n < len(text) and text[n] == '(':
        depth = 1
        n += 1
        while n < len(text) and depth:
            if text[n] == '(':
                depth += 1
            elif text[n] == ')':
                depth -= 1
            n += 1
    return (text[m:n].strip(), n) if n > m else (None, pos)


# function name -> parameters the glsl declared without `const`, collected
# before convert_signatures discards the qualifier
MUTABLE_PARAMS = {}


def collect_mutable_params(text):
    MUTABLE_PARAMS.clear()
    # comments in a parameter list carry parens of their own, which a
    # non-greedy paren match would stop at
    text = blank_comments(text)
    for m in re.finditer(r'^(?:' + TYPES + r')\s+(\w+)\s*\(([^)]*)\)\s*\{',
                         text, re.M | re.S):
        names = set()
        for line in m.group(2).split('\n'):
            pm = re.match(r'\s*(' + TYPES + r')\s+(\w+)\s*,?\s*$', line)
            if pm:
                names.add(pm.group(2))
        if names:
            MUTABLE_PARAMS[m.group(1)] = names


def shadow_mutated_params(text):
    """a glsl parameter is a mutable copy; a slang one is immutable.

    a function that assigns to its own parameter gets a local shadow so the
    body is otherwise unchanged.
    """
    out = text
    # reversed so an insertion never shifts a later match's offsets
    matches = list(re.finditer(r'^public func (\w+)\(([^)]*)\)[^{]*\{', out, re.M | re.S))
    for m in reversed(matches):
        # out/inout parameters are already writable in slang; shadowing one
        # would send every write to the local and never to the caller
        params = re.findall(r'^\t(?!out |inout )(\w+)\s*:', m.group(2), re.M)
        if not params:
            continue
        bodyStart = m.end()
        bodyEnd = end_of_body(out, m.end() - 1)
        body = out[bodyStart:bodyEnd]
        shadows = []
        mutable = MUTABLE_PARAMS.get(m.group(1), set())
        for name in params:
            # this codebase marks every read-only parameter `const`, so one
            # without it is a deliberately mutable copy -- and slang needs an
            # l-value for that even when the body only forwards it to an
            # `inout` argument rather than assigning it
            if name in mutable:
                shadows.append(name)
                continue
            if re.search(r'^\s*' + name + r'\s*(?:=[^=]|[-+*/|&^]=|\+\+|--)', body, re.M):
                shadows.append(name)
        if not shadows:
            continue
        prologue = ''.join(
            f'\n\t// glsl parameters are mutable copies; slang\'s are not\n'
            f'\tvar {n} = {n};' for n in shadows)
        out = out[:bodyStart] + prologue + out[bodyStart:]
    return out


_TRANSPOSED_NAMES = None


def transposed_matrix_names():
    """matrix identifiers whose slang value is the transpose of the glsl one.

    a `mat3(a, b, c)` fills columns in glsl and rows in slang, so a matrix
    built that way lands transposed and `M * v` becomes `mul(v, M)`. one
    loaded from memory does not: the layout flags make both languages agree,
    and the multiply keeps its order. defaulting to "not transposed" matches
    the hand-written util-manifold-mesh.slang
    """
    global _TRANSPOSED_NAMES
    if _TRANSPOSED_NAMES is not None:
        return _TRANSPOSED_NAMES
    names = set()
    roots = [pathlib.Path('lib/ponder/shaders'), pathlib.Path('app/viewer/shaders'),
             pathlib.Path('app/observer/shaders'), pathlib.Path('unit-test/shaders')]
    # tolerates the house wrapping `mat3 m = (\n  mat3(...)\n);` and a
    # transpose() around the constructor
    ctor = re.compile(
        r'\b' + MATRIX_TYPE + r'\s+(\w+)\s*=\s*\(?\s*(?:transpose\s*\(\s*)?'
        r'(' + MATRIX_TYPE + r')\s*\(([^;]*?)\)', re.S)
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob('*.glsl')) + list(root.rglob('*.comp')):
            try:
                text = f.read_text()
            except Exception:
                continue
            # a basis built by a helper is a constructor once removed
            names |= set(re.findall(
                r'\b' + MATRIX_TYPE + r'\s+(\w+)\s*=\s*\(?\s*'
                r'(?:transpose\s*\(\s*)?mx_orthonormal_basis\w*\s*\(', text))
            for m in ctor.finditer(text):
                args = m.group(3)
                # matN(matM) is a layout-preserving conversion, not a build
                if re.match(r'^\s*[\w.]+\s*$', args) and not re.search(
                        r'\b' + MATRIX_TYPE + r'\b', m.group(2) + args):
                    continue
                if re.match(r'^\s*[\w.]+\s*$', args):
                    continue
                names.add(m.group(1))
    _TRANSPOSED_NAMES = names
    return names


def fix_matrix_multiply(text):
    """glsl `M * v` is slang `mul(v, M)`, and a chain flips end to end.

    a glsl mat3(a..i) fills columns where slang's fills rows, so the same
    nine scalars land transposed and the operand order reverses:
    `A * B * v` becomes `mul(v, mul(B, A))`. matched by scanning rather than
    by regex, because the operands are often calls or dotted fields and a
    missed site is silently wrong rather than a compile error.
    """
    out = []
    i = 0
    while True:
        star = text.find('*', i)
        if star < 0:
            out.append(text[i:])
            break
        lhs, lhsStart = _operand_before(text, star)
        if not is_matrix_operand(lhs):
            out.append(text[i:star + 1])
            i = star + 1
            continue
        operands = [lhs]
        cursor = star
        while cursor < len(text) and text[cursor] == '*':
            rhs, after = _operand_after(text, cursor + 1)
            if rhs is None:
                break
            operands.append(rhs)
            cursor = after
            while cursor < len(text) and text[cursor] in ' \t\n':
                cursor += 1
        if len(operands) < 2:
            out.append(text[i:star + 1])
            i = star + 1
            continue
        # only a transposed (constructor-built) matrix reverses the order
        def transposed(op):
            if re.match(r'^(?:' + MATRIX_TYPE + r')\s*\(', op):
                return not re.match(
                    r'^(?:' + MATRIX_TYPE + r')\s*\(\s*[\w.]+\s*\)$', op)
            return op.split('.')[-1] in transposed_matrix_names()
        if all(transposed(op) for op in operands[:-1]):
            expr = operands[-1]
            for op in reversed(operands[:-1]):
                expr = f'mul({expr}, {op})'
        else:
            expr = operands[-1]
            for op in reversed(operands[:-1]):
                expr = f'mul({op}, {expr})'
        out.append(text[i:lhsStart])
        out.append(expr)
        i = cursor
    return ''.join(out)


_VECTOR_NAMES = None
VECTOR_TYPE = r'(?:[biu]?vec[234]|[fiu]32v[234])'
SCALAR_TYPE = r'(?:float|u?int|bool|f32|i32|u32|u64)'


def vector_names():
    """every identifier declared with a vector type anywhere in the tree.

    collected the same way as matrix_names, and for the same reason: glsl `==`
    on vectors is a scalar bool where slang's is componentwise.
    """
    global _VECTOR_NAMES
    if _VECTOR_NAMES is not None:
        return _VECTOR_NAMES
    names = set()
    scalars = set()
    roots = [pathlib.Path('lib/ponder/shaders'), pathlib.Path('lib/mor/include/mor'),
             pathlib.Path('app/viewer/shaders'), pathlib.Path('app/observer/shaders'),
             pathlib.Path('unit-test/shaders')]
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob('*.glsl')) + list(root.rglob('*.h')) + \
                 list(root.rglob('*.comp')):
            try:
                text = f.read_text()
            except Exception:
                continue
            names |= set(re.findall(r'\b' + VECTOR_TYPE + r'\s+(\w+)', text))
            scalars |= set(re.findall(r'\b' + SCALAR_TYPE + r'\s+(\w+)', text))
    # a name used for a scalar somewhere is ambiguous; dropping it costs at
    # worst a type error, where keeping it would fold an unrelated comparison
    _VECTOR_NAMES = names - scalars
    return _VECTOR_NAMES


def is_vector_operand(name):
    if name is None:
        return False
    # a vector constructor is unambiguous, where a bare name may be one the
    # scalar filter had to drop
    if re.match(r'^(?:' + VECTOR_TYPE + r')\s*\(', name):
        return True
    return name.split('.')[-1] in vector_names()


_WIDE_MATRIX_NAMES = None
WIDE_MATRIX_TYPE = r'(?:mat4x3|f32m4x3)'


def wide_matrix_names():
    """identifiers declared mat4x3 anywhere in the tree"""
    global _WIDE_MATRIX_NAMES
    if _WIDE_MATRIX_NAMES is not None:
        return _WIDE_MATRIX_NAMES
    names = set()
    roots = [pathlib.Path('lib/ponder/shaders'), pathlib.Path('app/viewer/shaders'),
             pathlib.Path('app/observer/shaders'), pathlib.Path('unit-test/shaders')]
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob('*.glsl')) + list(root.rglob('*.comp')):
            try:
                names |= set(re.findall(
                    r'\b' + WIDE_MATRIX_TYPE + r'\s+(\w+)', f.read_text()))
            except Exception:
                continue
    _WIDE_MATRIX_NAMES = names
    return names


def fix_wide_matrix_index(text):
    """a glsl mat4x3 indexes by column; the slang 3x4 holding it indexes by
    row, so `m[i]` has to gather the column by hand"""
    for name in wide_matrix_names():
        text = re.sub(
            r'\b' + name + r'\[([^\[\]]+)\](?!\[)',
            lambda m, n=name: (f'f32v3({n}[0][{m.group(1)}], '
                               f'{n}[1][{m.group(1)}], {n}[2][{m.group(1)}])'),
            text)
    return text


def fix_vector_equality(text):
    """glsl `u == v` on vectors is one bool; slang's is a bool per component.

    wrapping a scalar comparison in all()/any() is a no-op, so a false
    positive from the name scan costs nothing while a miss is a type error.
    """
    out = []
    i = 0
    while True:
        k = min([x for x in (text.find('==', i), text.find('!=', i)) if x >= 0],
                default=-1)
        if k < 0:
            out.append(text[i:])
            break
        op = text[k:k + 2]
        lhs, lhsStart = _operand_before(text, k)
        rhs, after = _operand_after(text, k + 2)
        if rhs is None or not (is_vector_operand(lhs) or is_vector_operand(rhs)):
            out.append(text[i:k + 2])
            i = k + 2
            continue
        fold = 'all' if op == '==' else 'any'
        out.append(text[i:lhsStart])
        out.append(f'{fold}({lhs} {op} {rhs})')
        i = after
    return ''.join(out)


def fix_atomics(text):
    """glsl passes the target by name; the util-types shims take its address"""
    out = []
    i = 0
    while True:
        m = re.search(r'\b(atomicAdd|atomicCompSwap)\(', text[i:])
        if not m:
            out.append(text[i:])
            break
        start = i + m.end()
        out.append(text[i:start])
        out.append('&')
        i = start
    return ''.join(out)


def fix_atan2(text):
    """glsl atan(y, x) is slang atan2; single-arg atan is unchanged"""
    out = []
    i = 0
    while True:
        k = text.find('atan(', i)
        if k < 0:
            out.append(text[i:])
            break
        if k > 0 and (text[k-1].isalnum() or text[k-1] == '_'):
            out.append(text[i:k+5]); i = k+5; continue
        depth = 1
        j = k + 5
        comma = False
        while j < len(text) and depth:
            c = text[j]
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            elif c == ',' and depth == 1: comma = True
            j += 1
        out.append(text[i:k])
        out.append('atan2(' if comma else 'atan(')
        out.append(text[k+5:j])
        i = j
    return ''.join(out)


def shared_buffer_elements():
    """buffer-reference element types, as shared-header-to-slang.py recorded
    them. reading the sidecar keeps the two tools agreeing by construction
    rather than by a hand-copied table."""
    elements = {}
    generated = pathlib.Path('lib/ponder/shaders/generated')
    for f in sorted(generated.glob('*.buffers.json')):
        elements.update(json.loads(f.read_text()))
    return elements


def take_push_constant(text, retype):
    """the push constant block becomes a struct plus a declared global"""
    m = re.search(
        r'layout\(push_constant[^)]*\)\s*uniform\s+(\w+)\s*\{(.*?)\}\s*(\w+)\s*;',
        text, re.S)
    if not m:
        return text, ''
    name, body, var = m.group(1), m.group(2), m.group(3)
    # `layout(offset = N)` reserves the root constants ahead of the block
    padBytes = 0
    offset = re.search(r'layout\(offset\s*=\s*(\d+)\)', body)
    if offset:
        padBytes = int(offset.group(1))
    body = re.sub(r'layout\(offset\s*=\s*\d+\)\s*', '', body)
    fields = body.strip('\n')
    pad = ''
    if padBytes:
        if padBytes % 16 != 0:
            raise SystemExit(f'push constant offset {padBytes} is not a multiple of 16')
        pad = f'\tu32v4 rootPad[{padBytes // 16}];\n'
    struct = (
        f'struct {name} {{\n{pad}{fields}\n}};\n\n'
        f'[[vk::push_constant]] {name} {var};\n'
    )
    return text[:m.start()] + text[m.end():], retype(struct)


_INCLUDED_ELEMENTS = None


def included_buffer_elements():
    """buffer-reference element types declared anywhere in the shader tree.

    a shader often uses a type one of its headers declared, and the
    declaration is gone by the time that header has become a module
    """
    global _INCLUDED_ELEMENTS
    if _INCLUDED_ELEMENTS is not None:
        return _INCLUDED_ELEMENTS
    found = {}
    roots = [pathlib.Path('lib/ponder/shaders'), pathlib.Path('unit-test/shaders'),
             pathlib.Path('app/viewer/shaders'), pathlib.Path('app/observer/shaders')]
    for root in roots:
        if not root.exists():
            continue
        for f in list(root.rglob('*.glsl')) + list(root.rglob('*.h')) + \
                 list(root.rglob('*.comp')):
            try:
                text = f.read_text()
            except Exception:
                continue
            for m in re.finditer(
                    r'layout\(buffer_reference[^)]*\)\s*buffer\s+(\w+)\s*\{\s*'
                    r'([\w]+)\s+(\w+)(\[\])?\s*;\s*\}\s*;', text):
                found.setdefault(m.group(1), (m.group(2).strip(), not m.group(4)))
    _INCLUDED_ELEMENTS = found
    return found


def _call_args(text, start):
    """the balanced argument list of a call whose `(` is at start"""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1
    inner = text[start + 1:i]
    args = []
    depth = 0
    cur = ''
    for c in inner:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if c == ',' and depth == 0:
            args.append(cur)
            cur = ''
            continue
        cur += c
    args.append(cur)
    clean = [re.sub(r'/\*.*?\*/', '', a, flags=re.S).strip() for a in args]
    return clean, i + 1


# the committed-hit accessors are a straight rename once the ray query is a
# slang object; each takes (rayQuery, committed) and drops both
RAY_QUERY_ACCESSORS = {
    'rayQueryGetIntersectionTypeEXT': 'CommittedStatus',
    'rayQueryGetIntersectionTEXT': 'CommittedRayT',
    'rayQueryGetIntersectionPrimitiveIndexEXT': 'CommittedPrimitiveIndex',
    'rayQueryGetIntersectionInstanceCustomIndexEXT': 'CommittedInstanceID',
    'rayQueryGetIntersectionInstanceIdEXT': 'CommittedInstanceIndex',
    'rayQueryGetIntersectionObjectToWorldEXT': 'CommittedObjectToWorld3x4',
    'rayQueryGetIntersectionBarycentricsEXT': 'CommittedTriangleBarycentrics',
    'rayQueryGetIntersectionFrontFaceEXT': 'CommittedTriangleFrontFace',
}

# the same glsl call with committed=false asks about the candidate hit
RAY_QUERY_CANDIDATE = {
    'rayQueryGetIntersectionTypeEXT': 'CandidateType',
    'rayQueryGetIntersectionTEXT': 'CandidateTriangleRayT',
    'rayQueryGetIntersectionPrimitiveIndexEXT': 'CandidatePrimitiveIndex',
    'rayQueryGetIntersectionInstanceCustomIndexEXT': 'CandidateInstanceID',
    'rayQueryGetIntersectionBarycentricsEXT': 'CandidateTriangleBarycentrics',
    'rayQueryGetIntersectionObjectToWorldEXT': 'CandidateObjectToWorld3x4',
}

RAY_QUERY_CONSTANTS = [
    ('gl_RayQueryCommittedIntersectionNoneEXT', 'COMMITTED_NOTHING'),
    ('gl_RayQueryCommittedIntersectionTriangleEXT', 'COMMITTED_TRIANGLE_HIT'),
    ('gl_RayFlagsOpaqueEXT', 'RAY_FLAG_FORCE_OPAQUE'),
    ('gl_RayFlagsTerminateOnFirstHitEXT',
     'RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH'),
    ('gl_RayFlagsNoneEXT', 'RAY_FLAG_NONE'),
    ('gl_RayQueryCandidateIntersectionTriangleEXT',
     'CANDIDATE_NON_OPAQUE_TRIANGLE'),
    ('gl_RayQueryCandidateIntersectionAABBEXT',
     'CANDIDATE_PROCEDURAL_PRIMITIVE'),
    ('gl_RayFlagsCullBackFacingTrianglesEXT',
     'RAY_FLAG_CULL_BACK_FACING_TRIANGLES'),
]


def fix_ray_query(text):
    """the GL_EXT_ray_query calls become slang's RayQuery methods.

    the ray is a RayDesc in slang where glsl passes origin/tmin/dir/tmax as
    four arguments, so the initialize call expands to five statements
    """
    text = re.sub(r'\brayQueryEXT\s+(\w+)\s*;',
                  r'RayQuery<RAY_FLAG_NONE> \1;', text)
    while True:
        m = re.search(r'\brayQueryInitializeEXT\s*\(', text)
        if not m:
            break
        args, end = _call_args(text, m.end() - 1)
        if len(args) != 8:
            raise SystemExit(
                f'rayQueryInitializeEXT with {len(args)} arguments')
        rq, tlas, flags, mask, origin, tMin, direction, tMax = args
        # the statement's own indentation, so the expansion lines up
        lineStart = text.rfind('\n', 0, m.start()) + 1
        indent = text[lineStart:m.start()]
        ray = f'{rq}Ray'
        expansion = (
            f'RayDesc {ray};\n'
            f'{indent}{ray}.Origin = {origin};\n'
            f'{indent}{ray}.TMin = {tMin};\n'
            f'{indent}{ray}.Direction = {direction};\n'
            f'{indent}{ray}.TMax = {tMax};\n'
            f'{indent}{rq}.TraceRayInline({tlas}, {flags}, {mask}, {ray})'
        )
        text = text[:m.start()] + expansion + text[end:]
    for glsl, method in RAY_QUERY_ACCESSORS.items():
        while True:
            m = re.search(r'\b' + glsl + r'\s*\(', text)
            if not m:
                break
            args, end = _call_args(text, m.end() - 1)
            pick = method
            if len(args) > 1 and args[1] == 'false':
                pick = RAY_QUERY_CANDIDATE.get(glsl)
                if pick is None:
                    raise SystemExit(f'no candidate-side form of {glsl}')
            text = text[:m.start()] + f'{args[0]}.{pick}()' + text[end:]
    while True:
        m = re.search(r'\brayQueryConfirmIntersectionEXT\s*\(', text)
        if not m:
            break
        args, end = _call_args(text, m.end() - 1)
        text = (text[:m.start()]
                + f'{args[0]}.CommitNonOpaqueTriangleHit()' + text[end:])
    while True:
        m = re.search(r'\brayQueryTerminateEXT\s*\(', text)
        if not m:
            break
        args, end = _call_args(text, m.end() - 1)
        text = text[:m.start()] + f'{args[0]}.Abort()' + text[end:]
    while True:
        m = re.search(r'\brayQueryProceedEXT\s*\(', text)
        if not m:
            break
        args, end = _call_args(text, m.end() - 1)
        text = text[:m.start()] + f'{args[0]}.Proceed()' + text[end:]
    for glsl, slang in RAY_QUERY_CONSTANTS:
        text = re.sub(r'\b' + glsl + r'\b', slang, text)
    return text


def convert_buffer_references(text):
    """`layout(buffer_reference) buffer FooBuffer { T data[]; };` declares a
    type used as `FooBuffer(addr).data[i]`. slang has no such declaration, so
    the type is dropped and every use becomes a pointer cast."""
    elem = {}
    # a buffer may hold an array (`T data[];`) or a single value (`T data;`)
    single = set()

    # the member is usually called `data`, but not always
    member = {}

    def grab(m):
        elem[m.group(1)] = m.group(2).strip()
        member[m.group(1)] = m.group(3)
        if not m.group(4):
            single.add(m.group(1))
        return ''
    text = re.sub(
        r'[ \t]*layout\(buffer_reference[^)]*\)\s*buffer\s+(\w+)\s*\{\s*'
        r'([\w]+)\s+(\w+)(\[\])?\s*;\s*\}\s*;\n?',
        grab, text)
    for name, ty in shared_buffer_elements().items():
        if name in text and name not in included_buffer_elements():
            elem.setdefault(name, ty)
    for name, (ty, isSingle) in included_buffer_elements().items():
        if name in text and name not in elem:
            elem[name] = ty
            if isSingle:
                single.add(name)
    for name, ty in elem.items():
        # FooBuffer(x).data[i] -> ((T *)(x))[i]; a single-value buffer has no
        # subscript of its own, so it gets [0]
        tail = '[0]' if name in single else ''
        field = member.get(name, 'data')
        text = re.sub(r'\b' + name + r'\(([^;]*?)\)\s*\.\s*' + field + r'\b',
                      lambda m, t=ty, s=tail: f'((%s *)({m.group(1).strip()}))%s' % (t, s), text)
        # `FooBuffer v = FooBuffer(x);` -> `T * v = (T *)(x);`
        text = re.sub(r'\b' + name + r'\s+(\w+)\s*=\s*' + name + r'\(([^;]*?)\)\s*;',
                      lambda m, t=ty: f'{t} * {m.group(1)} = ({t} *)({m.group(2).strip()});', text)
        # the constructor form on its own, `FooBuffer(x)` -> `(T *)(x)`
        text = re.sub(r'\b' + name + r'\(([^;()]*(?:\([^()]*\))?[^;()]*)\)',
                      lambda m, t=ty: f'({t} *)({m.group(1).strip()})', text)
        # remaining bare `FooBuffer x` params/locals -> `T * x`
        text = re.sub(r'\b' + name + r'\s+(\w+)\b', lambda m, t=ty: f'{t} * {m.group(1)}', text)
    # `v.data[i]` on what is now a pointer
    text = re.sub(r'\.data\[', '[', text)
    return text


# topological order of the shader modules. auto-imports may only reach
# backwards along this list, so the symbol resolver cannot build a cycle
# where the glsl include graph had none
# a header carrying macros beyond its include guard stays #include-able:
# macros do not cross a module boundary. the unit-test helpers are mostly
# materialx reference transcriptions, which lean on them heavily
INCLUDE_ONLY = {
    'util-nan-probe',
    'util-emitter-test', 'util-eon-materialx-reference',
    'util-fresnel-materialx-reference', 'util-fuzz-materialx-reference',
    'util-microfacet-materialx-reference', 'util-microfacet-prelude',
    'util-regir-test', 'util-roughness-reference', 'util-sampling-bins',
    'util-shading-frame-reference', 'util-thinfilm-materialx-reference',
}

MODULE_ORDER = [
    'util-types', 'util-mor-shared', 'util-global-pc', 'util-resolve-pc',
    'util-emitter-shared', 'util-regir-shared', 'util-material-openpbr-shared',
    'util-material-openpbr-tables', 'util-fresnel', 'util-roughness',
    'util-mis', 'util-random', 'util-shading-frame',
    'util-energy-compensation', 'util-blackbody', 'util-tonemap',
    'util-fog', 'util-nan-probe', 'util-material-openpbr-thinfilm',
    'util-material-openpbr-microfacet', 'util-material-openpbr-diffuse',
    'util-material-openpbr-metallic', 'util-material-openpbr-coat',
    'util-material-openpbr-fuzz', 'util-material-openpbr-transmission',
    'util-material-openpbr-subsurface', 'util-dielectric-specular',
    'util-material-openpbr', 'util-material-openpbr-ibl',
    'util-environment-map', 'util-mesh', 'util-mesh-raytrace',
    'util-emitter', 'util-vdb',
]


SHADER_DIR = pathlib.Path(__file__).resolve().parent


_OWNER = None


def definition_owner():
    """function name -> earliest module in MODULE_ORDER whose glsl defines it.

    a handful of functions are defined in two glsl files that are never
    included together; slang sees both through one import graph, so only the
    earliest module keeps its copy.
    """
    global _OWNER
    if _OWNER is not None:
        return _OWNER
    _OWNER = {}
    pat = re.compile(r'^(?:' + TYPES + r')\s+(\w+)\s*\(', re.M)
    for mod in MODULE_ORDER:
        f = pathlib.Path('lib/ponder/shaders') / (mod + '.glsl')
        if not f.exists():
            continue
        for fn in pat.findall(f.read_text()):
            _OWNER.setdefault(fn, mod)
    return _OWNER


def end_of_body(text, pos):
    """index just past the function body starting at/after pos.

    skips comments: the doc blocks carry latex like \\begin{align*}, whose
    braces would otherwise unbalance the scan
    """
    i = text.index('{', pos)
    depth = 0
    while i < len(text):
        c = text[i]
        if c == '/' and i + 1 < len(text):
            if text[i + 1] == '/':
                i = text.find('\n', i)
                if i < 0:
                    return len(text)
                continue
            if text[i + 1] == '*':
                i = text.find('*/', i + 2)
                if i < 0:
                    return len(text)
                i += 2
                continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def drop_duplicate_definitions(src, name):
    """a few functions are defined in two glsl files that are never included
    together. slang sees both through one import graph, so only the earliest
    module in MODULE_ORDER exports its copy; the later ones keep theirs but
    make it module-private, which needs no text surgery."""
    owner = definition_owner()
    for fn, mod in owner.items():
        if mod == name:
            continue
        src = re.sub(r'^public func ' + fn + r'\(', 'func ' + fn + '(', src, flags=re.M)
    return src, set()


def transitive_imports(mods):
    """every module reachable from mods through __exported import"""
    seen = set()
    stack = list(mods)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        f = pathlib.Path('lib/ponder/shaders') / (m + '.slang')
        if f.exists():
            stack += re.findall(r'(?:__exported )?import "([^"]+)"', f.read_text())
    return seen


def symbol_index(exclude):
    """public symbols each earlier module exports"""
    index = {}
    rank = MODULE_ORDER.index(exclude) if exclude in MODULE_ORDER else len(MODULE_ORDER)
    for f in sorted(pathlib.Path('lib/ponder/shaders').glob('util-*.slang')):
        mod = f.stem
        if mod in ('util-types', exclude):
            continue
        if mod not in MODULE_ORDER or MODULE_ORDER.index(mod) >= rank:
            continue
        text = f.read_text()
        syms = set(re.findall(r'^public (?:func|struct) (\w+)', text, re.M))
        syms |= set(re.findall(r'^public static const \w+ (\w+)', text, re.M))
        if syms:
            index[mod] = syms
    return index


# the dual headers with a generated slang view; keep in step with CMake's
# slang-shared-headers list
SHARED_HEADER_MODULES = {
    'mor-shared.h': 'util-mor-shared',
    'emitter-shared.h': 'util-emitter-shared',
    'util-regir-shared.h': 'util-regir-shared',
    'global_pc.h': 'util-global-pc',
    'resolve_pc.h': 'util-resolve-pc',
    # hand-ported rather than generated, but the mapping is the same
    'util-material-openpbr-shared.h': 'util-material-openpbr-shared',
    'bloom-blur-pc.h': 'util-bloom-blur-pc',
    'bloom-composite-pc.h': 'util-bloom-composite-pc',
    'bloom-extract-pc.h': 'util-bloom-extract-pc',
    'nan-check-pc.h': 'util-nan-check-pc',
    'pt-accumulate-pc.h': 'util-pt-accumulate-pc',
    'pt-denoise-pc.h': 'util-pt-denoise-pc',
    'manifold-shade-pc.h': 'util-manifold-shade-pc',
    'regir-pc.h': 'util-regir-pc',
    'scene-visibility-resolve-pc.h': 'util-scene-visibility-resolve-pc',
    'global-pc.h': 'util-observer-global-pc',
    'env-prefilter-pc.h': 'util-env-prefilter-pc',
}


MACROS_SUFFIX = '-macros.slang'


def macro_companion_names(module):
    """the macro names a generated header's companion file defines, if any"""
    path = pathlib.Path('lib/ponder/shaders') / (module + MACROS_SUFFIX)
    if not path.exists():
        return []
    return re.findall(r'^#define (\w+)\(', path.read_text(), flags=re.M)


def blank_comments(text):
    """comments replaced by whitespace, so offsets and line counts hold.

    the import heuristic searches for symbol names, and a name mentioned in a
    doc comment would otherwise pull in a whole module the code never calls
    """
    return ''.join(_blank_span(text))


def _blank_span(text):
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '/' and i + 1 < n and text[i + 1] in '/*':
            if text[i + 1] == '/':
                j = text.find('\n', i)
                j = n if j < 0 else j
            else:
                j = text.find('*/', i + 2)
                j = n if j < 0 else j + 2
            yield ''.join(' ' if c != '\n' else '\n' for c in text[i:j])
            i = j
            continue
        yield text[i]
        i += 1


def publicize_structs(text):
    """`struct X { T a; }` -> `public struct X { public T a; }`.

    the body is found by brace matching rather than by regex: a doc comment
    with `\\hat{h}` in it would end a non-greedy match at the wrong brace and
    silently leave the rest of the fields private
    """
    out = []
    i = 0
    for m in list(re.finditer(r'\bstruct\s+(\w+)\s*\{', text)):
        if m.start() < i:
            continue
        close = end_of_body(text, m.start()) - 1
        body = text[m.end():close]
        body = re.sub(r'^(\t)(?!//|public|\})', r'\1public ', body, flags=re.M)
        out.append(text[i:m.start()])
        out.append(f'public struct {m.group(1)} {{{body}}}')
        i = close + 1
    out.append(text[i:])
    return ''.join(out)


def convert(path_in):
    src = pathlib.Path(path_in).read_text()
    name = pathlib.Path(path_in).stem
    # strip the include guard and the alias #define block
    src = re.sub(r'#ifndef ([A-Z_0-9]+)\n#define \1\n', '', src, count=1)
    src = re.sub(r'#endif\s*(//[^\n]*)?\s*$', '', src)
    src = re.sub(r'#ifndef f32\n(?:#define \w+ \w+\n)+#endif\n', '', src)
    imports = re.findall(r'#include "([^"]+)\.glsl"', src)
    hdrs = re.findall(r'#include "([^"]+\.h)"', src)
    src = re.sub(r'#include "[^"]+"\n', '', src)
    # vkof's set-0 bindings live in util-types; a module that redeclares one
    # makes every use of it ambiguous
    src = re.sub(r'^layout\(set = 0[^)]*\)\s*uniform[^;]*;\n', '', src, flags=re.M)
    # an include-only header may declare the push constant its includers
    # share; a module cannot, so only that form keeps it
    pushConstant = ''
    if name in INCLUDE_ONLY:
        src, pushConstant = take_push_constant(src, lambda t: t)
    collect_mutable_params(src)
    src = fix_ray_query(src)
    src = convert_buffer_references(src)
    src = convert_signatures(src)
    def rename_builtins(chunk):
        for a, b in BUILTIN:
            chunk = re.sub(a, b, chunk)
        for a, b in TYPEMAP:
            chunk = re.sub(a, b, chunk)
        return chunk
    src = map_outside_comments(src, rename_builtins)
    src = fix_matrix_multiply(src)
    src = shadow_mutated_params(src)
    src = fix_atan2(src)
    src = fix_atomics(src)
    src = fix_vector_equality(src)
    src = fix_wide_matrix_index(src)
    # glsl vector comparisons are functions; slang uses the operators
    def rewrite_comparisons(chunk):
        chunk = re.sub(r'\blessThan\(([^,]+),\s*', r'(\1 < ', chunk)
        chunk = re.sub(r'\bgreaterThan\(([^,]+),\s*', r'(\1 > ', chunk)
        chunk = re.sub(r'\blessThanEqual\(([^,]+),\s*', r'(\1 <= ', chunk)
        chunk = re.sub(r'\bgreaterThanEqual\(([^,]+),\s*', r'(\1 >= ', chunk)
        chunk = re.sub(r'\bequal\(([^,]+),\s*', r'(\1 == ', chunk)
        chunk = re.sub(r'\bnotEqual\(([^,]+),\s*', r'(\1 != ', chunk)
        return re.sub(r'\bnot\(([^()]*(?:\([^()]*\))?[^()]*)\)', r'(!(\1))', chunk)
    src = map_outside_comments(src, rewrite_comparisons)
    # glsl mix(a, b, bvec) is a select, not a lerp. slang's lerp would take
    # the bool through an implicit 0/1 conversion: numerically the same here
    # but wrong in spirit, and different on nan operands. runs last so the
    # selector has already become `(!(x))`
    src = fix_bool_mix(src)
    # struct members and the struct itself become public
    src = publicize_structs(src)
    # a few functions are defined in two glsl files that are never included
    # together; slang sees both through one import graph, so the later
    # module drops its copy and imports the earlier one instead
    src, dupOwners = drop_duplicate_definitions(src, name)
    imports += [m for m in dupOwners if m not in imports]
    if name in INCLUDE_ONLY:
        # macros do not cross a module boundary, so this one is #included
        header = '#pragma once\n__exported import "util-types";\n'
    else:
        header = f'module "{name}";\n__exported import "util-types";\n'
    kept = []
    for imp in imports:
        if imp in INCLUDE_ONLY:
            header += f'#include "{imp}.slang"\n'
            continue
        if imp in transitive_imports(kept):
            continue
        header += f'__exported import "{imp}";\n'
        kept.append(imp)
    imports = kept
    # glsl relied on include order for symbols it never included; slang needs
    # an explicit import, so resolve them against what the other modules
    # export. anything already reachable through an __exported import is
    # skipped: importing it again makes every call ambiguous
    # a plain import still propagates symbols transitively, so importing a
    # module that is already reachable makes every shared call ambiguous
    reachable = transitive_imports(imports)
    # a shared header with a generated slang view is an import, not a todo
    for h in hdrs:
        mod = SHARED_HEADER_MODULES.get(h.rsplit('/', 1)[-1])
        if mod is None or mod in reachable or mod == name:
            continue
        header += f'__exported import "{mod}";\n'
        reachable |= transitive_imports([mod]) | {mod}
    code = blank_comments(src)
    # the import carries the structs; the macros only arrive by #include
    for h in hdrs:
        mod = SHARED_HEADER_MODULES.get(h.rsplit('/', 1)[-1])
        if mod is None:
            continue
        names = macro_companion_names(mod)
        if any(re.search(r'\b' + n + r'\b', code) for n in names):
            header += f'#include "{mod}{MACROS_SUFFIX}"\n'
    for mod, syms in sorted(symbol_index(name).items(),
                            key=lambda kv: MODULE_ORDER.index(kv[0])):
        if mod == name or mod in reachable:
            continue
        if mod in INCLUDE_ONLY:
            continue
        if any(re.search(r'\b' + sym + r'\b', code) for sym in syms):
            header += f'__exported import "{mod}";\n'
            reachable |= transitive_imports([mod]) | {mod}
    # a value-only #define is invisible to an importer, so it becomes a real
    # constant. function-like and expression ones are left alone
    # `#ifndef X / #define X ...` is a provide-if-absent guard, and util-types
    # already exports PI/TAU/IPI; converting one of those to a constant makes
    # every use ambiguous
    guarded = set(re.findall(r'^#ifndef (\w+)\n#define \1\b', src, flags=re.M))
    # util-types owns these; a module that redefines one, guarded or not,
    # makes every use ambiguous
    src = re.sub(r'^#define (?:PI|TAU|IPI)\s+[^\n]*\n', '', src, flags=re.M)

    def define_to_const(m, ty, suffix):
        if m.group(1) in guarded:
            return m.group(0)
        return f'public static const {ty} {m.group(1)} = {m.group(2)}{suffix};'
    src = re.sub(r'^#define (\w+) (\d+)u?\s*$',
                 lambda m: define_to_const(m, 'u32', 'u'), src, flags=re.M)
    src = re.sub(r'^#define (\w+) (-?\d*\.\d+(?:e-?\d+)?f?)\s*$',
                 lambda m: define_to_const(m, 'f32', ''), src, flags=re.M)
    # an expression-valued define over literals and constants is still a
    # value, and a macro would not reach an importer
    src = re.sub(r'^#define (\w+) (\([^\n]*\))\s*$',
                 lambda m: define_to_const(m, 'f32', ''), src, flags=re.M)
    # PI/TAU/IPI live in util-types so every module sees one definition
    # glsl file-scope `const` is slang `static const`
    src = re.sub(r'^const\s+(' + TYPES + r')\s+(\w+)\s*=', r'public static const \1 \2 =', src, flags=re.M)
    # a mutable file-scope variable is per-invocation state in both
    # languages, and slang needs it spelled `static`
    src = re.sub(r'^(' + TYPES + r')\s+(g[A-Z]\w*)\s*=',
                 r'public static \1 \2 =', src, flags=re.M)
    src = re.sub(r'^shared\s+', 'groupshared ', src, flags=re.M)
    src = re.sub(r'\bbarrier\(\)', 'GroupMemoryBarrierWithGroupSync()', src)
    src = re.sub(r'\bmemoryBarrierShared\(\)', 'GroupMemoryBarrier()', src)
    for h in hdrs:
        if h.rsplit('/', 1)[-1] not in SHARED_HEADER_MODULES:
            header += f'// TODO port shared header: {h}\n'
    return header + (('\n' + pushConstant) if pushConstant else '') + src

if __name__ == '__main__':
    sys.stdout.write(convert(sys.argv[1]))
