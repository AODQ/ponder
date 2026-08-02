#!/usr/bin/env python3
"""GLSL entry point (.comp) -> Slang, for the shader port's stage 6.

    comp2slang.py <shader.comp> > <shader.slang>

Reuses glsl2slang for the body; what is entry-point specific lives here:

  * `#version` / `#extension` lines          -> dropped
  * `#include "util-x.glsl"`                 -> `import "util-x";`
  * `layout(push_constant) uniform PC {...}` -> `struct PC {...};` plus a
    `[[vk::push_constant]]` global. `layout(offset = 128)` on the first
    field becomes a `u32v4 rootPad[8]`, matching the root-constant
    reservation vkof makes
  * `layout(local_size_x = N) in;`           -> `[numthreads(N, 1, 1)]`
  * `void main()`                            -> a compute entry point whose
    parameters are *named after* the gl_ builtins the body already uses, so
    the body needs no edit at all

The body conversion is glsl2slang's, so a converter fix reaches both.
"""
import importlib.util
import pathlib
import re
import sys

_spec = importlib.util.spec_from_file_location(
    'glsl2slang', str(pathlib.Path(__file__).parent / 'glsl2slang.py'))
g2s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g2s)

# the gl_ builtins an entry point can take, and the semantic each maps to.
# the parameter keeps the glsl name so the body is untouched
BUILTIN_SEMANTICS = [
    ('gl_GlobalInvocationID', 'SV_DispatchThreadID'),
    ('gl_LocalInvocationID', 'SV_GroupThreadID'),
    ('gl_WorkGroupID', 'SV_GroupID'),
    ('gl_LocalInvocationIndex', 'SV_GroupIndex'),
]


# the entry points spell these out where the library headers use the aliases
ENTRY_TYPEMAP = [
    (r'\buint64_t\b', 'u64'), (r'\bint64_t\b', 'i64'),
    (r'\buint32_t\b', 'u32'), (r'\bint32_t\b', 'i32'),
]


def retype(text):
    for a, b in ENTRY_TYPEMAP + g2s.TYPEMAP:
        text = re.sub(a, b, text)
    return text


def take_workgroup_size(text):
    """`layout(local_size_x = N) in;` -> the numthreads attribute"""
    m = re.search(r'layout\(\s*local_size_x\s*=\s*(\d+)\s*'
                  r'(?:,\s*local_size_y\s*=\s*(\d+)\s*)?'
                  r'(?:,\s*local_size_z\s*=\s*(\d+)\s*)?\)\s*in\s*;', text)
    if not m:
        return text, '[numthreads(64, 1, 1)]'
    x = m.group(1)
    y = m.group(2) or '1'
    z = m.group(3) or '1'
    return (text[:m.start()] + text[m.end():],
            f'[numthreads({x}, {y}, {z})]')


def entry_signature(body, numthreads):
    """the entry point's parameters are the gl_ builtins the body uses"""
    params = [
        f'\tconst u32v3 {name} : {semantic},'
        for name, semantic in BUILTIN_SEMANTICS
        if re.search(r'\b' + name + r'\b', body)
    ]
    if not params:
        return f'[shader("compute")]\n{numthreads}\nvoid main()'
    joined = '\n'.join(params)
    return (
        f'[shader("compute")]\n{numthreads}\n'
        f'void main(\n{joined}\n)'
    )


def convert_array_initialisers(text):
    """`T[n](a, b, c)` is `{ a, b, c }` in slang, closing brace included"""
    pattern = re.compile(r'=\s*' + g2s.TYPES + r'\[\d*\]\(')
    while True:
        m = pattern.search(text)
        if not m:
            return text
        depth = 0
        i = m.end() - 1
        while i < len(text):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        text = text[:m.start()] + '= {' + text[m.end():i] + '}' + text[i + 1:]


SEARCH_DIRS = [
    'unit-test/shaders/slang', 'lib/ponder/shaders', 'app/viewer/shaders',
]


def is_include_only(module):
    """a generated module that opens with `#pragma once` is #included"""
    for d in SEARCH_DIRS:
        path = pathlib.Path(d) / (module + '.slang')
        if path.exists():
            return path.read_text().lstrip().startswith('#pragma once')
    return False


def convert(path_in):
    src = pathlib.Path(path_in).read_text()
    src = re.sub(r'^#version[^\n]*\n', '', src, flags=re.M)
    src = re.sub(r'^#extension[^\n]*\n', '', src, flags=re.M)
    imports = re.findall(r'#include "([^"]+)\.glsl"', src)
    hdrs = re.findall(r'#include "([^"]+\.h)"', src)
    src = re.sub(r'#include "[^"]+"\n', '', src)

    # vkof's set-0 bindings are declared once in util-types; an entry point
    # that redeclares them makes every use ambiguous
    src = re.sub(r'^layout\(set = 0[^)]*\)\s*uniform[^;]*;\n', '', src, flags=re.M)
    # util-types owns PI/TAU/IPI; an entry point redefining one shadows or
    # clashes with it
    src = re.sub(r'^#define (?:PI|TAU|IPI)\s+[^\n]*\n', '', src, flags=re.M)
    src = re.sub(r'^#ifndef (?:PI|TAU|IPI)\n#endif\n', '', src, flags=re.M)
    src, pushConstant = g2s.take_push_constant(src, retype)
    if not pushConstant:
        # a per-pass header often owns the block; the generated module drops
        # it, so the entry point declares it here
        for h in hdrs:
            for d in SEARCH_DIRS + ['app/observer/shaders']:
                hp = pathlib.Path(d) / h.rsplit('/', 1)[-1]
                if not hp.exists():
                    continue
                text = hp.read_text()
                _, pushConstant = g2s.take_push_constant(text, retype)
                if pushConstant:
                    hdrs += [x for x in re.findall(r'#include "([^"]+\.h)"', text)
                             if x not in hdrs]
                    break
            if pushConstant:
                break
    src, numthreads = take_workgroup_size(src)
    # hide main's signature from the library converter, which would rewrite
    # it into a `public func` like any other
    if not re.search(r'^void main\(\)\s*\{', src, re.M):
        raise SystemExit(f'{path_in}: no `void main()` to convert')
    src = re.sub(r'^void main\(\)\s*\{', '__ENTRY_MAIN__ {', src, count=1, flags=re.M)
    src = g2s.fix_ray_query(src)
    src = g2s.convert_buffer_references(src)
    src = g2s.convert_signatures(src)

    def rename_builtins(chunk):
        for a, b in g2s.BUILTIN:
            chunk = re.sub(a, b, chunk)
        return retype(chunk)
    src = g2s.map_outside_comments(src, rename_builtins)
    src = g2s.fix_matrix_multiply(src)
    src = g2s.shadow_mutated_params(src)
    src = g2s.fix_atan2(src)
    src = g2s.fix_atomics(src)
    src = g2s.fix_vector_equality(src)
    src = g2s.fix_wide_matrix_index(src)

    def rewrite_comparisons(chunk):
        chunk = re.sub(r'\blessThan\(([^,]+),\s*', r'(\1 < ', chunk)
        chunk = re.sub(r'\bgreaterThan\(([^,]+),\s*', r'(\1 > ', chunk)
        chunk = re.sub(r'\blessThanEqual\(([^,]+),\s*', r'(\1 <= ', chunk)
        chunk = re.sub(r'\bgreaterThanEqual\(([^,]+),\s*', r'(\1 >= ', chunk)
        chunk = re.sub(r'\bequal\(([^,]+),\s*', r'(\1 == ', chunk)
        chunk = re.sub(r'\bnotEqual\(([^,]+),\s*', r'(\1 != ', chunk)
        return re.sub(r'\bnot\(([^()]*(?:\([^()]*\))?[^()]*)\)', r'(!(\1))', chunk)
    src = g2s.map_outside_comments(src, rewrite_comparisons)
    src = g2s.fix_bool_mix(src)
    src = re.sub(r'^shared\s+', 'groupshared ', src, flags=re.M)
    src = re.sub(r'\bbarrier\(\)', 'GroupMemoryBarrierWithGroupSync()', src)
    src = re.sub(r'\bmemoryBarrierShared\(\)', 'GroupMemoryBarrier()', src)
    src = g2s.publicize_structs(src)
    # glsl file-scope `const` is slang `static const`; an array initialiser
    # loses its `T[n](...)` spelling too
    src = re.sub(r'^const\s+(' + g2s.TYPES + r')\s+(\w+)(\[\d*\])?\s*=',
                 r'static const \1 \2\3 =', src, flags=re.M)
    src = convert_array_initialisers(src)
    src = re.sub(r'\bdebugPrintfEXT\(', 'printf(', src)
    # an entry point exports nothing, so the module-level `public` the library
    # converter adds is noise here
    src = re.sub(r'^public (func|struct) ', r'\1 ', src, flags=re.M)
    src = re.sub(r'^(\t)public ', r'\1', src, flags=re.M)

    src = src.replace('__ENTRY_MAIN__', entry_signature(src, numthreads), 1)

    header = ''
    for imp in imports:
        # a macro-carrying helper stays #include-able; macros do not cross
        # a module boundary
        if is_include_only(imp):
            # relative to this file, so a test does not have to add the
            # helper directory to its includePaths
            prefix = '' if (pathlib.Path('lib/ponder/shaders') / (imp + '.slang')).exists() else '../'
            header += f'#include "{prefix}{imp}.slang"\n'
        else:
            header += f'import "{imp}";\n'
    for h in hdrs:
        mod = g2s.SHARED_HEADER_MODULES.get(h.rsplit('/', 1)[-1])
        if mod:
            header += f'import "{mod}";\n'
            names = g2s.macro_companion_names(mod)
            if any(re.search(r'\b' + n + r'\b', src) for n in names):
                header += f'#include "{mod}{g2s.MACROS_SUFFIX}"\n'
        else:
            header += f'// TODO port shared header: {h}\n'
    if not imports and not hdrs:
        header = 'import "util-types";\n'
    src = re.sub(r'\n{3,}', '\n\n', src).strip()
    return header + '\n' + pushConstant + '\n' + src + '\n'


if __name__ == '__main__':
    sys.stdout.write(convert(sys.argv[1]))
