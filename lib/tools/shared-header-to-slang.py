#!/usr/bin/env python3
"""Generates a Slang module from a dual C++/GLSL shared header.

The `#ifdef __cplusplus` headers are the one thing Slang cannot consume: it
has no `layout(buffer_reference)` and no `layout(offset=)`. Hand-mirroring
them is a silent-failure risk, since a field reordered within the same size
scrambles values with no error, so the Slang view is generated instead.

    shared-header-to-slang.py <header.h> <out.slang> [--module NAME]
                              [--macros-dir DIR]

What it does:
  * keeps the GLSL branch of every `#ifdef __cplusplus`
  * function-like macros move to a `<module>-macros.slang` companion, since
    a macro does not cross a module boundary; a consumer #includes it next
    to its import
  * `#define FOO 3`                  -> `public static const u32 FOO = 3u;`
  * `struct X { T a; };`             -> `public struct X { public T a; };`
  * `VA(T) name;`                    -> `u64 name;` (a device address)
  * `layout(buffer_reference) buffer XBuffer { T data[]; };`
                                     -> dropped; slang casts the address to T*
  * `layout(push_constant) uniform PC {...} pc;`
                                     -> `struct PC {...}; [[vk::push_constant]] PC pc;`
  * `static_assert(...)`             -> dropped, C++ only

Buffer element types are written to a sidecar `.json` so the shader converter
can rewrite `XBuffer(addr).data[i]` into `((T *)addr)[i]`.
"""
import json
import pathlib
import re
import sys

# the companion file holding a header's function-like macros
MACROS_SUFFIX = '-macros.slang'

# spellings the shared headers use, mapped to what util-types exports
TYPE_ALIASES = {
    'float': 'f32', 'int': 'i32', 'uint': 'u32',
    'vec2': 'f32v2', 'vec3': 'f32v3', 'vec4': 'f32v4',
    'ivec2': 'i32v2', 'ivec3': 'i32v3', 'ivec4': 'i32v4',
    'uvec2': 'u32v2', 'uvec3': 'u32v3', 'uvec4': 'u32v4',
    'mat3': 'f32m3x3', 'mat4': 'f32m4x4',
}


def strip_cplusplus(text):
    """keep the GLSL side of every __cplusplus conditional.

    other conditionals pass through untouched, directives included, so a
    nested `#ifndef FOO` keeps its own `#endif`
    """
    out = []
    stack = []
    for ln in text.split('\n'):
        s = ln.strip()
        m = re.match(r'#(ifdef|ifndef)\s+(\w+)', s)
        if m:
            if m.group(2) == '__cplusplus':
                stack.append(['cpp' if m.group(1) == 'ifdef' else 'glsl', True])
                continue
            stack.append(['pass', False])
            if not any(e[0] == 'cpp' for e in stack):
                out.append(ln)
            continue
        if re.match(r'^#else\b', s) and stack:
            top = stack[-1]
            if top[1]:
                top[0] = 'glsl' if top[0] == 'cpp' else 'cpp'
                continue
        if re.match(r'^#endif\b', s) and stack:
            top = stack.pop()
            if top[1]:
                continue
            if not any(e[0] == 'cpp' for e in stack):
                out.append(ln)
            continue
        if any(e[0] == 'cpp' for e in stack):
            continue
        out.append(ln)
    return '\n'.join(out)


def strip_guard_and_aliases(text):
    text = re.sub(r'#ifndef ([A-Z_0-9]+)\n#define \1\n', '', text, count=1)
    # the `#ifndef f32 ... #endif` alias block is replaced by util-types
    text = re.sub(r'#ifndef f32\n(?:#define [^\n]*\n)+#endif\n', '', text)
    text = re.sub(r'#define (?:i32|u32|f32|VA)\b[^\n]*\n', '', text)
    text = re.sub(r'#endif\s*(//[^\n]*)?\s*$', '', text)
    return text


def take_buffer_references(text):
    """drop the reference-type declarations, returning their element types"""
    elements = {}

    def grab(m):
        elements[m.group(1)] = m.group(2).strip()
        return ''

    text = re.sub(
        r'layout\(buffer_reference[^)]*\)\s*buffer\s+(\w+)\s*\{\s*'
        r'([\w]+)\s+data(?:\[\])?\s*;\s*\}\s*;',
        grab, text)
    return text, elements


def drop_push_constant(text):
    """a module cannot declare the push constant an entry point supplies, so
    the block is dropped; the entry point declares its own using the structs
    generated here. the `#ifndef *_NO_PC` guard around it goes too, since it
    exists only to let a glsl shader opt out of the same declaration."""
    text = re.sub(
        r'layout\(push_constant[^)]*\)\s*uniform\s+\w+\s*\{.*?\}\s*\w+\s*;',
        '', text, flags=re.S)
    text = re.sub(r'^#ifndef \w*_NO_PC\s*\n', '', text, flags=re.M)
    return text


def publicize(text):
    def pub(m):
        head, body = m.group(1), m.group(2)
        body = re.sub(r'^(\t+)(?=[A-Za-z_])', r'\1public ', body, flags=re.M)
        return f'public struct {head} {{{body}'
    return re.sub(r'\bstruct\s+(\w+)\s*\{(.*?)\n\}', lambda m: pub(m) + '\n}', text,
                  flags=re.S)


# a define body slang can evaluate: literals, already-converted constants and
# arithmetic over them. anything else stays a macro
SK_INT_LITERAL = re.compile(r'^(?:0[xX][0-9a-fA-F]+|\d+)[uU]?$')
SK_FLOAT_LITERAL = re.compile(r'^-?(?:\d+\.\d*|\.?\d+)(?:[eE][-+]?\d+)?[fF]?$')


def logical_define_lines(text):
    """splits text into lines, joining a `\\`-continued #define into one"""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('#define'):
            while line.endswith('\\') and i + 1 < len(lines):
                i += 1
                line = line[:-1].rstrip() + ' ' + lines[i].strip()
        out.append(line)
        i += 1
    return out


def define_constant_type(body, known):
    """u32/f32 for a body slang can evaluate, None to leave it a macro"""
    tokens = re.findall(r'[A-Za-z_]\w*|[0-9][\w.+-]*|\S', body)
    if not tokens:
        return None
    ty = 'u32'
    for t in tokens:
        if t in '()+-*/%<>&|^~ ':
            continue
        if SK_INT_LITERAL.match(t):
            continue
        if SK_FLOAT_LITERAL.match(t):
            ty = 'f32'
            continue
        if t in known:
            if known[t] == 'f32':
                ty = 'f32'
            continue
        return None
    return ty


def convert_defines(text):
    """value and expression #defines become real constants.

    a macro does not cross a module boundary, so anything an importer needs
    has to be a constant. function-like macros are left alone.
    """
    known = {}
    out = []
    for line in logical_define_lines(text):
        m = re.match(r'^#define (\w+)(?:\s+(.*?))?\s*$', line)
        if not m or line[len('#define ') + len(m.group(1)):len('#define ') + len(m.group(1)) + 1] == '(':
            out.append(line)
            continue
        name, body = m.group(1), (m.group(2) or '').strip()
        ty = define_constant_type(body, known) if body else None
        if ty is None:
            # `#define X X` is a header's own re-entry guard, not a constant
            if body and body != name:
                print(f'  note: {name} left as a macro '
                      f'(not visible to importers)', file=sys.stderr)
            out.append(line)
            continue
        known[name] = ty
        if ty == 'u32' and not body.rstrip().endswith(('u', 'U')) \
                and SK_INT_LITERAL.match(body):
            body += 'u'
        out.append(f'public static const {ty} {name} = {body};')
    return '\n'.join(out)


def take_function_macros(text):
    """splits the function-like macros out of the module body.

    a macro does not cross a module boundary, so these go to a companion
    file the consumer #includes next to its import. convert_defines has
    already joined every continuation, so each one is a single line
    """
    kept = []
    macros = []
    for line in text.split('\n'):
        m = re.match(r'^#define (\w+)\(', line)
        if m:
            macros.append(line)
            continue
        kept.append(line)
    return '\n'.join(kept), macros


def convert(path_in, module_name):
    src = pathlib.Path(path_in).read_text()
    src = strip_cplusplus(src)
    src = strip_guard_and_aliases(src)
    src = re.sub(r'static_assert\([^;]*\);\n?', '', src)
    src, elements = take_buffer_references(src)

    # a VA(T) field is a device address; the element type is kept in a
    # comment so a reader knows what to cast it to
    src = re.sub(r'\bVA\((\w+)\)\s+(\w+);', r'u64 \2; // VA(\1)', src)

    src = convert_defines(src)
    src = re.sub(r'^#include [^\n]*\n', '', src, flags=re.M)

    for glsl, slang in TYPE_ALIASES.items():
        src = re.sub(r'\b' + glsl + r'\b', slang, src)

    src = re.sub(r'^const\s+(\w+)\s+(\w+)\s*=',
                 r'public static const \1 \2 =', src, flags=re.M)
    src = drop_push_constant(src)
    src = publicize(src)
    src, macros = take_function_macros(src)
    src = re.sub(r'\n{3,}', '\n\n', src).strip()

    provenance = (
        f'// autogen from {pathlib.Path(path_in).name} by\n'
        f'// lib/tools/shared-header-to-slang.py\n'
    )
    header = (
        f'{provenance}\n'
        f'module "{module_name}";\n'
        f'__exported import "util-types";\n\n'
    )
    macrosText = None
    if macros:
        macrosText = (
            f'{provenance}//\n'
            f'// the function-like macros only. a macro does not cross a\n'
            f'// module boundary, so a consumer #includes this next to its\n'
            f'// `import "{module_name}"`\n'
            f'#pragma once\n\n'
            + '\n'.join(macros) + '\n'
        )
    return header + src + '\n', elements, macrosText


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    module = None
    macros_dir = None
    for i, a in enumerate(sys.argv):
        if a == '--module':
            module = sys.argv[i + 1]
        if a == '--macros-dir':
            macros_dir = sys.argv[i + 1]
    args = [a for a in args if a not in (module, macros_dir)]
    path_in, path_out = args[0], args[1]
    if module is None:
        module = pathlib.Path(path_out).stem
    text, elements, macrosText = convert(path_in, module)
    pathlib.Path(path_out).write_text(text)
    if elements:
        pathlib.Path(path_out).with_suffix('.buffers.json').write_text(
            json.dumps(elements, indent=2) + '\n')
    # the companion lands straight in the shader directory, written only when
    # it changed so it does not retrigger every dependent build
    macrosDir = pathlib.Path(macros_dir or pathlib.Path(path_out).parent)
    macrosPath = macrosDir / (module + MACROS_SUFFIX)
    if macrosText:
        if not macrosPath.exists() or macrosPath.read_text() != macrosText:
            macrosPath.write_text(macrosText)
    elif macrosPath.exists():
        macrosPath.unlink()
    print(f'{path_in} -> {path_out} ({len(elements)} buffer types'
          f'{", macros" if macrosText else ""})')


if __name__ == '__main__':
    main()
