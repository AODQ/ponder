# CLAUDE.md

This file provides guidance to Claude Code

Claude/AI are only allowed to edit files in unit-test, and must follow below
code style

### Naming

| Category | Convention | Examples |
|---|---|---|
| Types / structs | `PascalCase` | `HandlePool`, `BufferCreateInfo`, `AllocVirtualRange` |
| Free functions | `snake_case` | `buffer_create`, `render_node_add_image`, `handle_make` |
| Member functions | `camelCase` | `isIndexAlive`, `allocatedCount`, `printAllocationStats` |
| Struct fields / locals | `camelCase` | `byteCount`, `elementCount`, `debugName` |
| Static constexpr | `sk` prefix + `PascalCase` | `skTileSize`, `skSliceDynamicExtent`, `skEpsilon` |
| Enum values | `snake_case` or `camelCase` | `write_on_test_off`, `readWrite`, `DeviceOnly` |
| File names | `kebab-case` with dashes, never underscores | `util-ggx-sample.glsl`, `core-types.hpp`, `main.cpp` |
| Shader util headers | `util-` prefix + `kebab-case` | `util-ggx-sample.glsl`, `util-taa-filter.glsl` |

### Formatting

- **Indentation:** tabs.
- **Brace placement:** opening brace on a new line for `struct`/`class`/`namespaec` bodies. For functions, if parameters break into a new line, the brace can also go on a new line. Otherwise braces always go on same line.
- **`const` placement:** east const — `u32 const maxHandles`, `char const * debugName`, `Handle const & handle`.
- **`const`** ALWAYS **ALWAYS** use const. Add const to everything including function parameters. Treat code as SSA, immutable, etc - except in cases where it makes sense to mutate.
- The one exception to const is GLSL buffer references, those actually can not be const and will result in compile error
- **Pointer spacing:** `T * ptr`, `T * const ptr` — space before and after `*` or `&`. Treat them as keywords that require spaces.
- **`alignment`** do not ever align parameters or struct fields. This means NO aligning `=` signs, NO aligning `:` separators, NO aligning `->`, NO padding any tokens to form columns — in C++, GLSL, or any other language. A single space before and after the token is all that is ever used.
- prefer blocked indentation everywhere.
- **`comments`** never place a comment on the same line as code. comments always go on their own line, above the code they describe.
- **`indent parameter blocks`** if a list of items -- for a function call, function signature, if statement, etc -- breaks into multiple lines, then break into a new line and indent the block. Do not ever align them.
- **`multi-line expression wrapping`** when breaking a long expression across lines, the outer parens must wrap the *entire* expression including any trailing operator, division, or comparison. Nothing is left outside the closing paren. Example — wrong: `return vec3(...\n) / 255.0;` — correct: `return (\n    vec3(...) / 255.0\n);`
- to re-iterate the above as its crucially important, if you break a line into multiple lines of code, you must wrap it in parens, the first parens on the same line as the break, and the last parens on its own line at the end. treat breaking lines into parens both alignment/indentation-wise and the placement of parens exactly like a new code-block with braces: the brace placement and the align/indent should match in both cases.

### C++ idioms

- Use auto and type inference only if the type is obvious from the right-hand side, such as iterators or `auto it = static_cast<Foo *>(ptr)`. Otherwise, prefer explicit types for readability.
- Small structs used as named-parameter bundles passed by `const &` (the `*CreateInfo` / `*Info` pattern) instead of long argument lists.
- `static` factory methods (`AllocVirtualRange::create`, `HandlePool::create`) rather than constructors when setup can fail or requires non-trivial allocation.
