# Node-graph render graph library — implementation plan

A standalone library that stores a render graph as data (nodes, pins, edges), validates it,
compiles it into an ordered pass list, runs that list on vkof, and edits it through an
ImGui node editor.

Out of scope: moving `app/viewer` or `app/observer` onto the library, and any
viewer-specific pass types. This plan ends when the library works on its own and has been
proven in `unit-test/`.

Working name: `lib/rgraph` (placeholder, see §9).

Division of labour: **you write `lib/`; I write `unit-test/`.** The tests in §8 are
specified as a contract.

---

## 1. Layering

Three layers, each depending only on the ones above it. This mirrors `lib/stage`, which
keeps `stage-imgui.cpp` separate from `stage.cpp`.

| file | depends on | contents |
|---|---|---|
| `rgraph.cpp` | srat, rapidjson | registry, document, validation, compile, JSON |
| `rgraph-vkof.cpp` | + vkof | turns a `CompiledGraph` into `render_node_*` calls |
| `rgraph-imgui.cpp` | + imgui, imnodes | the editor widget |

The core layer knows nothing about Vulkan or ImGui, which is what lets most of §8 run
without a GPU.

---

## 2. Data model

### 2.1 Pass types (registered in C++)

```cpp
enum struct PinKind { image, buffer, externalImage, };
enum struct PinDir  { in, out, };

struct PinDesc {
	std::string name;
	PinKind kind;
	PinDir dir;
	// in-pins only; an unconnected required pin is a compile error
	bool required;
	// out-pins only; used to create the transient image
	vkof::ImageFormat format;
	f32 scaleWidth, scaleHeight;
	u32 mipLevels;
	// out-pins only; access mode of the in-pin with the same name, for readWrite pairs
	std::string readWriteOf;
};

enum struct ParamKind { f32, i32, u32, boolean, enumeration, };
struct ParamDesc {
	std::string name;
	ParamKind kind;
	f32 min, max;
	ParamValue defaultValue;
	std::vector<std::string> enumLabels;
};

struct PassContext;  // resolved pin handles + param values, see §6

struct PassTypeDesc {
	std::string name;
	vkof::CommandQueue queue;
	std::vector<PinDesc> pins;
	std::vector<ParamDesc> params;
	// declared passthrough when bypassed: out-pin name -> in-pin name
	std::vector<std::pair<std::string, std::string>> bypass;
	std::function<void(PassContext const &, vkof::CommandBuffer const &)> record;
};

PassType pass_type_register(PassTypeDesc const & desc);
```

Pass types are identified by `name` in documents, so renaming a type breaks saved graphs.
That's intended; see the unknown-type handling in §5.

### 2.2 Graph document (the editable data)

```cpp
struct Node {
	u32 id;             // stable, never reused within a document
	std::string type;   // PassTypeDesc::name
	std::string label;  // user-facing, defaults to type
	bool bypassed;
	std::vector<std::pair<std::string, ParamValue>> params;
	f32v2 editorPos;
};

struct Edge {
	u32 srcNode; std::string srcPin;
	u32 dstNode; std::string dstPin;
};

struct HistoryResource {
	u32 id;
	std::string name;
	vkof::ImageFormat format;
	f32 scaleWidth, scaleHeight;
};

struct GraphDocument {
	std::vector<Node> nodes;
	std::vector<Edge> edges;
	std::vector<HistoryResource> history;
	u32 outputNode; std::string outputPin;
};
```

Rules:
- An out-pin can fan out to any number of in-pins. An in-pin takes at most one edge.
- Edges carry no data of their own: the resource *is* the source out-pin.

### 2.3 History resources (the one non-obvious part)

A resource read last frame and written this frame (an accumulator, TAA history) is a cycle
if you draw it as an edge. It's modelled instead as two editor nodes backed by one
`HistoryResource`:
- `History Read <name>`: one out-pin, which is last frame's contents.
- `History Write <name>`: one in-pin, which becomes next frame's contents.

Neither node is a real pass. The compiler treats a read node as a graph source and a write
node as a sink, so the sort never sees a loop. Underneath, both map to one double-buffered
transient (`TransientImageCreateInfo::isDoubleBuffered`) with the slot swapped per frame.
Writing the history in place with read-write access is allowed, via `readWriteOf` on a pass
whose in- and out-pins are both wired to the same history.

---

## 3. Validation

`validate(doc) -> std::vector<GraphError>`, where each error has `{ nodeId, pinName,
message }` so the editor can draw it on the offending node. It reports every error rather
than stopping at the first.

| check | error when |
|---|---|
| unknown type | `Node::type` isn't registered |
| unknown pin | an edge names a pin the type lacks |
| direction | edge isn't out→in |
| kind mismatch | image pin wired to buffer pin, etc. |
| fan-in | two edges into one in-pin |
| required | required in-pin is unconnected (after bypass, §4) |
| cycle | report the nodes of one cycle, not just "has a cycle" |
| output | `outputNode/outputPin` missing or not an image out-pin |
| history | a history write node that has more than one incoming edge |
| params | value is outside `[min,max]` or the enum index is out of range; clamp on load, warn |

---

## 4. Compile

`compile(doc) -> CompiledGraph` runs only on a document that validated clean.

1. **Bypass.** For each bypassed node, rewire every consumer of its out-pin to the source
   of the in-pin named in `bypass`, then drop the node. A bypassed node whose type
   declares no mapping for a consumed out-pin is a compile error.
2. **Topological sort.** Use Kahn's algorithm and break ties by ascending `Node::id`. The
   order must be deterministic across runs and platforms, because §8 compares output
   exactly.
3. **Resources.** Allocate one resource per out-pin that has at least one consumer, or is
   the graph output. An out-pin with no consumers still gets a resource: the pass writes
   it regardless, and culling is out of scope (§10).
4. **Lifetimes.** For each resource, record `firstWrite` and `lastRead` as indices into
   the sorted pass list. Nothing uses them yet, but computing them now is what makes
   aliasing (§10) a local change later.
5. **Access modes.** For each pass, list `(resource, read | write | readWrite)`, merging
   pairs linked by `readWriteOf` into readWrite.

```cpp
struct CompiledResource {
	PinKind kind;
	vkof::ImageFormat format; f32 scaleWidth, scaleHeight; u32 mipLevels;
	u32 historyId;  // ~0u if not history
	u32 firstWrite, lastRead;
};

struct CompiledPass {
	u32 nodeId;
	PassType type;
	std::vector<std::pair<u32, vkof::RenderNodeAccess>> resources;
	// pin name -> resource index, for PassContext
	std::vector<std::pair<std::string, u32>> pinResources;
	std::vector<std::pair<std::string, ParamValue>> params;
};

struct CompiledGraph {
	std::vector<CompiledPass> passes;
	std::vector<CompiledResource> resources;
	u32 outputResource;
};
```

`CompiledGraph` is plain data with no Vulkan handles. It's what the core tests check.

---

## 5. Serialization

The library reads and writes JSON with rapidjson.

- **Top-level `"version": 1`.** Refuse newer versions; migrate older ones explicitly.
- **Unknown pass types.** Load them, keep them, write them back unchanged, and flag them
  through validation. A graph authored with a type from a newer build must survive being
  opened and saved in an older one.
- **Unknown params.** Keep them. A missing param takes its default.
- **Stable output.** Write nodes sorted by id and edges sorted by
  `(dstNode, dstPin)`, so diffs of saved graphs stay readable.

---

## 6. vkof emitter

`GraphInstance` owns the vkof side of one compiled graph.

```cpp
struct GraphInstance;
GraphInstance * instance_create(CompiledGraph const & graph);
void instance_destroy(GraphInstance *);
// creates this frame's render nodes; caller passes them to render_graph_execute
srat::slice<vkof::RenderNode const> instance_record(GraphInstance *);
vkof::TransientImage instance_output(GraphInstance const *);
vkof::TransientImage instance_resource_image(GraphInstance const *, u32 resourceIndex);
```

- **Transients.** `instance_create` makes one transient per compiled resource. Transients
  are recreated only when the graph changes, never per frame.
- **Recording.** `instance_record` creates one `vkof::RenderNode` per pass in sorted order.
  It declares each resource with its access mode and wraps `record` in a
  `render_node_callback`.
- **`PassContext`** gives a pass the transient image, storage handle and buffer address for
  each pin by name, plus its param values. A pass never sees resource indices.
- **Profiling.** Pass index i equals node index i in `render_graph_execute`, so
  `vkof::node_gpu_ms(i)` maps back to a `Node::id` through `CompiledGraph::passes[i]`.
- **Dependency on vkof.** `render_node_add_persistent_image` currently discards
  `info.access` (`vkof.cpp:2380`). External-image pins pass access through, but it's
  dropped until vkof honours it. That fix belongs to vkof, not this library; it's listed
  in §9.

---

## 7. Editor (`rgraph-imgui.cpp`)

It's built on imnodes and exposes one function:

```cpp
struct EditorState;  // imnodes context, selection, pending edits
struct EditorResult {
	bool documentChanged;     // caller should re-validate + recompile
	u32 previewResource;      // ~0u if none
};
EditorResult editor_draw(EditorState &, GraphDocument & doc,
	std::vector<GraphError> const & errors,
	srat::slice<double const> passGpuMs);
```

- **Editing:**
  - Add a node from a right-click menu of registered types.
  - Delete nodes and edges; deleting a node removes its edges.
  - Drag from pin to pin to connect. A connection that would break a validation rule is
    refused while dragging, and the reason shows as a tooltip.
  - Edit params inline: a slider for numbers, a checkbox for bool, a combo for enum.
  - Toggle bypass per node.
- **Errors** from `validate` draw as a red node title with the message on hover.
- **Timings:** each node shows `passGpuMs` in its title bar.
- **Preview:** clicking an out-pin sets `previewResource`. The editor doesn't draw images
  itself; whoever hosts it decides what to show.
- **Undo** keeps a snapshot of `GraphDocument` per edit. Documents are small, so a copy
  per edit is fine; don't build a command system.
- **Node positions** are stored in `Node::editorPos`, so the layout survives save and load.

The editor never calls compile. It changes the document and reports that it did, so it
stays testable without a GPU (§8).

---

## 8. Tests (contract, mine to write)

These run CPU-only unless marked, to stay within the default GPU budget.

**Validation:** one test per row of the §3 table, each on a minimal document that
triggers exactly that error, plus one clean document that returns no errors.

**Compile:**
- **Sort:** a diamond (A→B, A→C, B→D, C→D) gives the order A, B, C, D with the id tiebreak.
  Reversing insertion order doesn't change the result.
- **Bypass:** A→B→C with B bypassed compiles to A→C with the consumer rewired. Bypassing a
  node with no mapping is an error.
- **History:** read → pass → write compiles with no cycle. The read resource and write
  resource share a `historyId`.
- **Lifetimes:** in a chain A→B→C, the resource written by A has `firstWrite=0`,
  `lastRead=1`.
- **readWriteOf** merges into one readWrite access.

**Serialization:**
- **Round trip:** load → save → load gives an equal document, byte-identical on the
  second save.
- **Unknown type:** survives the round trip unchanged and is flagged by validate.
- **Version:** a newer version is refused; a missing param takes its default.

**vkof emitter (GPU, small):**
- **Chain:** register three pass types (fill constant → add constant → scale) as tiny
  compute shaders. Wire them up, run one frame and read back the output. It must equal the
  CPU-computed value exactly.
- **Bypass:** the same graph with the middle pass bypassed, checked against its own
  expected value.
- **History:** accumulate a constant over N frames through a history resource. The
  readback must be N×constant.

**Editor:**
- Drive `editor_draw` headless: an ImGui context with no renderer, and synthetic input
  events for a connect and a delete. Check the document diff and `documentChanged`. If
  imnodes can't be driven headless, drop this test and say so; don't fake it.

---

## 9. Decisions for you

1. **Library name.** `rgraph` is a placeholder.
2. **Node-editor library.** imnodes (recommended: small, and matches the plain-ImGui
   style) or imgui-node-editor (richer UX, heavier, with its own canvas and style system).
3. **vkof persistent-image access fix (§6).** This needs a scoped vkof edit exception, or
   you make the change yourself.
4. **Param storage.** A tagged `ParamValue` union (recommended) or everything as `f32`.

---

## 10. Later (deliberately not in this plan)

- **Culling:** walk back from `outputResource` and drop passes whose outputs never reach
  it. This needs a per-type `hasSideEffects` flag first, so readback and counter passes
  survive.
- **Memory aliasing:** use the §4 lifetimes to place non-overlapping transients in shared
  memory. This needs vkof support for creating a transient over supplied memory.
- **Subgraphs:** group nodes into a reusable composite type, for example a bloom chain.
