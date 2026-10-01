// debug-only NaN/Inf probe: fires debugPrintfEXT with a compile-time literal
// tag at the exact call site that produced a bad value, so a NaN found in
// the final image can be traced back to a specific lobe/term instead of
// bisecting by hand. gated two ways so it costs nothing when unused:
//   * PONDER_NAN_PROBES must be #defined by the consuming shader before its
//     first #include of this file (or of anything that includes it), else
//     every NAN_CHECK* call compiles to nothing -- no debugPrintfEXT token
//     even reaches the compiler, so shaders that never opt in don't need
//     the GL_EXT_debug_printf extension declared at all
//   * even when enabled, an atomic counter caps firings to
//     NAN_PROBE_LIMIT_PER_FRAME: a NaN storm hits every pixel at once, and
//     the validation layer's printf ring buffer is small enough that an
//     unthrottled flood just gets silently dropped past its first ~dozen
//     messages anyway
//
// to remove entirely: delete this file, its #include in
// util-material-openpbr.glsl, and the NAN_CHECK* call sites; to just switch
// it off, drop the PONDER_NAN_PROBES #define from the one shader that set it
// (see resolve.comp)

#ifndef UTIL_NAN_PROBE_GLSL
#define UTIL_NAN_PROBE_GLSL

#ifdef PONDER_NAN_PROBES

layout(buffer_reference, scalar) buffer GpuNanProbeCounterBuffer {
	uint count;
};

#define NAN_PROBE_LIMIT_PER_FRAME 4u

// set once per invocation via nanProbeInit(); ordinary (non-buffer) globals
// in GLSL are private per-invocation, not shared across threads, so this is
// safe despite looking like shared state
u64 gNanProbeCounterVa = u64(0);
ivec2 gNanProbeCoord = ivec2(0);

// -----------------------------------------------------------------------------
// -- nanProbeInit
// -----------------------------------------------------------------------------

void nanProbeInit(const u64 counterVa, const ivec2 coord) {
	gNanProbeCounterVa = counterVa;
	gNanProbeCoord = coord;
}

// true at most NAN_PROBE_LIMIT_PER_FRAME times per frame across every
// invocation combined; callers only reach this after already confirming a
// NaN/Inf, so the atomic only pays for itself on the bad path. false (no
// slot, no printf) if nanProbeInit was never called -- the default state
// for any shader that includes this file without wiring it up
bool nanProbeReserveSlot() {
	if (gNanProbeCounterVa == u64(0)) { return false; }
	GpuNanProbeCounterBuffer counter = (
		GpuNanProbeCounterBuffer(gNanProbeCounterVa)
	);
	return atomicAdd(counter.count, 1u) < NAN_PROBE_LIMIT_PER_FRAME;
}

// fmt is a single literal owned entirely by the call site (e.g.
// "NaN openPbrEvaluateF.fDiffuse px(%d,%d)=%v3f"); %d,%d take the pixel
// coord, the third specifier takes v directly via debugPrintfEXT's vector
// format extension (%f/%v2f/%v3f/%v4f)
#define NAN_CHECK1(v, fmt) \
	if (isnan(v) || isinf(v)) { \
		if (nanProbeReserveSlot()) { \
			debugPrintfEXT(fmt, gNanProbeCoord.x, gNanProbeCoord.y, v); \
		} \
	}

#define NAN_CHECK2(v, fmt) \
	if (any(isnan(v)) || any(isinf(v))) { \
		if (nanProbeReserveSlot()) { \
			debugPrintfEXT(fmt, gNanProbeCoord.x, gNanProbeCoord.y, v); \
		} \
	}

#define NAN_CHECK3(v, fmt) \
	if (any(isnan(v)) || any(isinf(v))) { \
		if (nanProbeReserveSlot()) { \
			debugPrintfEXT(fmt, gNanProbeCoord.x, gNanProbeCoord.y, v); \
		} \
	}

#define NAN_CHECK4(v, fmt) \
	if (any(isnan(v)) || any(isinf(v))) { \
		if (nanProbeReserveSlot()) { \
			debugPrintfEXT(fmt, gNanProbeCoord.x, gNanProbeCoord.y, v); \
		} \
	}

#else

#define NAN_CHECK1(v, fmt)
#define NAN_CHECK2(v, fmt)
#define NAN_CHECK3(v, fmt)
#define NAN_CHECK4(v, fmt)
#define nanProbeInit(counterVa, coord)

#endif // PONDER_NAN_PROBES

#endif // UTIL_NAN_PROBE_GLSL
