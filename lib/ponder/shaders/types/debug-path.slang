implementing types;

// -----------------------------------------------------------------------------
// -- debug path recorder
// -----------------------------------------------------------------------------
// mirrors lib/ponder/include/ponder/debug-path.hpp

#if defined(PONDER_DEBUG_PATH)
public static const bool skPonderDebugPath = true;
#else
public static const bool skPonderDebugPath = false;
#endif

namespace vkof {

public static const u32 skDebugPathVertexCapacity = 64u;
public static const u32 skDebugPathPointCapacity = 2048u;
public static const u32 skDebugPathDimensionCapacity = 512u;
// point kind of an intermediate chord point
public static const u32 skDebugPathPointChord = 0xFFFFFFFFu;
// DebugPathVertex.neeState
public static const u32 skDebugPathNeeNone = 0u;
public static const u32 skDebugPathNeeBlocked = 1u;
public static const u32 skDebugPathNeeVisible = 2u;
// DebugPathVertex.neeStrategy, how nee picked its light
public static const u32 skDebugPathNeeStrategyNone = 0u;
public static const u32 skDebugPathNeeStrategyPowerCdf = 1u;
public static const u32 skDebugPathNeeStrategyRegir = 2u;

public struct DebugPathVertex {
	public u32 event;
	public u32 bounce;
	public f32 transmittance;
	public f32 frequencyShift;
	public f32v3 position;
	public f32 emitTime;
	public f32v3 direction;
	public f32 pathLength;
	public f32v3 throughputRgb;
	public f32 rrChance;
	public f32v3 emissionRgb;
	// surface vertices: the hit instance and its material index
	public u32 surfaceInstance;
	public f32v4 wavelengths;
	// added to the pixel's irradiance at this vertex
	public f32v3 contributionRgb;
	public u32 surfaceMaterial;
	// per wavelength; the rgb build stores rgb with w = 0
	public f32v4 throughputAmounts;
	// the four mis pdfs of PathVertex
	public f32 pdfBsdfFromHit;
	public u32 isDeltaDirac;
	public f32 pdfEmitFromHit;
	public f32 pdfEmitFromNee;
	// the nee part of contributionRgb
	public f32v3 directRgb;
	public f32 pdfBsdfFromNee;
	// the emitter sample nee connected this vertex to, and skDebugPathNee*
	public f32v3 neePosition;
	public u32 neeState;
	// the usable regir cell nee looked up, or skRegirSlotInvalid
	public u32 regirSlot;
	// skDebugPathNeeStrategy*
	public u32 neeStrategy;
	// the reservoir a regir nee sample came from
	public u32 neeReservoir;
	u32 pad0;
};

public struct DebugPathPoint {
	public f32v3 position;
	// TransportEvent of the vertex at this point, or skDebugPathPointChord
	public u32 kind;
};

public struct DebugPath {
	public u32 vertexCount;
	public u32 pointCount;
	public u32 dimensionCount;
	// bit 0 vertices, bit 1 points, bit 2 dimensions
	public u32 overflow;
	public u32v2 pixel;
	public u32 sampleIndex;
	// size of DebugPath as compiled into the shader
	public u32 byteCount;
	public DebugPathVertex vertices[skDebugPathVertexCapacity];
	public DebugPathPoint points[skDebugPathPointCapacity];
	public u32 dimensions[skDebugPathDimensionCapacity];
};

}

static bool sImplDebugPathActive = false;

namespace vkof {

// starts recording if pixel is the selected debug pixel
public func debugPathBegin(pixel: u32v2) {
	if (!skPonderDebugPath) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	::sImplDebugPathActive = (
		path != nullptr && all(pixel == vkof::global.debugPixel)
	);
	if (!::sImplDebugPathActive) { return; }
	path.vertexCount = 0u;
	path.pointCount = 0u;
	path.dimensionCount = 0u;
	path.overflow = 0u;
	path.pixel = pixel;
	path.sampleIndex = vkof::global.sampleIndex;
	path.byteCount = u32(sizeof(DebugPath));
}

// continues recording in a later wavefront kernel, without resetting
public func debugPathResume(pixel: u32v2) {
	if (!skPonderDebugPath) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	::sImplDebugPathActive = (
		path != nullptr && all(pixel == vkof::global.debugPixel)
	);
}

public func debugPathActive() -> bool {
	return skPonderDebugPath && ::sImplDebugPathActive;
}

public func debugPathPoint(position: f32v3, kind: u32) {
	if (!debugPathActive()) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	if (path.pointCount >= skDebugPathPointCapacity) {
		path.overflow |= 2u;
		return;
	}
	path.points[path.pointCount] = { position, kind };
	path.pointCount += 1u;
}

public func debugPathVertex(vertex: DebugPathVertex) {
	if (!debugPathActive()) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	debugPathPoint(vertex.position, vertex.event);
	if (path.vertexCount >= skDebugPathVertexCapacity) {
		path.overflow |= 1u;
		return;
	}
	path.vertices[path.vertexCount] = vertex;
	path.vertexCount += 1u;
}

// russian roulette runs after the vertex is recorded
public func debugPathVertexRrChance(rrChance: f32) {
	if (!debugPathActive()) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	if (path.vertexCount == 0u) { return; }
	path.vertices[path.vertexCount - 1u].rrChance = rrChance;
}

// record the nee sample of the last vertex
public func debugPathVertexNee(
	directRgb: f32v3,
	pdfEmitFromNee: f32,
	pdfBsdfFromNee: f32,
	position: f32v3,
	state: u32,
	regirSlot: u32,
	strategy: u32,
	reservoir: u32,
) {
	if (!debugPathActive()) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	if (path.vertexCount == 0u) { return; }
	let i: u32 = path.vertexCount - 1u;
	path.vertices[i].directRgb = directRgb;
	path.vertices[i].contributionRgb += directRgb;
	path.vertices[i].pdfEmitFromNee = pdfEmitFromNee;
	path.vertices[i].pdfBsdfFromNee = pdfBsdfFromNee;
	path.vertices[i].neePosition = position;
	path.vertices[i].neeState = state;
	path.vertices[i].regirSlot = regirSlot;
	path.vertices[i].neeStrategy = strategy;
	path.vertices[i].neeReservoir = reservoir;
}

public func debugPathVertexContribution(contributionRgb: f32v3) {
	if (!debugPathActive()) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	if (path.vertexCount == 0u) { return; }
	path.vertices[path.vertexCount - 1u].contributionRgb = contributionRgb;
}

public func debugPathDimension(dimension: u32) {
	if (!debugPathActive()) { return; }
	let path: DebugPath * = vkof::global.debugPath;
	if (path.dimensionCount >= skDebugPathDimensionCapacity) {
		path.overflow |= 4u;
		return;
	}
	path.dimensions[path.dimensionCount] = dimension;
	path.dimensionCount += 1u;
}

}
