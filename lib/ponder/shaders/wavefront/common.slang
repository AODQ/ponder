implementing wavefront;

import types;
import medium;
import util;

namespace vkof {

// a simple transport state for the camera and bounce wavefront
public struct WavefrontTransportState
{
	public f32v3 ori;
	public f32v3 dir;
	public vkof::Wavelengths λ;
	public f32 cameraTime;
	public f32 pathLength;
	public vkof::MediumStack mediumStack;
};

public struct WavefrontPath {
	// into the path buffers
	public u32 index;
	public u32v2 pixel;
	public u32 sampleIndex;
};

static const u32 skWavefrontTileSize = 16u;
static const u32 skWavefrontTileSlotCount = 256u;

// path for given thread
public func wavefrontPath(thread: u32) -> Optional<WavefrontPath> {
	if (thread >= vkof::global.passPathCount) { return none; }
	let slot: u32 = vkof::global.passSlotFirst + thread;
	let tile: u32 = slot / skWavefrontTileSlotCount;
	let local: u32 = slot % skWavefrontTileSlotCount;
	let offset: u32v2 = (
		u32v2(
			tile % vkof::global.passRegionTilesW,
			tile / vkof::global.passRegionTilesW
		) * skWavefrontTileSize
		+ u32v2(local % skWavefrontTileSize, local / skWavefrontTileSize)
	);
	if (any(offset >= vkof::global.passRegionExtent)) { return none; }
	let path: WavefrontPath = {
		/*index=*/ thread,
		/*pixel=*/ vkof::global.passRegionOrigin + offset,
		/*sampleIndex=*/ vkof::global.sampleIndex,
	};
	return path;
}

// path index for pixel position, as the wavefront order can be ran
// asynchronous
public func wavefrontPixelPathOrder(pixel: u32v2) -> Optional<u32> {
	if (any(pixel < vkof::global.passRegionOrigin)) { return none; }
	let offset: u32v2 = pixel - vkof::global.passRegionOrigin;
	if (any(offset >= vkof::global.passRegionExtent)) { return none; }
	let tile: u32v2 = offset / skWavefrontTileSize;
	let local: u32v2 = offset % skWavefrontTileSize;
	let slot: u32 = (
		(tile.y * vkof::global.passRegionTilesW + tile.x)
			* skWavefrontTileSlotCount
		+ local.y * skWavefrontTileSize + local.x
	);
	if (slot < vkof::global.passSlotFirst) { return none; }
	let order: u32 = slot - vkof::global.passSlotFirst;
	if (order >= vkof::global.passPathCount) { return none; }
	return order;
}

public func wavefrontTransportStateLoad(
	pathIndex: u32
) -> WavefrontTransportState {
	let states: WavefrontTransportState * = (
		(WavefrontTransportState *)(vkof::global.wavefrontTransportStates)
	);
	return states[pathIndex];
}

public func wavefrontTransportStateStore(
	pathIndex: u32,
	state: WavefrontTransportState
) {
	let states: WavefrontTransportState * = (
		(WavefrontTransportState *)(vkof::global.wavefrontTransportStates)
	);
	states[pathIndex] = state;
}

} // namespace vkof
