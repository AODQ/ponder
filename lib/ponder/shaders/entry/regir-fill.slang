import light;
import path;
import types;
import util;

// read shaders/light/regir.slang for more information about the regir algorithm

// mirrors wavefrontregirfillpc in wavefront.cpp
struct StagePushConstants {
	u32 pad0;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

// one entry per reservoir
groupshared f32 sNormalizationEstimates[vkof::skRegirReservoirCount];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 groupId : SV_GroupID,
	const u32v3 threadId : SV_GroupThreadID,
) {
	vkof::globalBind(pc.global);
	let slot: u32 = groupId.x;
	let reservoirIndex: u32 = threadId.x;
	if (vkof::global.regirEnabled == 0u || slot >= vkof::global.regirCellCount) {
		return;
	}
	let guard: u32 = vkof::global.regirGuards[slot];
	if (guard == 0u || guard == vkof::skRegirSlotEvicted) {
		return;
	}
	// -- initialize cell
	var cell: vkof::RegirCell = vkof::global.regirCells[slot];
	// -- evict cells if they expired
	if (vkof::regirCellExpired(cell)) {
		if (reservoirIndex == 0u) {
			vkof::global.regirGuards[slot] = vkof::skRegirSlotEvicted;
		}
		return;
	}
	// -- last frame's shading point becomes the representative
	cell.representativeOri = cell.pendingOri;
	cell.representativeNor = cell.pendingNor;
	// -- fill the reservoir
	let fill: vkof::RegirFillResult = (
		vkof::regirFillReservoir(
			/*cell=*/ cell,
			/*slot=*/ slot,
			/*reservoirIndex=*/ reservoirIndex,
		)
	);

	// -- check if the reservoir fails geometrical visibility
	var reservoir: vkof::RegirReservoir = fill.reservoir;
	if (
		reservoir.weight > 0.0f && !vkof::regirVisible(cell, fill.position)
	) {
		reservoir.weight = 0.0f;
	}

	// -- store the normalization estimate for this reservoir
	let reservoirSlot: u32 = slot * vkof::skRegirReservoirCount + reservoirIndex;
	vkof::global.regirReservoirs[reservoirSlot] = reservoir;

	// -- store the normalization estimate for this reservoir
	sNormalizationEstimates[reservoirIndex] = fill.normalizationEstimate;
	GroupMemoryBarrierWithGroupSync();
	if (reservoirIndex != 0u) {
		return;
	}
	// -- sum over all reservoirs in the cell
	var total: f32 = 0.0f;
	for (u32 i = 0u; i < vkof::skRegirReservoirCount; ++i) {
		total += sNormalizationEstimates[i];
	}
	// -- store the normalization estimate for this cell
	vkof::global.regirCells[slot].normalizationNoncanonical = (
		total / f32(vkof::skRegirReservoirCount)
	);
	vkof::global.regirCells[slot].lastFillIndex = vkof::global.regirFillIndex;
	vkof::global.regirCells[slot].representativeOri = cell.representativeOri;
	vkof::global.regirCells[slot].representativeNor = cell.representativeNor;
}
