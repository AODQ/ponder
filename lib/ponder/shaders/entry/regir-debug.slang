import light;
import types;
import util;

// debug only: resolves one regir cell's reservoirs to world-space light
// points for the viewer's regir inspector (lib/ponder/src/regir-debug.cpp)

// mirrors WavefrontRegirDebugPc in regir-debug.cpp
struct StagePushConstants {
	u64 outputVa;
	u32 slot;
	u32 pad0;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

// mirrors RegirDebugSample in regir-debug.hpp
struct RegirDebugSample {
	f32v3 position;
	f32 weight;
	f32v3 normal;
	f32 targetValue;
	u32 lightIndex;
	// 1 when the reservoir holds a usable sample
	u32 valid;
	// p̂ shading would report for this point, area measure
	f32 pdfArea;
	u32 pad0;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	vkof::globalBind(pc.global);
	let index: u32 = threadId.x;
	if (index >= vkof::skRegirReservoirCount) { return; }
	let output: RegirDebugSample * = (RegirDebugSample *)(pc.stage.outputVa);
	var sample: RegirDebugSample;
	sample.position = f32v3(0.0f);
	sample.weight = 0.0f;
	sample.normal = f32v3(0.0f);
	sample.targetValue = 0.0f;
	sample.lightIndex = 0u;
	sample.valid = 0u;
	sample.pdfArea = 0.0f;
	sample.pad0 = 0u;
	let slot: u32 = pc.stage.slot;
	if (
		slot < vkof::global.regirCellCount
		&& vkof::global.regirGuards[slot] != 0u
		&& vkof::global.regirGuards[slot] != vkof::skRegirSlotEvicted
	) {
		let reservoir: vkof::RegirReservoir = (
			vkof::global.regirReservoirs[
				slot * vkof::skRegirReservoirCount + index
			]
		);
		sample.weight = reservoir.weight;
		sample.targetValue = reservoir.targetValue;
		sample.lightIndex = reservoir.lightIndex;
		// -- the same reconstruction shading uses; ξ lands exactly on index
		let selection: vkof::RegirSelection = (
			vkof::regirSelectReservoir(
				/*slot=*/ slot,
				/*ξ=*/ (f32(index) + 0.5f) / f32(vkof::skRegirReservoirCount)
			)
		);
		if (selection.valid) {
			sample.position = selection.geometry.position;
			sample.normal = selection.geometry.geometricNormal;
			sample.valid = 1u;
			sample.pdfArea = (
				vkof::regirPdfArea(
					/*slot=*/ slot,
					/*lightIndex=*/ selection.lightIndex,
					/*lightOri=*/ selection.geometry.position,
					/*lightNor=*/ selection.geometry.geometricNormal
				)
			);
		}
	}
	output[index] = sample;
}
