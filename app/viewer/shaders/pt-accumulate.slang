import "util-tonemap";
import "util-global-pc";
import "util-pt-accumulate-pc";

struct PC {
	GpuGlobalPc global;
	GpuPtAccumulatePC ptAccumulate;
};

[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	const i32v2 coord = i32v2(gl_GlobalInvocationID.xy);
	// the dispatch is rounded up to the 16x16 threadgroup; out-of-bounds
	// invocations must not reach the atomicAdd below or the sample counter
	// overcounts (the imageStores alone would just have been discarded)
	if (
		coord.x >= i32(unf.renderWidth) || coord.y >= i32(unf.renderHeight)
	) {
		return;
	}

	// FRAME_MODE_CLEAR: full-screen dispatch (forced from the cpu side),
	// hard-resets the accumulation buffer to a known-zero state. doesn't
	// read inputHandle at all -- resolve.comp doesn't even run this frame,
	// see GpuGlobalExtended.frameMode's comment
	if (unf.frameMode == 1) {
		imageStore(
			vkofImages[pc.ptAccumulate.accumHandle], coord, f32v4(0.0f)
		);
		imageStore(
			vkofImages[pc.ptAccumulate.outputHandle], coord,
			f32v4(0.0f, 0.0f, 0.0f, 1.0f)
		);
		return;
	}

	// FRAME_MODE_VIEW: full-screen dispatch, resolve.comp's 1spp env-lit
	// preview -- tonemapped straight through for display, but never
	// touches the persistent accumulation buffer, so the real accumulated
	// result survives underneath for when pathtrace mode resumes
	if (unf.frameMode == 0) {
		const f32v4 current = (
			imageLoad(vkofImages[pc.ptAccumulate.inputHandle], coord)
		);
		const f32v4 outColor = (
			f32v4(fnAces(current.rgb * pc.global.exposure), 1.0f)
		);
		imageStore(vkofImages[pc.ptAccumulate.outputHandle], coord, outColor);
		return;
	}

	// resolve only dispatched the frame's tile window (cpu-side frame
	// budget), sliding over the active region -- the whole screen, or a
	// focus rect's sub-grid. pixels outside the region, or outside the
	// window within it, hold a stale input image and must not fold it into
	// their average. the offset subtraction is deliberately unsigned:
	// coords left of / above the region wrap huge and fail the bound
	// checks (localTileX bounds x; tile < tileTotal then bounds y, since
	// tileTotal = tileGridW * gridH). region coverage (tileCount >=
	// tileTotal) skips the wrap test but never the bound checks
	const u32 localTileX = (
		u32(coord.x) / kPtTileSize - pc.ptAccumulate.regionTileX
	);
	const u32 localTileY = (
		u32(coord.y) / kPtTileSize - pc.ptAccumulate.regionTileY
	);
	const u32 tile = localTileY * pc.ptAccumulate.tileGridW + localTileX;
	bool covered = (
		localTileX < pc.ptAccumulate.tileGridW
		&& tile < pc.ptAccumulate.tileTotal
	);
	if (covered && pc.ptAccumulate.tileCount < pc.ptAccumulate.tileTotal) {
		covered = (
			(tile + pc.ptAccumulate.tileTotal - pc.ptAccumulate.tileFirst)
				% pc.ptAccumulate.tileTotal
		) < pc.ptAccumulate.tileCount;
	}

	f32v4 accum;
	if (pc.ptAccumulate.freeze != 0u || !covered) {
		// max spp reached (or outside this frame's tile window): re-output
		// the existing average (exposure/tonemap below still apply live)
		// without consuming a new sample
		accum = imageLoad(vkofImages[pc.ptAccumulate.accumHandle], coord);
	} else {
		const f32v4 current = (
			imageLoad(vkofImages[pc.ptAccumulate.inputHandle], coord)
		);
		// current.a is the sample's validity flag (utilIrradiancePropagation-
		// Entry's float(hit), forwarded by resolve.comp): 1 when the path
		// received direct/indirect light, 0 when it never found any. only
		// valid samples advance a pixel's count or fold into its average, so
		// a pixel's spp measures light-carrying samples, not frames
		f32 prevCount;
		if (pc.ptAccumulate.reset != 0u) {
			prevCount = 0.0f;
			accum = f32v4(current.rgb * current.a, 1.0f);
		} else {
			const f32v4 prev = (
				imageLoad(vkofImages[pc.ptAccumulate.accumHandle], coord)
			);
			prevCount = prev.a;
			if (
				pc.ptAccumulate.maxSpp > 0u
				&& prev.a >= f32(pc.ptAccumulate.maxSpp)
			) {
				// under a partial window pixels reach the max-spp target on
				// different frames, and the cpu-side freeze only engages
				// once every pixel has -- hold finished pixels individually
				// (redundant under full coverage, where the freeze fires
				// the same frame the last sample lands)
				accum = prev;
			} else {
				// an invalid sample (current.a == 0) bumps the count to
				// account for zero-radiance paths. Somewhat unintuitive, but the
				// pixel has to track even when energy is completely lost
				const f32 newCount = prev.a + 1.0f;
				if (current.a == 0.0f) {
					accum = f32v4(prev.rgb, newCount);
				} else {
					accum = (
						newCount > 0.0f
							? f32v4(
								(prev.rgb * prev.a + current.rgb * current.a)
									/ newCount,
								newCount
							)
							: f32v4(0.0f)
					);
				}
			}
		}
		imageStore(vkofImages[pc.ptAccumulate.accumHandle], coord, accum);
		// completion tally for the imgui progress bar: exactly one bump per
		// pixel, on the frame its sample count crosses the target
		if (
			pc.ptAccumulate.maxSpp > 0u
			&& prevCount < f32(pc.ptAccumulate.maxSpp)
			&& accum.a >= f32(pc.ptAccumulate.maxSpp)
		) {
			atomicAdd(&
				((u32 *)(pc.ptAccumulate.sppCounterVa))[0], 1u
			);
		}
	}

	// apply HDR to result
	const f32v4 outColor = f32v4(fnAces(accum.rgb * pc.global.exposure), 1.0f);

	imageStore(
		vkofImages[pc.ptAccumulate.outputHandle],
		coord,
		outColor
	);
}
