#ifndef UTIL_SAMPLING_BINS_GLSL
#define UTIL_SAMPLING_BINS_GLSL

// sphere bin grid + direction<->bin mapping for the sampler-vs-pdf
// chi-square tests. ported 1:1 from cull app/shaders/bsdf-verify-shared.h
// and app/shaders/util-bsdf-verify.glsl (bin helpers only; the probe-capture
// plumbing is cull-specific and not needed here). requires utilCalculateXy
// from util-material-openpbr-microfacet.glsl to be included first.

// full-sphere bin grid in the shading-normal-aligned frame;
// theta in [0, pi] (0 = along the normal), phi in [0, 2pi)
#define BSDF_VERIFY_THETA_BINS 40u
#define BSDF_VERIFY_PHI_BINS 80u
#define BSDF_VERIFY_BIN_COUNT (BSDF_VERIFY_THETA_BINS * BSDF_VERIFY_PHI_BINS)

// the two helpers below are the single source of truth for the bin frame and
// phi convention; the histogram and expected shaders must both use them so
// the mapping can never diverge

// maps a world-space direction to a bin index in the nor-aligned frame
uint bsdfVerifyDirectionToBin(const f32v3 nor, const f32v3 dir) {
	f32v3 tanX, tanY;
	utilCalculateXy(nor, tanX, tanY);
	const f32 theta = acos(clamp(dot(dir, nor), -1.0f, 1.0f));
	const f32 phi = atan(dot(dir, tanY), dot(dir, tanX)) + PI;
	// clamp: acos can return exactly pi and atan exactly +pi, either of
	// which would otherwise index one past the end
	const uint thetaBin = min(
		uint(theta / PI * f32(BSDF_VERIFY_THETA_BINS)),
		BSDF_VERIFY_THETA_BINS - 1u
	);
	const uint phiBin = min(
		uint(phi / TAU * f32(BSDF_VERIFY_PHI_BINS)),
		BSDF_VERIFY_PHI_BINS - 1u
	);
	return thetaBin * BSDF_VERIFY_PHI_BINS + phiBin;
}

// inverse mapping for the expected-side integration: (theta, phi) with the
// same nor-aligned frame and phi offset as bsdfVerifyDirectionToBin
f32v3 bsdfVerifyDirectionFromAngles(
	const f32v3 nor,
	const f32 theta,
	const f32 phi
) {
	f32v3 tanX, tanY;
	utilCalculateXy(nor, tanX, tanY);
	const f32 phiFrame = phi - PI;
	const f32 sinTheta = sin(theta);
	return (
		tanX * (sinTheta * cos(phiFrame))
		+ tanY * (sinTheta * sin(phiFrame))
		+ nor * cos(theta)
	);
}

#endif // UTIL_SAMPLING_BINS_GLSL
