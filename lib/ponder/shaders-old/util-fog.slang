module "util-fog";
__exported import "util-types";
__exported import "util-random";



// -----------------------------------------------------------------------------
// -- homogeneous fog
// -----------------------------------------------------------------------------

public static const f32 skUtilFogPhaseIsotropic = (1.0f / (4.0f * skPi));
// sigma is extinction per world unit (GpuGlobalExtended.fogThickness);
// caller guards sigma > 0. free-flight distance through constant sigma is
// exactly exponential and transmittance is closed-form beer-lambert -- no
// majorant/rejection loop needed, unlike the heterogeneous vdb case
public func utilFogSampleDistance(
	sigma: f32,
	inout state: u64,
) -> f32 {
	return -log(1.0f - fnSampleUniform(state)) / sigma;
}

public func utilFogTransmittance(
	sigma: f32,
	dist: f32,
) -> f32 {
	return exp(-sigma * dist);
}

