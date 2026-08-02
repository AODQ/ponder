implementing types;

// -----------------------------------------------------------------------------
// -- units and conventions
// -----------------------------------------------------------------------------
//
// space:       meters. right-handed. +y up
// angles:      radians
// wavelength:  nanometers, sampled over [skLambdaMin, skLambdaMax]
// radiance:    relative spectral radiance; luminance Y = 1 is display white
//              at exposure 1. absolute sources are converted on the cpu
// temperature: kelvin
//
// relativity:  geometrized units, G = c = 1, metric signature (-, +, +, +)
//   time:      meters of light travel (1 m = 3.336 ns)
//   velocity:  fraction of c
//   mass:      gravitational radius G M / c^2 in meters (sun = 1476.6 m)
//   coords:    kerr-schild cartesian; a black hole spins about its local +y
//
// -----------------------------------------------------------------------------
// -- math symbols
// -----------------------------------------------------------------------------
// each symbol has one meaning everywhere, like a rendering book's notation
// table. only symbols shared across algorithms are listed; anything used by
// a single algorithm is spelled out in plain english instead
//
// -- geometry
//    x    position; x0, x1, …; positions along path. [usually 'ori']
//    t    distance along a ray [usually 'dist']
//    n    shading normal [usually 'nor']
//    ω    unit direction
//    ωi   incoming direction, pointing toward the surface/medium
//    ωo   outgoing direction, pointing away from the surface/medium
//         paths run camera to light, the reverse of pbrt's ωi and ωo
//    θ    polar angle in a local frame, about n or about ωi in media
//    φ    azimuth in a local frame
//
// -- radiometry
//    λ    wavelength in nanometers, or a path's wavelength set
//    L    radiance; Le is emitted radiance, Li incident radiance
//    β    path throughput
//    Tr   transmittance
//
// -- scattering
//    f    scattering function: the bsdf at surfaces, the phase function
//         in media
//    ρ    albedo; σs / σt in media
//    g    anisotropy, the mean cosine of scattering
//    α    microfacet roughness
//    η    index of refraction
//
// -- participating media, per meter
//    σa   absorption coefficient
//    σs   scattering coefficient
//    σt   extinction coefficient, σa + σs
//    τ    optical depth, the integral of σt along a ray
//
// -- sampling
//    ξ    uniform random numbers in [0, 1)
//    pdf  probability density, always spelled out
//
// -- constants
//    π
//
// -----------------------------------------------------------------------------

public static const f32 π = 3.1415926535897932384626433832795f;

namespace vkof {

// visible range covered by the cie tables and the rgb -> spectrum fit
public static const f32 skLambdaMin = 360.0f;
public static const f32 skLambdaMax = 830.0f;

} // namespace vkof

// -----------------------------------------------------------------------------
// TODO move to new module

public func max(v: f32v3) -> f32 {
	return max(max(v.x, v.y), v.z);
}
