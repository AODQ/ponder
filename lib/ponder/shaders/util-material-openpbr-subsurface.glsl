#ifndef UTIL_MATERIAL_OPENPBR_SUBSURFACE_GLSL
#define UTIL_MATERIAL_OPENPBR_SUBSURFACE_GLSL

vec3 openPbrSubsurfaceEvaluateF(
	const MaterialTableHandles tables,
	const OpenPbrMaterial mat,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo
) {
	/*
		E_{subsurface} = E_{spec} + E_{multiscatter}\\
		E_{multiscatter} = (1 - E_{spec}) * C

		where,
			C = subsurface color

		E_{spec} is the same top-of-slab dielectric interface as the
		glossy-diffuse lobe (openPbrGlossyDiffuseOrenNayerEvaluateF), which
		likewise contributes no specular term of its own: the caller
		(openPbrEvaluateF) already adds one shared fSpecular term outside
		the mix(fDiffuse, fSubsurface, subsurfaceWeight), and weights the
		whole mixed substrate by (1 - fSpecularDirectionalAlbedo). Giving
		this lobe its own copy of E_{spec}/(1 - E_{spec}) here would double
		the specular energy and double-attenuate the multiscatter term,
		breaking the furnace-test energy-conservation invariant
	*/
	return mat.subsurfaceColor * IPI;
}

#endif // UTIL_MATERIAL_OPENPBR_SUBSURFACE_GLSL
