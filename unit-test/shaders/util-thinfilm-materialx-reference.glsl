#ifndef UTIL_THINFILM_MATERIALX_REFERENCE_GLSL
#define UTIL_THINFILM_MATERIALX_REFERENCE_GLSL

#ifndef f32
#define i32v2 ivec2
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#endif

#include "util-microfacet-materialx-reference.glsl"

// vec3 overload; util-microfacet-materialx-reference.glsl only ported the
// scalar mx_square (materialx's mx_math.glsl has both, but nothing prior to
// this file needed the vector form)
f32v3 mx_square(f32v3 x) {
	return x*x;
}

// independent cross-check only, not used by the real ponder implementation.
// ported from AcademySoftwareFoundation/MaterialX
// libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl (mx_fresnel_airy
// and its direct dependencies), main branch as of 2026-07. the FresnelData
// struct and its FRESNEL_MODEL_SCHLICK branch are not ported -- ponder's
// openPbrThinfilmFresnel always takes an explicit real/complex ior base
// (etaBase/kappaBase), which corresponds exactly to materialx's non-schlick
// (dielectric/conductor) branch; the schlick branch belongs to the metallic
// caller's own f82 blend, one layer up, and is out of scope for this
// function-level crosscheck. mx_atan is spelled as the builtin two-arg
// atan(y, x), same spirit as mx_cos/mx_sin in util-microfacet-materialx-
// reference.glsl. const placement here is west (glsl does not accept the
// east-const `type const name` form the rest of this repo's c++ uses).

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
f32v2 mxFresnelDielectricPolarized(f32 cosTheta, f32 ior) {
	const f32 cosTheta2 = mx_square(clamp(cosTheta, 0.0, 1.0));
	const f32 sinTheta2 = 1.0 - cosTheta2;

	const f32 t0 = max(ior * ior - sinTheta2, 0.0);
	const f32 t1 = t0 + cosTheta2;
	const f32 t2 = 2.0 * sqrt(t0) * cosTheta;
	const f32 Rs = (t1 - t2) / (t1 + t2);

	const f32 t3 = cosTheta2 * t0 + sinTheta2 * sinTheta2;
	const f32 t4 = t2 * sinTheta2;
	const f32 Rp = Rs * (t3 - t4) / (t3 + t4);

	return f32v2(Rp, Rs);
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
void mxFresnelConductorPolarized(
	f32 cosTheta, f32v3 n, f32v3 k, out f32v3 Rp, out f32v3 Rs
) {
	const f32 cosTheta2 = mx_square(clamp(cosTheta, 0.0, 1.0));
	const f32 sinTheta2 = 1.0 - cosTheta2;
	const f32v3 n2 = n * n;
	const f32v3 k2 = k * k;

	const f32v3 t0 = n2 - k2 - f32v3(sinTheta2);
	const f32v3 a2plusb2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
	const f32v3 t1 = a2plusb2 + f32v3(cosTheta2);
	const f32v3 a = sqrt(max(0.5 * (a2plusb2 + t0), f32v3(0.0)));
	const f32v3 t2 = 2.0 * a * cosTheta;
	Rs = (t1 - t2) / (t1 + t2);

	const f32v3 t3 = cosTheta2 * a2plusb2 + f32v3(sinTheta2 * sinTheta2);
	const f32v3 t4 = t2 * sinTheta2;
	Rp = Rs * (t3 - t4) / (t3 + t4);
}

// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
void mxFresnelConductorPhasePolarized(
	f32 cosTheta, f32 eta1, f32v3 eta2, f32v3 kappa2,
	out f32v3 phiP, out f32v3 phiS
) {
	const f32v3 k2 = kappa2 / eta2;
	const f32v3 sinThetaSqr = f32v3(1.0) - cosTheta * cosTheta;
	const f32v3 A = eta2*eta2*(f32v3(1.0)-k2*k2) - eta1*eta1*sinThetaSqr;
	const f32v3 B = sqrt(A*A + mx_square(2.0*eta2*eta2*k2));
	const f32v3 U = sqrt((A+B)/2.0);
	const f32v3 V = max(f32v3(0.0), sqrt((B-A)/2.0));

	phiS = atan(2.0*eta1*V*cosTheta, U*U + V*V - mx_square(eta1*cosTheta));
	phiP = atan(
		2.0*eta1*eta2*eta2*cosTheta * (2.0*k2*U - (f32v3(1.0)-k2*k2) * V),
		mx_square(eta2*eta2*(f32v3(1.0)+k2*k2)*cosTheta) - eta1*eta1*(U*U+V*V)
	);
}

// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
// note: materialx's opd here is in meters (see mxThinfilmFresnel's
// distMeters conversion); ponder's utilEvaluateXyzSensitivity instead takes
// opd already pre-scaled by 1e-6 inside openPbrThinfilmFresnel, since its
// thickness convention is micrometers rather than materialx's nanometers.
f32v3 mxEvalSensitivity(f32 opd, f32v3 shift) {
	const f32 phase = 2.0*MX_M_PI * opd;
	const f32v3 val = f32v3(5.4856e-13, 4.4201e-13, 5.2481e-13);
	const f32v3 pos = f32v3(1.6810e+06, 1.7953e+06, 2.2084e+06);
	const f32v3 var = f32v3(4.3278e+09, 9.3046e+09, 6.6121e+09);
	f32v3 xyz = (
		val * sqrt(2.0*MX_M_PI * var) * cos(pos * phase + shift)
		* exp(- var * phase*phase)
	);
	xyz.x += (
		9.7470e-14 * sqrt(2.0*MX_M_PI * 4.5282e+09)
		* cos(2.2399e+06 * phase + shift[0]) * exp(- 4.5282e+09 * phase*phase)
	);
	return xyz / 1.0685e-7;
}

// A Practical Extension to Microfacet Theory for the Modeling of Varying
// Iridescence (Belcour & Barla 2017). ported from mx_fresnel_airy, with the
// FresnelData struct inlined and restricted to the non-schlick (real/
// complex ior base) path -- tfThicknessNm is materialx's tf_thickness
// (nanometers); baseIor/baseKappa are fd.ior/fd.extinction.
f32v3 mxThinfilmFresnel(
	f32 cosTheta, f32 tfThicknessNm, f32 tfIor, f32v3 baseIor, f32v3 baseKappa
) {
	const mat3 XYZ_TO_RGB = mat3(
		2.3706743, -0.5138850, 0.0052982,
		-0.9000405, 1.4253036, -0.0146949,
		-0.4706338, 0.0885814, 1.0093968
	);

	const f32 eta1 = 1.0;
	const f32 eta2 = max(tfIor, eta1);
	const f32v3 eta3 = baseIor;
	const f32v3 kappa3 = baseKappa;
	const f32 cosThetaT = (
		sqrt(1.0 - (1.0 - mx_square(cosTheta)) * mx_square(eta1 / eta2))
	);

	// first interface
	f32v2 R12 = mxFresnelDielectricPolarized(cosTheta, eta2 / eta1);
	if (cosThetaT <= 0.0) {
		R12 = f32v2(1.0);
	}
	const f32v2 T121 = f32v2(1.0) - R12;

	// second interface
	f32v3 R23p, R23s;
	mxFresnelConductorPolarized(cosThetaT, eta3 / eta2, kappa3 / eta2, R23p, R23s);

	// phase shift
	const f32 cosB = cos(atan(eta2 / eta1));
	const f32v2 phi21 = f32v2(cosTheta < cosB ? 0.0 : MX_M_PI, MX_M_PI);
	f32v3 phi23p, phi23s;
	mxFresnelConductorPhasePolarized(cosThetaT, eta2, eta3, kappa3, phi23p, phi23s);

	const f32v3 r123p = max(sqrt(R12.x*R23p), f32v3(0.0));
	const f32v3 r123s = max(sqrt(R12.y*R23s), f32v3(0.0));

	f32v3 I = f32v3(0.0);
	f32v3 Cm, Sm;

	// optical path difference; tf_thickness is in nanometers
	const f32 distMeters = tfThicknessNm * 1.0e-9;
	const f32 opd = 2.0 * eta2 * cosThetaT * distMeters;

	// parallel polarization: m=0 (DC term), then m>0
	f32v3 Rs = (mx_square(T121.x) * R23p) / (f32v3(1.0) - R12.x*R23p);
	I += R12.x + Rs;
	Cm = Rs - T121.x;
	for (int m = 1; m <= 3; ++m) {
		Cm *= r123p;
		Sm = 2.0 * mxEvalSensitivity(f32(m) * opd, f32(m)*(phi23p+f32v3(phi21.x)));
		I += Cm*Sm;
	}

	// perpendicular polarization: m=0 (DC term), then m>0
	f32v3 Rp = (mx_square(T121.y) * R23s) / (f32v3(1.0) - R12.y*R23s);
	I += R12.y + Rp;
	Cm = Rp - T121.y;
	for (int m = 1; m <= 3; ++m) {
		Cm *= r123s;
		Sm = 2.0 * mxEvalSensitivity(f32(m) * opd, f32(m)*(phi23s+f32v3(phi21.y)));
		I += Cm*Sm;
	}

	I *= 0.5;
	return clamp(XYZ_TO_RGB * I, f32v3(0.0), f32v3(1.0));
}

#endif
