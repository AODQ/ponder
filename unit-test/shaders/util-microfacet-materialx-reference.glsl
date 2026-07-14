#ifndef UTIL_MICROFACET_MATERIALX_REFERENCE_GLSL
#define UTIL_MICROFACET_MATERIALX_REFERENCE_GLSL

#ifndef f32
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#endif

// independent cross-check only, not used by the real ponder implementation.
// ported from AcademySoftwareFoundation/MaterialX
// libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl

#define MX_M_PI 3.1415926535897932

f32 mx_square(f32 x)
{
    return x*x;
}

// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// Appendix B.2 Equation 13
f32 mx_ggx_NDF(f32v3 H, f32v2 alpha)
{
    f32v2 He = H.xy / alpha;
    f32 denom = dot(He, He) + mx_square(H.z);
    return 1.0 / (MX_M_PI * alpha.x * alpha.y * mx_square(denom));
}

// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
// Equation 34
f32 mx_ggx_smith_G1(f32 cosTheta, f32 alpha)
{
    f32 cosTheta2 = mx_square(cosTheta);
    f32 tanTheta2 = (1.0 - cosTheta2) / cosTheta2;
    return 2.0 / (1.0 + sqrt(1.0 + mx_square(alpha) * tanTheta2));
}

// Height-correlated Smith masking-shadowing
// http://jcgt.org/published/0003/02/03/paper.pdf
// Equations 72 and 99
f32 mx_ggx_smith_G2(f32 NdotL, f32 NdotV, f32 alpha)
{
    f32 alpha2 = mx_square(alpha);
    f32 lambdaL = sqrt(alpha2 + (1.0 - alpha2) * mx_square(NdotL));
    f32 lambdaV = sqrt(alpha2 + (1.0 - alpha2) * mx_square(NdotV));
    return 2.0 * NdotL * NdotV / (lambdaL * NdotV + lambdaV * NdotL);
}

// Rational quadratic fit to Monte Carlo data for GGX directional albedo.
// scalar F0/F90 specialization (materialx's own is vec3) to compare
// directly against cull's scalar utilMicrofacetGgxDirectionalAlbedo.
f32 mx_ggx_dir_albedo_analytic(f32 NdotV, f32 alpha, f32 F0, f32 F90)
{
    f32 x = NdotV;
    f32 y = alpha;
    f32 x2 = mx_square(x);
    f32 y2 = mx_square(y);
    f32v4 r = f32v4(0.1003, 0.9345, 1.0, 1.0) +
             f32v4(-0.6303, -2.323, -1.765, 0.2281) * x +
             f32v4(9.748, 2.229, 8.263, 15.94) * y +
             f32v4(-2.038, -3.748, 11.53, -55.83) * x * y +
             f32v4(29.34, 1.424, 28.96, 13.08) * x2 +
             f32v4(-8.245, -0.7684, -7.507, 41.26) * y2 +
             f32v4(-26.44, 1.436, -36.11, 54.9) * x2 * y +
             f32v4(19.99, 0.2913, 15.86, 300.2) * x * y2 +
             f32v4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2;
    f32v2 AB = clamp(r.xy / r.zw, 0.0, 1.0);
    return F0 * AB.x + F90 * AB.y;
}

#endif
