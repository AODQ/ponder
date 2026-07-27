#ifndef UTIL_FUZZ_MATERIALX_REFERENCE_GLSL
#define UTIL_FUZZ_MATERIALX_REFERENCE_GLSL

#ifndef f32
#define i32v2 ivec2
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#endif

// independent cross-check only, not used by the real ponder implementation.
// ported from AcademySoftwareFoundation/MaterialX
// libraries/pbrlib/genglsl/lib/mx_microfacet_sheen.glsl (zeltner sheen
// portion only) and libraries/pbrlib/genglsl/lib/mx_microfacet.glsl
// (mx_orthonormal_basis). requires mx_square, mx_cosine_sample_hemisphere
// and mx_cosine_hemisphere_PDF from util-microfacet-materialx-reference.glsl
// to be included first.
//
// the following functions are adapted from https://github.com/tizian/ltc-sheen
// "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines",
// Zeltner et al. -- the same paper ponder's utilZeltnerFuzzLookup fits a
// 32x32 texture table to; materialx instead uses closed-form rational fits
// to the same underlying data. openPbrFuzzSampleWo/openPbrFuzzPdf take the
// fitted (a, b) coefficients as direct parameters rather than doing the
// fit/lookup themselves, so feeding materialx's aInv/bInv straight into the
// ponder functions cross-checks the LTC transform/inversion math
// independently of which fit produced the coefficients.

// Gaussian fit to directional albedo table.
f32 mx_zeltner_sheen_dir_albedo(f32 x, f32 y)
{
    f32 s = y*(0.0206607 + 1.58491*y)/(0.0379424 + y*(1.32227 + y));
    f32 m = y*(-0.193854 + y*(-1.14885 + y*(1.7932 - 0.95943*y*y)))/(0.046391 + y);
    f32 o = y*(0.000654023 + (-0.0207818 + 0.119681*y)*y)/(1.26264 + y*(-1.92021 + y));
    return exp(-0.5*mx_square((x - m)/s))/(s*sqrt(2.0*MX_M_PI)) + o;
}

// Rational fits to LTC matrix coefficients.
f32 mx_zeltner_sheen_ltc_aInv(f32 x, f32 y)
{
    return (2.58126*x + 0.813703*y)*y/(1.0 + 0.310327*x*x + 2.60994*x*y);
}

f32 mx_zeltner_sheen_ltc_bInv(f32 x, f32 y)
{
    return sqrt(1.0 - x)*(y - 1.0)*y*y*y/(0.0000254053 + 1.71228*x - 1.71506*x*y + 1.34174*y*y);
}

// Construct an orthonormal basis from a unit vector.
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
mat3 mx_orthonormal_basis(f32v3 N)
{
    f32 sign = (N.z < 0.0) ? -1.0 : 1.0;
    f32 a = -1.0 / (sign + N.z);
    f32 b = N.x * N.y * a;
    f32v3 X = f32v3(1.0 + sign * N.x * N.x * a, sign * b, -sign * N.x);
    f32v3 Y = f32v3(b, sign + N.y * N.y * a, -N.y);
    return mat3(X, Y, N);
}

// V and N are assumed to be unit vectors.
mat3 mx_orthonormal_basis_ltc(f32v3 V, f32v3 N, f32 NdotV)
{
    // Generate a tangent vector in the plane of V and N.
    // This required to correctly orient the LTC lobe.
    f32v3 X = V - N*NdotV;
    f32 lenSqr = dot(X, X);
    if (lenSqr > 0.0)
    {
        X *= inversesqrt(lenSqr);
        f32v3 Y = cross(N, X);
        return mat3(X, Y, N);
    }

    // If lenSqr == 0, then V == N, so any orthonormal basis will do.
    return mx_orthonormal_basis(N);
}

// Multiplication by directional albedo is handled by the calling function.
f32 mx_zeltner_sheen_brdf(f32v3 L, f32v3 V, f32v3 N, f32 NdotV, f32 roughness)
{
    mat3 toLTC = transpose(mx_orthonormal_basis_ltc(V, N, NdotV));
    f32v3 w = toLTC * L;

    f32 aInv = mx_zeltner_sheen_ltc_aInv(NdotV, roughness);
    f32 bInv = mx_zeltner_sheen_ltc_bInv(NdotV, roughness);

    // Transform w to original configuration (clamped cosine).
    //                 |aInv    0 bInv|
    // wo = M^-1 . w = |   0 aInv    0| . w
    //                 |   0    0    1|
    f32v3 wo = f32v3(aInv*w.x + bInv*w.z, aInv * w.y, w.z);
    f32 lenSqr = dot(wo, wo);

    // D(w) = Do(M^-1.w / ||M^-1.w||) . |M^-1| / ||M^-1.w||^3
    //      = Do(M^-1.w) . |M^-1| / ||M^-1.w||^4
    //      = Do(wo) . |M^-1| / dot(wo, wo)^2
    //      = Do(wo) . aInv^2 / dot(wo, wo)^2
    //      = Do(wo) . (aInv / dot(wo, wo))^2
    return mx_cosine_hemisphere_PDF(wo.z) * mx_square(aInv / lenSqr);
}

f32v3 mx_zeltner_sheen_importance_sample(
    f32v2 Xi, f32v3 V, f32v3 N, f32 roughness, out f32 pdf
)
{
    f32 NdotV = clamp(dot(N, V), 0.0, 1.0);

    f32v3 wo = mx_cosine_sample_hemisphere(Xi);

    f32 aInv = mx_zeltner_sheen_ltc_aInv(NdotV, roughness);
    f32 bInv = mx_zeltner_sheen_ltc_bInv(NdotV, roughness);

    // Transform wo from original configuration (clamped cosine).
    //              |1/aInv      0 -bInv/aInv|
    // w = M . wo = |     0 1/aInv          0| . wo
    //              |     0      0          1|
    f32v3 w = f32v3(wo.x/aInv - wo.z*bInv/aInv, wo.y / aInv, wo.z);

    f32 lenSqr = dot(w, w);
    w *= inversesqrt(lenSqr);

    // D(w) = Do(wo) . ||M.wo||^3 / |M|
    //      = Do(wo / ||M.wo||) . ||M.wo||^4 / |M|
    //      = Do(w) . ||M.wo||^4 / |M| (possible because M doesn't change z component)
    //      = Do(w) . dot(w, w)^2 * aInv^2
    //      = Do(w) . (aInv * dot(w, w))^2
    pdf = mx_cosine_hemisphere_PDF(w.z) * mx_square(aInv * lenSqr);

    mat3 fromLTC = mx_orthonormal_basis_ltc(V, N, NdotV);
    w = fromLTC * w;

    return w;
}

#endif
