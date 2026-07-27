#ifndef UTIL_MICROFACET_PRELUDE_GLSL
#define UTIL_MICROFACET_PRELUDE_GLSL

// the bare microfacet primitives (ggx D, smith G1/G2) live in
// util-material-openpbr-microfacet.glsl, which also carries the kulla-conty
// table lookups and the dielectric specular lobe -- so it needs the bindless
// texture array, the material struct and the thin-film fresnel in scope.
// this prelude supplies exactly that, for the lobe-only test shaders that
// only want the primitives themselves.
//
// must be included before any declaration: it emits an #extension directive.

#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform sampler2D vkofTextures[];

#ifndef PI
#define PI  3.14159265358979323846
#endif
#ifndef TAU
#define TAU 6.28318530717958647692
#endif

#include "util-material-openpbr-tables.glsl"
#include "util-material-openpbr-shared.h"
#include "util-material-openpbr-thinfilm.glsl"
#include "util-material-openpbr-microfacet.glsl"

#endif // UTIL_MICROFACET_PRELUDE_GLSL
