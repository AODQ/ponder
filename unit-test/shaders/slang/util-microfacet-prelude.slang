#pragma once
__exported import "util-types";
__exported import "util-material-openpbr-tables";
__exported import "util-material-openpbr-thinfilm";
__exported import "util-material-openpbr-microfacet";

// the bare microfacet primitives (ggx D, smith G1/G2) live in
// util-material-openpbr-microfacet.glsl, which also carries the kulla-conty
// table lookups and the dielectric specular lobe -- so it needs the bindless
// texture array, the material struct and the thin-film fresnel in scope.
// this prelude supplies exactly that, for the lobe-only test shaders that
// only want the primitives themselves.
//
// must be included before any declaration: it emits an #extension directive.

#extension GL_EXT_nonuniform_qualifier : require


#ifndef PI
#endif
#ifndef TAU
#endif


