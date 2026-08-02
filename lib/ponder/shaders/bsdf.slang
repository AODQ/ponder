module bsdf;

__include "bsdf/lambertian.slang";
__include "bsdf/shading-frame.slang";
__include "bsdf/bsdf-math.slang";
__include "bsdf/bsdf-interface.slang";
__include "bsdf/bsdf-openpbr.slang";
__include "bsdf/bsdf-fresnel.slang";
__include "bsdf/bsdf-microfacet.slang";
__include "bsdf/bsdf-energy-compensation.slang";
__include "bsdf/bsdf-openpbr-shared.slang";
__include "bsdf/bsdf-thinfilm.slang";
__include "bsdf/bsdf-openpbr-specular.slang";
__include "bsdf/bsdf-openpbr-metallic.slang";
__include "bsdf/bsdf-openpbr-diffuse.slang";
__include "bsdf/bsdf-openpbr-coat.slang";
__include "bsdf/bsdf-openpbr-fuzz.slang";
__include "bsdf/bsdf-openpbr-transmission.slang";
__include "bsdf/bsdf-openpbr-subsurface.slang";
#if defined(PONDER_AUTODIFF_CHECK)
__include "bsdf/autodiff-check.slang";
#endif
