#pragma once

// differentiable only in the autodiff check build,
// see entry/autodiff-check.slang
#if defined(PONDER_AUTODIFF_CHECK)
#define PonderDifferentiable [Differentiable]
#define PonderDifferentiableMethod [Differentiable] [NoDiffThis]
#define PonderNoDiff no_diff
#else
#define PonderDifferentiable
#define PonderDifferentiableMethod
#define PonderNoDiff
#endif
