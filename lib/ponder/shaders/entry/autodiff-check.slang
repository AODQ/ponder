import types;
import util;
import bsdf;
import medium;

// compiled only by the ponder-autodiff-check target, never dispatched
// each module lists and differentiates its own PonderDifferentiable functions

#if !defined(PONDER_AUTODIFF_CHECK)
#error "build with -DPONDER_AUTODIFF_CHECK"
#endif

RWStructuredBuffer<f32> checkOutput;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
	checkOutput[0] = (
		vkof::utilAutodiffCheck()
		+ vkof::bsdfAutodiffCheck()
		+ vkof::mediumAutodiffCheck()
	);
}
