# ponder

Piece of shit path tracer. Vulkan and Slang

Features:
 * openpbr materials
 * spectral sampling
 * mis / nee
 * wavefront kernels
 * participating media stack
 * curved ray transport
 * schwarzchild geodiscs
 * regir

Planned:
 * restir
 * "paint by light" using slang autodiff, probably the project capstone
 * differentiable caustics also using autodiff

AI slop was used during development:
* C++ entrypoint/renderpass boilerplate, imgui widgets and debug tools
* unit and regression tests
* the shader daemon recompiler tool
* debugging
* some shader data plumbing, such as most of `PushConstantGlobal`
* planning / scheduling
* DLSS owned by AI

Majority of the shaders / materials are hand-written.
