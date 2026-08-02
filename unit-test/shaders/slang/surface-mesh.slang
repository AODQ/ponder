#include "util-test.slang"
import "util-manifold-mesh";

// -----------------------------------------------------------------------------
// -- tests
// -----------------------------------------------------------------------------

// a bend-point on the hemisphere facing -z, away from grazing incidence
struct Config {
	manifold.MeshPointState bendPointState;
	f32v3 pointWi;
	f32 eta;
}

func config_random(inout rng: TestRng, mesh: manifold.MeshData) -> Config {
	var c: Config;
	// any triangle; the mesh is a sphere so every one is a valid caster
	let primitiveIndex = u32(rng.range(0.0, f32(mesh.triangleCount) - 0.001));
	// away from the edges, where a perturbation would immediately leave
	let uv = f32v2(rng.range(0.15, 0.4), rng.range(0.15, 0.4));
	c.pointWi = f32v3(
		rng.range(-0.5, 0.5), rng.range(-0.5, 0.5), rng.range(-4.0, -3.0)
	);
	c.bendPointState = manifold.mesh_point_state(mesh, primitiveIndex, uv, c.pointWi);
	c.eta = rng.range(1.2, 1.8);
	return c;
}

func test_solve_mesh() {
	let mesh = ((manifold.MeshData *)t_user_va())[0];
	var rng = t_rng();
	let c = config_random(rng, mesh);

	let start = manifold.SurfaceMesh.sample(f32v2(0.0), c.bendPointState);
	// only bend-points the ray origin can actually see are reachable
	if (dot(start.normal, normalize(c.pointWi - start.point)) < 0.3) { return; }

	// -- build a path this bend-point already satisfies
	let wi = normalize(c.pointWi - start.point);
	let wo = refract(-wi, start.normal, 1.0 / c.eta);
	T_CHECK(dot(wo, wo) > 0.0);

	var path: manifold.Path;
	path.pointWi = c.pointWi;
	path.pointWo = start.point + 0.5 * wo;
	path.eta = c.eta;

	T_CHECK_NEAR(
		manifold.constraint_half_vector<manifold.SurfaceMesh>(
			f32v2(0.0), c.bendPointState, path
		),
		f32v2(0.0),
		1e-4
	);

	// -- perturb, which may cross into a neighbouring triangle, then solve
	var guess = c.bendPointState;
	let ok = manifold.SurfaceMesh.advance(guess, rng.range3(-0.1, 0.1).xy);
	if (!ok) { return; }

	let result = manifold.solve<manifold.SurfaceMesh>(guess, path);
	T_CHECK(result.converged);

	let solved = manifold.SurfaceMesh.sample(f32v2(0.0), result.bendPointState);
	T_CHECK_NEAR(solved.point, start.point, 1e-3);
}

[shader("compute")]
[numthreads(skTestGroupSize, 1, 1)]
void main(const uint3 tid : SV_DispatchThreadID) {
	if (!t_begin(tid.x)) { return; }
	test_solve_mesh();
}
