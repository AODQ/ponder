#include "util-test.slang"
import "util-manifold";

// -----------------------------------------------------------------------------
// -- configs
// -----------------------------------------------------------------------------

struct ConfigTriangle {
	manifold.TrianglePointState bendPointState;
	f32v3 pointWi;
	f32v3 pointWo;
	f32 eta;
}

// triangle near z = 0, wi side below, wo side above, uv away from the edges
func config_random_triangle(inout rng: TestRng) -> ConfigTriangle {
	var c: ConfigTriangle;
	c.bendPointState.p0 = f32v3(-1.0, -1.0, 0.0) + rng.range3(-0.2, 0.2);
	c.bendPointState.p1 = f32v3(1.0, -1.0, 0.0) + rng.range3(-0.2, 0.2);
	c.bendPointState.p2 = f32v3(0.0, 1.0, 0.0) + rng.range3(-0.2, 0.2);
	c.bendPointState.nor0 = normalize(f32v3(0.0, 0.0, 1.0) + rng.range3(-0.3, 0.3));
	c.bendPointState.nor1 = normalize(f32v3(0.0, 0.0, 1.0) + rng.range3(-0.3, 0.3));
	c.bendPointState.nor2 = normalize(f32v3(0.0, 0.0, 1.0) + rng.range3(-0.3, 0.3));
	c.bendPointState.uv = f32v2(rng.range(0.1, 0.45), rng.range(0.1, 0.45));
	c.pointWi = f32v3(
		rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-3.0, -1.0)
	);
	c.pointWo = f32v3(
		rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(1.0, 3.0)
	);
	c.eta = rng.range(1.2, 1.8);
	return c;
}

struct ConfigSphere {
	manifold.SpherePointState bendPointState;
	f32v3 pointWi;
	// inside the sphere, where a ray entering through the bend-point goes
	f32v3 pointWo;
	f32 eta;
}

// sphere near the origin, wi side on -z, bend-point on the facing hemisphere
// away from grazing incidence
func config_random_sphere(inout rng: TestRng) -> ConfigSphere {
	var c: ConfigSphere;
	let center = rng.range3(-0.2, 0.2);
	let radius = rng.range(0.8, 1.2);
	let nor = normalize(
		f32v3(rng.range(-0.5, 0.5), rng.range(-0.5, 0.5), -1.0)
	);
	c.bendPointState = manifold.sphere_point_state(
		center, radius, center + radius * nor
	);
	c.pointWi = center + f32v3(
		rng.range(-0.5, 0.5), rng.range(-0.5, 0.5), rng.range(-3.0, -2.0)
	);
	c.pointWo = center + 0.5 * radius * normalize(
		f32v3(rng.range(-0.5, 0.5), rng.range(-0.5, 0.5), 1.0)
	);
	c.eta = rng.range(1.2, 1.8);
	return c;
}

// a real two-interface path through a solid glass sphere: light -> entry ->
// exit -> receiver. built by refracting forwards twice, so both bend-points
// satisfy snell's law by construction
struct ConfigTwoInterface {
	manifold.SpherePointState entryState;
	manifold.SpherePointState exitState;
	f32v3 pointLight;
	f32v3 pointReceiver;
	// absolute ior of the glass; the interfaces take it and its reciprocal
	f32 ior;
}

// light on -z, entry on the facing hemisphere, offset far enough off-axis
// that the refraction is substantially oblique -- a near-axial path satisfies
// the constraint under either eta and would make the test vacuous
func config_random_two_interface(
	inout rng: TestRng,
	out valid: bool,
) -> ConfigTwoInterface {
	var c: ConfigTwoInterface;
	valid = false;

	let center = rng.range3(-0.2, 0.2);
	let radius = rng.range(0.8, 1.2);
	c.ior = rng.range(1.3, 1.7);
	c.pointLight = center + f32v3(
		rng.range(-0.3, 0.3), rng.range(-0.3, 0.3), rng.range(-4.0, -3.0)
	);

	// -- entry point, well off the axis
	let entryNor = normalize(
		f32v3(rng.range(0.35, 0.6), rng.range(0.35, 0.6), -1.0)
	);
	let entryPoint = center + radius * entryNor;

	// -- refract in. refract takes eta_i/eta_t, air to glass here
	let dirIn = normalize(entryPoint - c.pointLight);
	if (dot(dirIn, entryNor) >= 0.0) { return c; }
	let dirInside = refract(dirIn, entryNor, 1.0 / c.ior);
	if (dot(dirInside, dirInside) <= 0.0) { return c; }

	// -- march to the far side. the entry point is on the sphere, so the
	// second root of |o + t*d|^2 = r^2 is just -2*dot(o, d)
	let offset = entryPoint - center;
	let travel = -2.0 * dot(offset, dirInside);
	if (travel <= 1e-4) { return c; }
	let exitPoint = entryPoint + travel * dirInside;
	let exitNor = normalize(exitPoint - center);

	// -- refract out. the ray arrives from inside, so the facing normal is
	// the inward one and the ratio is the reciprocal of the entry's
	if (dot(dirInside, exitNor) <= 0.0) { return c; }
	let dirOut = refract(dirInside, -exitNor, c.ior);
	if (dot(dirOut, dirOut) <= 0.0) { return c; }

	c.entryState = manifold.sphere_point_state(center, radius, entryPoint);
	c.exitState = manifold.sphere_point_state(center, radius, exitPoint);
	c.pointReceiver = exitPoint + rng.range(1.5, 3.0) * dirOut;
	valid = true;
	return c;
}

// -----------------------------------------------------------------------------
// -- test helpers
// -----------------------------------------------------------------------------

// refracts through the bend-point to place a pointWo the constraint must
// already be satisfied by, pinning eta's direction and the wi/wo convention
func path_refracting_forward<S : manifold.ISurface>(
	bendPointState: S.PointState,
	pointWi: f32v3,
	eta: f32,
	distance: f32,
) -> manifold.Path {
	let surface = S.sample(f32v2(0.0), bendPointState);
	let wi = normalize(pointWi - surface.point);
	let norFacing = dot(surface.normal, wi) < 0.0 ? -surface.normal : surface.normal;
	// refract takes eta_i/eta_t, the reciprocal of the constraint's eta
	let wo = refract(-wi, norFacing, 1.0 / eta);
	// zero on total internal reflection
	t_check(__LINE__, dot(wo, wo) > 0.0);

	var path: manifold.Path;
	path.pointWi = pointWi;
	path.pointWo = surface.point + distance * wo;
	path.eta = eta;
	return path;
}

// autodiff's jacobian against central differences, plus a non-zero
// determinant: two matching zero columns would agree perfectly while telling
// the solver nothing.
//
// checkLine is the caller's, so a failure names the surface under test
func check_constraint_jacobian<S : manifold.ISurface>(
	checkLine: uint,
	bendPointState: S.PointState,
	path: manifold.Path,
) {
	// -- the jacobian's two columns, one forward-mode pass each
	let du = (
		fwd_diff(manifold.constraint_half_vector<S>)(
			diffPair(f32v2(0.0f), f32v2(1.0f, 0.0f)), bendPointState, path
		)
	);
	let dv = (
		fwd_diff(manifold.constraint_half_vector<S>)(
			diffPair(f32v2(0.0f), f32v2(0.0f, 1.0f)), bendPointState, path
		)
	);

	// -- central differences. 1e-3 keeps roundoff (eps/h) an order of
	// magnitude under the tolerance; a smaller step makes this worse
	let h = 1e-3;
	let duPlus = manifold.constraint_half_vector<S>(
		f32v2(h, 0.0), bendPointState, path
	);
	let duMinus = manifold.constraint_half_vector<S>(
		f32v2(-h, 0.0), bendPointState, path
	);
	let dvPlus = manifold.constraint_half_vector<S>(
		f32v2(0.0, h), bendPointState, path
	);
	let dvMinus = manifold.constraint_half_vector<S>(
		f32v2(0.0, -h), bendPointState, path
	);

	t_check_near(checkLine, du.d, (duPlus - duMinus) / (2.0 * h), 1e-3);
	t_check_near(checkLine, dv.d, (dvPlus - dvMinus) / (2.0 * h), 1e-3);

	let jacobianDet = du.d.x * dv.d.y - du.d.y * dv.d.x;
	t_check(checkLine, abs(jacobianDet) > 1e-6);
}

// -----------------------------------------------------------------------------
// -- tests
// -----------------------------------------------------------------------------

func test_constraint_triangle() {
	var rng = t_rng();
	let c = config_random_triangle(rng);

	var path: manifold.Path;
	path.pointWi = c.pointWi;
	path.pointWo = c.pointWo;
	path.eta = c.eta;
	check_constraint_jacobian<manifold.SurfaceTriangle>(
		__LINE__, c.bendPointState, path
	);

	// -- the constraint must vanish on a path built by refracting forwards.
	// the jacobian checks above cannot catch an inverted eta or a swapped
	// wi/wo: they differentiate whatever constraint is written
	{
		let valid = path_refracting_forward<manifold.SurfaceTriangle>(
			c.bendPointState, c.pointWi, c.eta, /*distance=*/ 2.0
		);
		T_CHECK_NEAR(
			manifold.constraint_half_vector<manifold.SurfaceTriangle>(
				f32v2(0.0), c.bendPointState, valid
			),
			f32v2(0.0),
			1e-4
		);
	}
}

func test_constraint_sphere() {
	var rng = t_rng();
	let c = config_random_sphere(rng);

	var path: manifold.Path;
	path.pointWi = c.pointWi;
	path.pointWo = c.pointWo;
	path.eta = c.eta;
	// -- catches a reprojection left outside the differentiated region, which
	// otherwise only shows up as the solver taking extra iterations
	check_constraint_jacobian<manifold.SurfaceSphere>(
		__LINE__, c.bendPointState, path
	);

	{
		let valid = path_refracting_forward<manifold.SurfaceSphere>(
			c.bendPointState, c.pointWi, c.eta, /*distance=*/ 0.5
		);
		T_CHECK_NEAR(
			manifold.constraint_half_vector<manifold.SurfaceSphere>(
				f32v2(0.0), c.bendPointState, valid
			),
			f32v2(0.0),
			1e-4
		);
	}
}

// the chain of plan-caustics-integration.md 1a hangs on one convention: the
// exit interface takes the reciprocal eta of the entry. a single shared eta
// still converges, just to the wrong point, so pin it down here before the
// chain solver exists to pin it down in
func test_two_interface_eta_convention() {
	var rng = t_rng();
	bool valid;
	let c = config_random_two_interface(rng, valid);
	T_CHECK(valid);
	if (!valid) { return; }

	let entryPoint = (
		manifold.SurfaceSphere.sample(f32v2(0.0), c.entryState).point
	);
	let exitPoint = (
		manifold.SurfaceSphere.sample(f32v2(0.0), c.exitState).point
	);

	// -- entry: the wi side is air and the wo side is glass, so eta is the
	// glass ior as the single-bend-point case already uses it
	var entryPath: manifold.Path;
	entryPath.pointWi = c.pointLight;
	entryPath.pointWo = exitPoint;
	entryPath.eta = c.ior;
	T_CHECK_NEAR(
		manifold.constraint_half_vector<manifold.SurfaceSphere>(
			f32v2(0.0), c.entryState, entryPath
		),
		f32v2(0.0),
		1e-4
	);

	// -- exit: the wi side is now glass and the wo side air, so eta inverts
	var exitPath: manifold.Path;
	exitPath.pointWi = entryPoint;
	exitPath.pointWo = c.pointReceiver;
	exitPath.eta = 1.0 / c.ior;
	T_CHECK_NEAR(
		manifold.constraint_half_vector<manifold.SurfaceSphere>(
			f32v2(0.0), c.exitState, exitPath
		),
		f32v2(0.0),
		1e-4
	);

	// -- the negative. carrying the entry's eta over to the exit must leave
	// the constraint clearly unsatisfied, or the check above proves nothing
	var sharedEtaPath = exitPath;
	sharedEtaPath.eta = c.ior;
	let residual = (
		manifold.constraint_half_vector<manifold.SurfaceSphere>(
			f32v2(0.0), c.exitState, sharedEtaPath
		)
	);
	T_CHECK(length(residual) > 1e-2);
}

func test_solve_triangle() {
	var rng = t_rng();
	let c = config_random_triangle(rng);
	let path = path_refracting_forward<manifold.SurfaceTriangle>(
		c.bendPointState, c.pointWi, c.eta, /*distance=*/ 2.0
	);

	// -- perturb off the known bend-point, then solve back to it
	var guess = c.bendPointState;
	guess.uv += rng.range3(-0.1, 0.1).xy;

	let result = manifold.solve<manifold.SurfaceTriangle>(guess, path);
	T_CHECK(result.converged);
	T_CHECK_NEAR(result.bendPointState.uv, c.bendPointState.uv, 1e-4);
}

func test_solve_sphere() {
	var rng = t_rng();
	let c = config_random_sphere(rng);
	let path = path_refracting_forward<manifold.SurfaceSphere>(
		c.bendPointState, c.pointWi, c.eta, /*distance=*/ 0.5
	);
	let expected = manifold.SurfaceSphere.sample(f32v2(0.0), c.bendPointState);

	// -- perturb along the tangent plane; advance reprojects onto the sphere
	var guess = c.bendPointState;
	manifold.SurfaceSphere.advance(
		guess, 0.05 * c.bendPointState.radius * rng.range3(-1.0, 1.0).xy
	);

	let result = manifold.solve<manifold.SurfaceSphere>(guess, path);
	T_CHECK(result.converged);

	// -- world units; there is no shared parameterization to compare
	let solved = manifold.SurfaceSphere.sample(f32v2(0.0), result.bendPointState);
	T_CHECK_NEAR(solved.point, expected.point, 1e-4);
}

// -----------------------------------------------------------------------------
// -- test entry
// -----------------------------------------------------------------------------

[shader("compute")]
[numthreads(skTestGroupSize, 1, 1)]
void main(const uint3 tid : SV_DispatchThreadID) {
	if (!t_begin(tid.x)) { return; }

	test_constraint_triangle();
	test_constraint_sphere();
	test_solve_triangle();
	test_solve_sphere();
}
