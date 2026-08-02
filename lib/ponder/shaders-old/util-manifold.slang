module "util-manifold";
__exported import "util-types";

// specular manifold sampling
// solves for the point on a surface from a pointWi to the pointWo,
// using snell's law

/*
   pointWi
      \
       \
        \  wi
         \
          \  ^ normal
           \ |
            \|
-----------  x ------------ surface / bend-point
            /
           /
          / wo
         /
        /
       /
      /
     pointWo

	h = wi + eta * wo
	parallel to nor when path obey's snell's law at the bend point
*/

public namespace manifold
{

// -----------------------------------------------------------------------------
// -- shared types
// -----------------------------------------------------------------------------

// orthonormal pair spanning a tangent plane.
// both should be perpendicular to the frame normal
public struct Frame : IDifferentiable {
	public f32v3 s;
	public f32v3 t;
};

// a point on a surface and its tangent frame
public struct Sample : IDifferentiable {
	public f32v3 point;
	public f32v3 normal;
	// tangent frame at point, perpendicular to normal
	public Frame frame;
};

// the path vertices between a bend-point: pointWi->bend->pointWo.
public struct Path {
	// the point where the \omega_i ray starts
	public f32v3 pointWi;
	// the point where the \omega_o ray ends
	public f32v3 pointWo;
	// relative ior across the interface, \frac{\eta_o}{\eta_i}
	public f32 eta;
};

// surface tolerance, e.g. to tune world-space vs barycentric
public struct Settings {
	// give up after this many steps
	public int maxIter = 20;
	// how many times a step may be halved before the seed is abandoned
	public int maxBacktrack = 8;
	// residual below this counts as a valid path
	public f32 residualTol = 1e-5;
	// below this the jacobian is singular and the step is rejected
	public f32 jacobianDetFloor = 1e-6;
};

// orthonormal tangent frame for a unit surface normal.
//
// reference picks the direction the returned frame's S vector will point,
// but must not be parallel to the normal.
[Differentiable]
public func frame_from_normal(nor: f32v3, reference: no_diff f32v3) -> Frame {
	let s = normalize(reference - nor * dot(reference, nor));
	return Frame(s, cross(nor, s));
}

// -----------------------------------------------------------------------------
// -- surface
// -----------------------------------------------------------------------------

// surface for the solver to walk.
public interface ISurface {
	// carries the walk's position and other related bend-point state,
	// such as triangle data
	associatedtype PointState;

	// the surface a step of x away from where bend-point state currently sits
	[Differentiable]
	static func sample(
		// offset from state's position from how the surface parameterizes itself
		// E.g. barycentrics of a triangle, world-space tangent offset of a sphere
		x: f32v2,
		// solver's bend-point state
		no_diff bendPointState: PointState
	) -> Sample;

	// moves state's position by x. The new state after calling this will
	// satisfy sample(f32v2(0.0), state) == sample(x, oldState)
	// This is required to re-linearize the constraint around the new position
	//
	// returns false when the step cannot be taken, such as a surface whose
	// reprojection ray misses the geometry
	static func advance(inout bendPointState: PointState, x: f32v2) -> bool;
}

// per iteration manifold sampling. Stores the solver's state and if the
// iteration has converged.
public struct IterationResult<S : ISurface> {
	// the walk's final position. Can retrieve the bend-point using
	//   S.sample(f32v2(0.0), result.state)
	public S.PointState bendPointState;

	// iterations performed before stopping
	public int iterCount;

	// if the solver converged to a valid bend-point. This being false is not
	// an error, some starting guesses do not converge to a valid bend-point.
	public bool converged;
};

// -----------------------------------------------------------------------------
// -- constraint
// -----------------------------------------------------------------------------

// returns the measure of how poorly the snell's law constraint is satisfied,
//   for pointWi -> pointBend -> pointWo, with relative ior eta.
// h is parallel to surface normal when Snell's law is satisfied; if this
//   function returns (0, 0) then the path is valid.
[Differentiable]
public func constraint_half_vector<S : ISurface>(
	x: f32v2,
	no_diff bendPointState: S.PointState,
	no_diff path: Path,
) -> f32v2 {
	let surface = S.sample(x, bendPointState);
	let wi = normalize(path.pointWi - surface.point);
	let wo = normalize(path.pointWo - surface.point);
	let h = normalize(wi + path.eta * wo);
	// project h onto the surface tangent frame,
	// if this is (0, 0) then h is parallel to the surface normal and the path
	// obeys Snell's law at the bend-point.
	return f32v2(dot(h, surface.frame.s), dot(h, surface.frame.t));
}

// -----------------------------------------------------------------------------
// -- solver
// -----------------------------------------------------------------------------

// walks the surface from bend-point guess looking for a bend-point where the
// path obeys Snell's law, by newton's method on constraint_half_vector.
//
// most guesses do not reach one; will need to check IterationResult.converged
public func solve<S : ISurface>(
	bendPointGuess: S.PointState,
	path: Path,
	settings: Settings = {},
) -> IterationResult<S> {
	var bendPointState = bendPointGuess;
	var converged = false;
	int i = 0;
	for (i = 0; i < settings.maxIter; ++i) {
		// -- the jacobian's two columns, one forward-mode pass each.
		let dfdu = (
			fwd_diff(constraint_half_vector<S>)(
				diffPair(f32v2(0.0f), f32v2(1.0f, 0.0f)), bendPointState, path
			)
		);
		let dfdv = (
			fwd_diff(constraint_half_vector<S>)(
				diffPair(f32v2(0.0f), f32v2(0.0f, 1.0f)), bendPointState, path
			)
		);

		// -- residual compare
		let residual = length(dfdu.p);
		if (residual < settings.residualTol) {
			converged = true;
			break;
		}

		// -- compose the determinant
		let jacobian = transpose(f32m2x2(dfdu.d, dfdv.d));
		let jacobianDet = (
			jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
		);
		if (abs(jacobianDet) < settings.jacobianDetFloor) { break; }

		// -- compute the inverse-jacobian
		// will guide the step to the new bend-point
		let invJacobian = (
			f32m2x2(
				jacobian[1][1], -jacobian[0][1],
				-jacobian[1][0], jacobian[0][0]
			) / jacobianDet
		);
		let delta = mul(invJacobian, dfdu.p);

		// -- dampen the step to prevent over-shooting
		// same technique that mitsuba uses for their newton solver: halve on a
		// step that worsens the residual or fails to advance
		var beta = 1.0f;
		var stepped = false;
		for (int b = 0; b < settings.maxBacktrack; ++b) {
			let trial = (
				constraint_half_vector<S>(-beta * delta, bendPointState, path)
			);
			if (length(trial) < residual) {
				// check the advance is valid before committing to the step
				var steppedState = bendPointState;
				if (S.advance(steppedState, -beta * delta)) {
					bendPointState = steppedState;
					stepped = true;
					break;
				}
			}
			beta *= 0.5f;
		}
		if (!stepped) { break; }
	}
	return IterationResult<S>(bendPointState, i, converged);
}

// -----------------------------------------------------------------------------
// -- triangle surface
// -----------------------------------------------------------------------------

// bend-point's state on a triangle.
public struct TrianglePointState {
	// -- triangle vertices in world-space
	public f32v3 p0;
	public f32v3 p1;
	public f32v3 p2;

	// -- shading normals
	// must not be face normals, or the jacobian will be singular
	public f32v3 nor0;
	public f32v3 nor1;
	public f32v3 nor2;

	// barycentrics of the current bend-point
	public f32v2 uv;
};

// triangle surface
public struct SurfaceTriangle : ISurface {
	public typealias PointState = TrianglePointState;

	[Differentiable]
	public static func sample(
		x: f32v2,
		no_diff bendPointState: TrianglePointState,
	) -> Sample {
		let uv = bendPointState.uv + x;
		let w = 1.0f - uv.x - uv.y;
		let point = (
			w * bendPointState.p0 + uv.x * bendPointState.p1
			+ uv.y * bendPointState.p2
		);
		let nor = normalize(
			w * bendPointState.nor0 + uv.x * bendPointState.nor1
			+ uv.y * bendPointState.nor2
		);
		let edge = bendPointState.p1 - bendPointState.p0;
		return Sample(point, nor, frame_from_normal(nor, edge));
	}

	public static func advance(
		inout bendPointState: TrianglePointState,
		x: f32v2,
	) -> bool {
		bendPointState.uv += x;
		// a plane extends forever, so every step lands on a valid point
		return true;
	}
}

// -----------------------------------------------------------------------------
// -- sphere surface
// -----------------------------------------------------------------------------

// bend-point's state on a sphere.
public struct SpherePointState {
	// world-space center
	public f32v3 center;
	// world-space radius
	public f32 radius;
	// current bend-point, on the sphere, and its tangent frame
	public f32v3 anchor;
	// the current bend-point's tangent frame
	public Frame frame;
};

// construact bend-point state at a point on the sphere, with an arbitrary frame
// orientation.
public func sphere_point_state(
	center: f32v3,
	radius: f32,
	anchor: f32v3,
) -> SpherePointState {
	let nor = normalize(anchor - center);
	let reference = (
		abs(nor.z) < 0.9f ? f32v3(0.0f, 0.0f, 1.0f) : f32v3(1.0f, 0.0f, 0.0f)
	);
	return SpherePointState(
		center, radius, anchor, frame_from_normal(nor, reference)
	);
}

// sphere surface
public struct SurfaceSphere : ISurface {
	public typealias PointState = SpherePointState;

	[Differentiable]
	public static func sample(
		x: f32v2,
		no_diff bendPointState: SpherePointState,
	) -> Sample {
		let offset = (
			bendPointState.anchor
			+ x.x * bendPointState.frame.s + x.y * bendPointState.frame.t
		);
		let nor = normalize(offset - bendPointState.center);
		return Sample(
			bendPointState.center + nor * bendPointState.radius,
			nor,
			// the previous frame keeps this one from spinning as the walk
			// crosses the poles
			frame_from_normal(nor, bendPointState.frame.s)
		);
	}

	public static func advance(
		inout bendPointState: SpherePointState,
		x: f32v2,
	) -> bool {
		let surface = sample(x, bendPointState);
		bendPointState.anchor = surface.point;
		bendPointState.frame = surface.frame;
		// every tangent-plane offset projects back onto the sphere, so
		// every step is valid
		return true;
	}
}

} // namespace manifold
