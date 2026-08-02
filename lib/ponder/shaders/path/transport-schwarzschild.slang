implementing path;

import types;
import util;
import model;

// note: since this goes into general relativity and has lots of
// math/derivations, the following symbols are shorthands that break
// my typical standard convention:
// x is position,
// p is momentum,
// t is coordinate time,
// r is radius,
// n is unit vector,
// f is schwarzschild factor,
// L is angular momentum,
// h is angular momentum magnitude,
// δ is the affine parameter along the geodesic
// pt is the time component of the momentum, i.e. p_t

namespace vkof {

// -----------------------------------------------------------------------------
// schwarzschild metric tensor, uses rk4 chord integration
public struct TraceTransport_TensorSchwarzschild
	: I_TraceTransport
{
	// media and index fields outside the lens sphere
	TraceTransport_Bend bend;

	public __init(vkof::WavefrontTransportState state) {
		bend = TraceTransport_Bend(state);
	}

	public func state() -> vkof::WavefrontTransportState {
		return bend.state();
	}

	func origin() -> f32v3 {
		return bend.tensor.flat.ori;
	}

	func direction() -> f32v3 {
		return bend.tensor.flat.dir;
	}

	// hands the ray back to the bend transport after a chord
	[mutating] func moveTo(ori: f32v3, dir: f32v3, elapsed: f32) {
		bend.tensor.flat.ori = ori;
		bend.tensor.flat.dir = dir;
		bend.tensor.flat.pathLength += elapsed;
	}

	[mutating] public func traceTransport(
		maxDistance: f32,
		inout vkof::SampleStream stream,
	) -> TraceTransportHit
	{
		let bhMass: f32 = vkof::global.blackholeMass;
		let bhSpin: f32 = vkof::global.blackholeSpin * bhMass;
		// rays traverse time in reverse of the light, which flips the spin
		let metricSpin: f32 = -bhSpin;
		let bhOri: f32v3 = vkof::global.blackholeOri;
		let bhLensRadius: f32 = bhMass * vkof::global.blackholeLensRadius;
		// -- calculate position relative to hole
		var relOri: f32v3 = this.origin() - bhOri;
		var vel: f32v3 = this.direction();

		// -- fallback to bend transport if black hole mass is zero
		if (bhMass <= 0.0f) {
			return this.bend.traceTransport(maxDistance, stream);
		}

		let distToLens: f32 = implRayIntersectSphere(relOri, vel, bhLensRadius);
		// -- fallback to bend transport if the ray will not enter the lens sphere
		if (distToLens < 0.0f || distToLens > maxDistance) {
			return this.bend.traceTransport(maxDistance, stream);
		}

		// -- check if geometry is between camera and bh lens
		let approachHit: ImplTraceStepHit = (
			implTraceStep(
				/*ori=*/ bhOri + relOri,
				/*dir=*/ vel,
				/*distMin=*/ 0.0f,
				/*distMax=*/ distToLens,
				/*alphaU=*/ stream.nextF32()
			)
		);
		if (approachHit.hit()) {
			let hit: TraceTransportHit = (
				implTransportHitFromStep(
					/*stepHit=*/ approachHit,
					/*chordOri=*/ bhOri + relOri,
					/*chordDirection=*/ vel,
					/*elapsedBefore=*/ 0.0f
				)
			);
			this.moveTo(hit.position, vel, hit.elapsedTime);
			return hit;
		}

		// -- calculate the starting origin on the surface of the lens sphere
		relOri += vel * distToLens;
		vkof::debugPathPoint(bhOri + relOri, vkof::skDebugPathPointChord);
		var elapsed: f32 = distToLens;

		// -- calculate geodesic initial state
		var state: GeodesicState = (
			GeodesicState(
				/*ori=*/ relOri,
				/*dir=*/ vel,
				/*time=*/ 0.0f,
			)
		);
		var prevState: GeodesicState = state;
		// -- calculate initial time component of momentum
		var pt: f32;
		geodesicStart(relOri, vel, bhMass, metricSpin, pt, state.p);

		// -- perform RK4 chord integration
		// each iteration:
		// 1) check if still inside influence of the black hole
		// 2) if not inside, fallback to flat metric
		// 3) if inside, perform one RK4 step to get new position and velocity
		// 4) cast a ray between the old and new position
		// 5) if the ray hits something, return the hit
		for (u32 chord = 0u; chord < vkof::global.blackholeMaxChords; ++chord) {
			let r: f32 = length(relOri);
			// -- inside the event horizon, terminate the path
			{
				let metric: KerrSchild = (
					implKerrSchildMetric(state.x, bhMass, metricSpin)
				);
				let rHorizon: f32 = bhMass + sqrt(bhMass*bhMass - bhSpin*bhSpin);
				if (metric.r < 1.01f*rHorizon) {
					return (
						implTransportHitLost(bhOri + relOri, normalize(vel), elapsed)
					);
				}
			}
			// -- leaving lens sphere, fallback to bend transport
			if (r > bhLensRadius && dot(relOri, vel) > 0.0) {
				this.moveTo(bhOri + relOri, normalize(vel), elapsed);
				var hit: TraceTransportHit = (
					this.bend.traceTransport(maxDistance - elapsed, stream)
				);
				hit.elapsedTime += elapsed;
				return hit;
			}

			// -- calculate one rk4 step; dx/dλ = vel, dv/dλ = a(relOri)
			let dt: f32 = (
				vkof::global.blackholeStepFactor * r
				/ length(
				geodesicStateDerivativeKerr(state, pt, bhMass, metricSpin).x
			)
			);
			let next: GeodesicState = (
				geodesicStateStepRk4(state, pt, bhMass, metricSpin, dt)
			);
			prevState = state;
			state = next;
			vkof::debugPathPoint(bhOri + state.x, vkof::skDebugPathPointChord);

			// -- trace the chord against geometry
			let chordVector: f32v3 = state.x - relOri;
			let chordLength: f32 = length(chordVector);
			let chordDirection: f32v3 = normalize(chordVector);
			let stepHit: ImplTraceStepHit = (
				implTraceStep(
					/*ori=*/ bhOri + relOri,
					/*dir=*/ chordDirection,
					/*distMin=*/ 0.0f,
					/*distMax=*/ chordLength,
					/*alphaU=*/ stream.nextF32()
				)
			);

			// -- trace the chord against an accretion disk
			// a hole spins about its local +y, so disk is hole-relative y
			[ForceUnroll]
			for (int stupid = 0; stupid < 1; ++stupid)
			{
				let y0: f32 = relOri.y;
				let y1: f32 = state.x.y;
				// TODO use inline expression instead of stupid "loop"
				if (y0 * y1 >= 0.0f) { break; }
				let t: f32 = y0 / (y0 - y1);
				let crossing: f32v3 = relOri + t * chordVector;
				let crossingOnDisk: f32v3 = f32v3(crossing.x, 0.0f, crossing.z);
				let radius: f32 = (
					sqrt(
						max(dot(crossingOnDisk, crossingOnDisk) - bhSpin*bhSpin, 0.0f)
					)
				);
				let diskInner: f32 = vkof::global.blackholeDiskInnerRadius * bhMass;
				let diskOuter: f32 = vkof::global.blackholeDiskOuterRadius * bhMass;
				let hitsDisk: bool = radius >= diskInner && radius <= diskOuter;

				// -- if it hits the disk, then return a transport hit at the disk
				// but only if the point is closer than the geometry hit, if any
				if (
					hitsDisk
					&& (!stepHit.hit() || t * chordLength < stepHit.distance)
				) {
					// -- return an emission event at the disk
					var hit: TraceTransportHit = (
						implTransportHitFromStep(
							/*stepHit=*/ stepHit,
							/*chordOri=*/ bhOri + relOri,
							/*chordDirection=*/ chordDirection,
							/*elapsedBefore=*/ elapsed
						)
					);
					let opticalDepth: f32 = (
						vkof::global.blackholeDiskOpticalDepth
						* pow(
							diskInner / radius,
							vkof::global.blackholeDiskOpticalDepthExponent
						)
					);
					let opacity: f32 = 1.0f - exp(-opticalDepth);
					hit.event = vkof::TransportEvent::Emission;
					hit.position = bhOri + crossingOnDisk;
					hit.elapsedTime = elapsed + t * chordLength;
					hit.incomingDirection = chordDirection;
					let omega: f32 = implDiskOmega(radius, bhMass, bhSpin);
					// geodesic time runs camera to disk, so the gas emitted earlier
					let emitTime: f32 = (
						vkof::global.blackholeDiskTime
						+ this.bend.emitTime()
						- distToLens
						- lerp(prevState.t, state.t, t)
					);
					// -- redshift factor
					{
						let pCross: f32v3 = lerp(prevState.p, state.p, t);
						// time-reversed gas velocity, matching metricSpin
						let velocity: f32v3 = (
							-omega * f32v3(crossingOnDisk.z, 0.0f, -crossingOnDisk.x)
						);
						// the gas' time component
						let metricCross: KerrSchild = (
							implKerrSchildMetric(crossingOnDisk, bhMass, metricSpin)
						);
						let lv: f32 = 1.0f + dot(metricCross.l, velocity);
						let utInverseSquared: f32 = (
							1.0f - dot(velocity, velocity) - metricCross.f*lv*lv
						);
						// inside the photon orbit no circular orbit exists
						if (utInverseSquared <= 0.0f) { break; }
						let ut: f32 = 1.0f / sqrt(utInverseSquared);
						// static camera; undefined inside the ergosphere
						let metricCam: KerrSchild = (
							implKerrSchildMetric(
								vkof::global.cameraOri - bhOri, bhMass, metricSpin
							)
						);
						let g: f32 = (
							pt
							/ (
								sqrt(max(1.0f - metricCam.f, 1.0e-6f))
								* ut
								* (pt + dot(pCross, velocity))
							)
						);
						hit.frequencyShift = g;
					}
					// -- emission and transmittance
					hit.emission.emission = (
						this.emissionSpectrum(
							/*λ=*/ this.bend.wavelengths(),
							/*temperature=*/ (
								this.diskTemperature(
									/*crossing=*/ crossingOnDisk,
									/*radius=*/ radius,
									/*omega=*/ omega,
									/*emitTime=*/ emitTime,
									/*spin=*/ bhSpin,
								)
							),
							/*frequencyShift=*/ hit.frequencyShift,
						)
						* opacity
					);
					hit.emission.transmittance = 1.0f - opacity;
					this.moveTo(hit.position, chordDirection, hit.elapsedTime);
					return hit;
				}
				skipHoleDisk: ;
			}

			if (stepHit.hit()) {
				let hit: TraceTransportHit = (
					implTransportHitFromStep(
						/*stepHit=*/ stepHit,
						/*chordOri=*/ bhOri + relOri,
						/*chordDirection=*/ chordDirection,
						/*elapsedBefore=*/ elapsed
					)
				);
				this.moveTo(hit.position, chordDirection, hit.elapsedTime);
				return hit;
			}
			// -- update ori and velocity for next chord
			elapsed += chordLength;
			relOri = state.x;
			// -- update velocity to the new momentum, normalized
			vel = (
				normalize(
				geodesicStateDerivativeKerr(state, pt, bhMass, metricSpin).x
			)
			);
		}
		// -- out of chords, the path is lost
		// e.g. it's orbiting the black hole
		return implTransportHitLost(bhOri + relOri, normalize(vel), elapsed);
	}

	public func visibilityQuery(
		origin: f32v3,
		target: f32v3,
		inout vkof::SampleStream stream,
	) -> vkof::TraceTransportConnection {
		return this.bend.visibilityQuery(origin, target, stream);
	}

	[mutating] public func newDirection(dir: f32v3) -> void {
		this.bend.newDirection(dir);
	}

	[mutating] public func terminateSecondaryWavelengths() -> void {
		bend.terminateSecondaryWavelengths();
	}

	public func wavelengths() -> Wavelengths {
		return this.bend.wavelengths();
	}

	public func emitTime() -> f32 {
		return this.bend.emitTime();
	}

	// shakura-sunyaev 1973
	// T(r) = T_max * f(r) / 0.488
	// f(r) = (r_in/r)^(3/4) * (1 - sqrt(r_in/r))^(1/4)
	// f(r) peaks at r = (49/36)*r_in, so dividing 0.488 is peak normalization
	func temperatureProfileShakuraSunyaev(r: f32) -> f32 {
		let rIn: f32 = (
			vkof::global.blackholeDiskInnerRadius * vkof::global.blackholeMass
		);
		let f: f32 = (
			pow(rIn/r, (3.0f/4.0f)) * pow(max(1.0f - sqrt(rIn / r), 0.0f), 0.25f)
		);
		return vkof::global.blackholeDiskTemperature * f / 0.488f;
	}

	// introduces some noise to the disk temperature profile
	func diskTemperature(
		crossing: f32v3,
		radius: f32,
		omega: f32,
		emitTime: f32,
		spin: f32,
	) -> f32 {
		let bhMass: f32 = vkof::global.blackholeMass;
		// -- co-rotating noise
		let angle: f32 = atan2(-crossing.z, crossing.x) - omega*emitTime;
		let noise: f32 = (
			vkof::noiseFbm(
				vkof::global.blackholeDiskNoiseScale
				* f32v3(log(radius / bhMass), cos(angle), sin(angle))
			)
		);
		let temperature: f32 = (
			this.temperatureProfileShakuraSunyaev(radius)
			* max(1.0f + vkof::global.blackholeDiskNoiseAmplitude*noise, 0.0f)
		);
		// -- hotspot on the inner disk
		let hotspotRadius: f32 = vkof::global.blackholeDiskHotspotRadius * bhMass;
		let hotspotAngle: f32 = (
			implDiskOmega(hotspotRadius, bhMass, spin) * emitTime
		);
		let hotspot: f32v3 = (
			hotspotRadius * f32v3(cos(hotspotAngle), 0.0f, -sin(hotspotAngle))
		);
		let hotspotSize: f32 = (
			max(vkof::global.blackholeDiskHotspotSize * bhMass, 1.0e-6f)
		);
		let offset: f32v3 = crossing - hotspot;
		let weight: f32 = (
			exp(-0.5f * dot(offset, offset) / (hotspotSize*hotspotSize))
		);
		// -- blend the hotspot temperature with the disk temperature
		return (
			lerp(
				temperature,
				vkof::global.blackholeDiskHotspotTemperature,
				weight
			)
		);
	}


	// planck's blackbody law
	func emissionSpectrum(
		λ: Wavelengths,
		temperature: f32,
		frequencyShift: f32
	) -> vkof::Spectrum {
		let h: f32 = 6.62607015e-34f;
		let c: f32 = 2.99792458e8f;
		let k: f32 = 1.380649e-23f;
		// will need to normalize to relative spectral radiance; the peak of
		// the blackbody absolute radiance and the typical 555nm for human vision
		let brightness: f32 = vkof::global.blackholeDiskBrightness;
		let reference: f32 = (
			implPlanck(555.0e-9f, vkof::global.blackholeDiskTemperature)
		);
		let λMeters: vkof::Throughput = λ.meters();
		var throughput: f32[vkof::skWavelengthSet];
		// -- calculate blackbody factor for each wavelength
		[ForceUnroll]
		for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
			throughput[i] = (
				brightness
				* pow(frequencyShift, 5.0f)
				* implPlanck(frequencyShift * λMeters[i], temperature)
				/ reference
			);
		}
		// -- normalize spectrum
		// the blackbody's spectrum peaks.
		// however normalizing the hottest temperature to 555nm should give
		// a good approximation
		return vkof::Spectrum::fromBlackbody(λ, throughput);
	}
}

// true when the path is curved from interference from the black hole
public func transportSchwarzschildCurves(
	state: WavefrontTransportState,
	maxDistance: f32,
) -> bool {
	let bhMass: f32 = vkof::global.blackholeMass;
	if (bhMass <= 0.0f) { return false; }
	let bhLensRadius: f32 = bhMass * vkof::global.blackholeLensRadius;
	let distToLens: f32 = (
		implRayIntersectSphere(
			state.ori - vkof::global.blackholeOri, state.dir, bhLensRadius
		)
	);
	return !(distToLens < 0.0f || distToLens > maxDistance);
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// returns the distance along the ray to the nearest intersection with sphere,
// -1 if no intersection
func implRayIntersectSphere(
	ori: f32v3,
	dir: f32v3,
	radius: f32,
) -> f32 {
	let c: f32 = dot(ori, ori) - radius * radius;
	if (c <= 0.0f) { return 0.0f; }
	let b: f32 = dot(ori, dir);
	let discriminant: f32 = b * b - c;
	if (discriminant < 0.0f || b >= 0.0f) { return -1.0f; }
	return -b - sqrt(discriminant);
}

// a(relOri) = -3 * M * h^2 * relOri / r^5
func implSchwarzschildAcceleration(
	ori: f32v3,
	mass: f32,
	h2: f32,
) -> f32v3 {
	let r2: f32 = dot(ori, ori);
	let r5: f32 = r2 * r2 * sqrt(r2);
	let acc: f32v3 = -3.0f * mass * h2 * ori / r5;
	return acc;
}

// B(λ, T) = 2hc^2 / λ^5 / (exp(hc / (λ k T)) - 1)
// λ in meters, T in kelvin, B in W/m^2/m
func implPlanck(λMeters: f32, temperature: f32) -> f32 {
	let h: f32 = 6.62607015e-34f;
	let c: f32 = 2.99792458e8f;
	let k: f32 = 1.380649e-23f;
	let factor1: f32 = 2.0f * h * c * c / pow(λMeters, 5.0f);
	let factor2: f32 = exp(h * c / (λMeters * k * temperature)) - 1.0f;
	return factor1 / factor2;
}

// position, spatial momentum and coordinate time
struct GeodesicState {
	f32v3 x;
	f32v3 p;
	f32 t;

	__init(f32v3 ori, f32v3 dir, f32 time) {
		this.x = ori;
		this.p = dir;
		this.t = time;
	}
}

// a + b*scale
func geodesicStateAdd(
	a: GeodesicState,
	b: GeodesicState,
	scale: f32
) -> GeodesicState {
	return (
		GeodesicState(
			/*x=*/ a.x + b.x * scale,
			/*p=*/ a.p + b.p * scale,
			/*t=*/ a.t + b.t * scale
		)
	);
}

// d(state)/dλ for schwarzschild in kerr-schild coordinates
func geodesicStateDerivative(
	s: GeodesicState,
	pt: f32,
	mass: f32,
) -> GeodesicState {
	// derivative of dx/dλ, dp/dλ,
	// where x is position and p is momentum
	// the derivatives / equations of motion are:
	// dx^u/dδ = δH/δpᵤ
	// dpᵤ/dδ = -δH/δx^u
	// δf/δxᵢ = -f·nᵢ/r
	// δL/δxᵢ = (pᵢ-nᵢ (p·n)) / r
	// dt/dλ = -pₜ + f · L
	// dx/dλ = p - f·L·n
	// dpₜ/dλ = 0
	// dp/dλ = (f/r) · [L·(p - n·(n·p))-½L²·n]
	let r: f32 = length(s.x);
	let n: f32v3 = s.x / r;
	let f: f32 = 2.0f * mass / r;
	let np: f32 = dot(n, s.p);
	let L: f32 = -pt + np;
	return (
		GeodesicState(
			/*x=*/ s.p - f * L * n,
			/*p=*/ (f / r) * (L * (s.p - n * np) - 0.5f * L * L * n),
			/*t=*/ -pt + f * L
		)
	);
}

func geodesicStateStepRk4(
	state: GeodesicState,
	pt: f32,
	mass: f32,
	spin: f32,
	dt: f32,
) -> GeodesicState {
	// -- compute the RK4 steps
	// slope at start
	let k1: GeodesicState = geodesicStateDerivativeKerr(state, pt, mass, spin);
	// slope at midpoint using k1; dt weighted by 0.5
	let k2: GeodesicState = (
		geodesicStateDerivativeKerr(
			geodesicStateAdd(state, k1, 0.5f * dt),
			pt,
			mass,
			spin
		)
	);
	// slope at midpoint using k2; dt weighted by 0.5
	let k3: GeodesicState = (
		geodesicStateDerivativeKerr(
			geodesicStateAdd(state, k2, 0.5f * dt),
			pt,
			mass,
			spin
		)
	);
	// slope at end using k3
	let k4: GeodesicState = (
		geodesicStateDerivativeKerr(
			geodesicStateAdd(state, k3, dt),
			pt,
			mass,
			spin
		)
	);
	// -- combine the RK4 steps to get the new state
	var result: GeodesicState = state;
	// the 6/3/3/6 are the weights for: (1, 2, 2, 1)/6
	result = geodesicStateAdd(result, k1, dt / 6.0f);
	result = geodesicStateAdd(result, k2, dt / 3.0f);
	result = geodesicStateAdd(result, k3, dt / 3.0f);
	result = geodesicStateAdd(result, k4, dt / 6.0f);
	return result;
}

// returns the initial geodesic momentum for a ray
func geodesicStart(
	x: f32v3,
	d: f32v3,
	mass: f32,
	spin: f32,
	out pt: f32,
	out p: f32v3,
) -> void
{
	let metric: KerrSchild = implKerrSchildMetric(x, mass, spin);
	let f: f32 = metric.f;
	let l: f32v3 = metric.l;
	// -- initial momentum scaled by the schwarzschild factor
	let s: f32 = dot(l, d);
	// -- solve for time component of momentum
	let qa: f32 = f-1.0f;
	let qb: f32 = 2.0f*f*s;
	let qc: f32 = 1.0f + f*s*s;
	let kt: f32 = (-qb - sqrt(qb*qb - 4.0f*qa*qc)) / (2.0f*qa);
	let lk: f32 = kt+s;
	// -- return the momentum components
	pt = -kt + f*lk;
	p = d + f*lk*l;
}

// hamiltonian for schwarzschild in kerr-schild coordinates
// H = ½ g^{uv} pᵤ pᵥ = ½ (-pₜ² + |p|² - f · L²)
// where L = -pₜ + n·p is the angular momentum
func geodesicHamiltonian(
	state: GeodesicState,
	pt: f32,
	mass: f32,
) -> f32 {
	let r: f32 = length(state.x);
	let f: f32 = 2.0f*mass/r;
	let l: f32 = -pt + dot(state.x/r, state.p);
	return 0.5f * (-pt*pt + dot(state.p, state.p) - f*l*l);
}

// returns the derivative of the geodesic state for a ray in the kerr metric
// adds spin a (0 .. M):
// w = |x|² - a²
// r = sqrt(½ (w + sqrt(w² + 4 a² z²)))
// f = 2 M r³ / (r⁴ + a² z²)
// l = ( (r x + a y) / (r² + a²) ); (r y - a x) / (r² + a²); z / r
// at a=0, this reduces to the schwarzschild metric
// r=|x|, f=2M/r, l = x/r
// equations of motion are the same but with n replaced by l
// the kerr gradients:
// ∇r = ( r² x, r² y, (r² - a²) z ) / (r (2r² - w))
// D = r⁴ + a² z²
// ∇f = 2M · ( 3r² ∇r·D - r³ ∇D (4r³ + 2 a² z ẑ) ) / D²
// d = r² + a²
// ∇lₓ = ( (∇r) x + r\hat{x} + a ŷ ) / d - (r x + a y) · 2r ∇r / d²
// ∇lᵧ = ( (∇r) y + r\hat{y} - a ŷ ) / d - (r y - a x) · 2r ∇r / d²
// ∇l_z = ẑ / r - z ∇r / r²
// ∇L = ∇lₓ · pₓ + ∇lᵧ · pᵧ + ∇l_z · p_z
// \hat{xyz} are unit axis vectors, ∇l is the gradient vector of l
func geodesicStateDerivativeKerr(
	state: GeodesicState,
	pt: f32,
	mass: f32,
	spin: f32,
) -> GeodesicState
{
	// spin along +z (the paper does not use +y)
	let x: f32v3 = state.x.zxy;
	let p: f32v3 = state.p.zxy;
	let a2: f32 = spin*spin;
	// -- ?
	let w: f32 = dot(x, x) - a2;
	// -- compute kerr-schild metric factors
	let metric: KerrSchild = implKerrSchildMetric(state.x, mass, spin);
	let r: f32 = metric.r;
	let f: f32 = metric.f;
	let lLocalFrame: f32v3 = metric.l.zxy;
	// -- radius in boyer-lindquist coordinates
	let d: f32 = r*r + a2;
	let D: f32 = r*r*r*r + a2*x.z*x.z;

	// -- the momentum along l
	let L: f32 = -pt + dot(lLocalFrame, p);
	// -- gradient of r and f
	let gr: f32v3 = (
		f32v3(
			r*r*x.x,
			r*r*x.y,
			(r*r + a2)*x.z
		)
		/ (r * (2.0f*r*r - w))
	);
	let gf: f32v3 = (
		2.0f*mass
		* (
			3.0f*r*r*gr*D
			- r*r*r * (4.0f*r*r*r*gr + f32v3(0.0f, 0.0f, 2.0f*a2*x.z))
		)
		/ (D*D)
	);
	// -- gradients of l
	let glx: f32v3 = (
		((gr*x.x + f32v3(r, spin, 0.0f)) / d)
		- ((r*x.x + spin*x.y) * 2.0f*r*gr / (d*d))
	);
	let gly: f32v3 = (
		((gr*x.y + f32v3(-spin, r, 0.0f)) / d)
		- ((r*x.y - spin*x.x) * 2.0f*r*gr / (d*d))
	);
	let glz: f32v3 = (
		f32v3(0.0f, 0.0f, 1.0f) / r - x.z*gr/(r*r)
	);
	// -- gradient of L
	let gL: f32v3 = glx*p.x + gly*p.y + glz*p.z;
	// -- return the derivative of the geodesic state
	return (
		GeodesicState(
			/*x=*/ (p - f*L*lLocalFrame).yzx,
			/*p=*/ (0.5f*gf*L*L + f*gL*L).yzx,
			/*t=*/ -pt + f*L
		)
	);
}

// kerr-schild metric at a point
struct KerrSchild {
	// boyer-lindquist radius
	f32 r;
	// metric strength
	f32 f;
	// kerr-schild direction
	f32v3 l;
};

func implKerrSchildMetric(
	x: f32v3,
	mass: f32,
	spin: f32,
) -> KerrSchild {
	let xl: f32v3 = x.zxy;
	let a2: f32 = spin*spin;
	let w: f32 = dot(xl, xl) - a2;
	let r: f32 = sqrt(0.5f * (w + sqrt(w*w + 4.0f*a2*xl.z*xl.z)));
	let d: f32 = r*r + a2;
	return {
		/*r=*/ r,
		/*f=*/ 2.0f*mass*r*r*r / (r*r*r*r + a2*xl.z*xl.z),
		/*l=*/ (
			f32v3(
				(r*xl.x + spin*xl.y)/d,
				(r*xl.y - spin*xl.x)/d,
				xl.z/r
			)
		).yzx,
	};
}

// kerr circular-orbit angular velocity
func implDiskOmega(radius: f32, mass: f32, spin: f32) -> f32 {
	let rotation: f32 = vkof::global.blackholeDiskRotation;
	return (
		rotation * sqrt(mass)
		/ (pow(radius, 1.5f) + rotation*spin*sqrt(mass))
	);
}
