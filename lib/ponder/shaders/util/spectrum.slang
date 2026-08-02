implementing util;

import types;

#include "include/inline-expression.slang"
#include "include/differentiable.slang"

namespace vkof
{

// -----------------------------------------------------------------------------
// -- spectrum configuration
// -----------------------------------------------------------------------------

static public const u32 skWavelengthSet = 4u;

#if defined(PONDER_SPECTRUM_RGB)
public typealias Spectrum = Spectrum_Rgb;
#else
public typealias Spectrum = Spectrum_SampledSet<skWavelengthSet>;
#endif
public typealias Wavelengths = WavelengthSet<skWavelengthSet>;
public typealias Throughput = f32[skWavelengthSet];

// -----------------------------------------------------------------------------
// -- spectrum api
// -----------------------------------------------------------------------------

// a set of wavelengths in a spectrum, along with per-channel sampling pdf
// higher channel count means more sampled points in a spectrum, at the cost
// of more memory/bandwidth/register pressure
public struct WavelengthSet<let N: u32> {
	public __init(f32 lambdas[N], f32 pdfs[N]) {
		this.lambdas = lambdas;
		this.pdfs = pdfs;
	}

	public func meters() -> f32[N] {
		var m: f32[N];
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			// nm->m
			m[i] = lambdas[i] * 1e-9f;
		}
		return m;
	}

	// wavelength per channel in nanometers;
	// channel 0 is the hero wavelength
	public f32 lambdas[N];
	// per-land sampling pdf (1/nm) for each channel
	// a value of 0.0 means the channel is not sampled
	public f32 pdfs[N];
};

// sample a set of wavelengths across the visible spectrum with a
// uniform distribution
public func wavelengthSampleUniform(u: f32) -> Wavelengths
{
	let lambdaMin: f32 = vkof::skLambdaMin;
	let lambdaMax: f32 = vkof::skLambdaMax;
	// offset each channel by a quarter range into the spectrum
	let offset: f32v4 = f32v4(0.0f, 0.25f, 0.5f, 0.75f);
	var λ: Wavelengths;
	[ForceUnroll]
	for (u32 i = 0u; i < 4u; ++i) {
		λ.lambdas[i] = lambdaMin + frac(u + offset[i]) * (lambdaMax - lambdaMin);
		λ.pdfs[i] = 1.0f / (lambdaMax - lambdaMin);
	}
	return λ;
}

public func wavelengthStandard() -> Wavelengths
{
	return Wavelengths(
		{450.0f, 550.0f, 650.0f, 700.0f},
		{1.0f, 1.0f, 1.0f, 0.0f}
	);
}

// can represent a color either as rgb or a sampled spectrum
public interface I_Spectrum {
	public static func fromRgb(rgb: f32v3, λ: Wavelengths,) -> This;
	public static func fromBlackbody<uint N>(
		λ: Wavelengths,
		amounts: f32[N]
	) -> This;

	// converts the spectrum to rgb using provided wavelength
	// note that for rgb (fixed wavelength set) the wavelengths are ignored
	public func asRgb(λ: Wavelengths) -> f32v3;
	public func asXyz(λ: Wavelengths) -> f32v3;
	// the d65 illuminant, scaled to luminance 1
	public static func illuminantD65(λ: Wavelengths) -> This;

	// returns a spectrum with the exponential of each channel
	public func exp() -> This;
	// returns a spectrum with the average of all channels
	public func average() -> f32;
	// returns the value of the hero channel (channel 0)
	public func hero() -> f32;
	public static func zero() -> This;
	public static func one() -> This;

	public static func channelCount() -> u32;
	public func channel(index: u32) -> f32;
	// fromRgb for values above 1 (e.g. emission, extinction)
	public static func fromRgbUnbounded(rgb: f32v3, λ: Wavelengths) -> This;

	public This mul(This other);
	public This add(This other);
};

// -----------------------------------------------------------------------------

// stores the intensity of three fixed bands (450, 550, 650 nm)
public struct Spectrum_Rgb : I_Spectrum, IDifferentiable {
	f32v3 value;

	PonderDifferentiable
	__init(f32v3 value) {
		this.value = value;
	}

	public static func fromRgb(
		rgb: f32v3,
		λ: Wavelengths,
	) -> Spectrum_Rgb{
		return Spectrum_Rgb(rgb);
	}

	public static func fromBlackbody<uint N>(
		λ: Wavelengths,
		amounts: f32[N]
	) -> Spectrum_Rgb {
		// TODO- verify this is correct, or at least close enough
		// doesn't really need to be exact
		var value: f32v3 = f32v3(0.0f);
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			if (λ.pdfs[i] == 0.0f) { continue; }
			value += implCieXyz(λ.lambdas[i]) * (amounts[i] / λ.pdfs[i]);
		}
		value /= f32(N) * skCieYIntegral;
		return This(vkof::xyzToLinearRec709(value));
	}

	public func asRgb(λ: Wavelengths) -> f32v3  {
		return value;
	}

	// linear rec.709 primaries, d65 white -> cie xyz
	public func asXyz(λ: Wavelengths) -> f32v3 {
		return spectrumAsXyz(this, λ);
	}

	// d65 is the rec.709 white point
	public static func illuminantD65(λ: Wavelengths) -> Spectrum_Rgb {
		return Spectrum_Rgb(f32v3(1.0f));
	}

	public func exp() -> This {
		return spectrumExp(this);
	}
	public func average() -> f32 {
		return spectrumAverage(this);
	}
	public func hero() -> f32 {
		return value.x;
	}
	public static func fromRgbUnbounded(rgb: f32v3, λ: Wavelengths) -> This {
		return Spectrum_Rgb(rgb);
	}
	public static func channelCount() -> u32 {
		return 3u;
	}
	public func channel(index: u32) -> f32 {
		return value[index];
	}
	public static func zero() -> This {
		return Spectrum_Rgb(f32v3(0.0f));
	}
	public static func one() -> This {
		return Spectrum_Rgb(f32v3(1.0f));
	}

	public func mul(other: Spectrum_Rgb) -> Spectrum_Rgb {
		return this * other;
	}

	public func add(other: Spectrum_Rgb) -> Spectrum_Rgb {
		return this + other;
	}
};


// -----------------------------------------------------------------------------

// stores the intensity at each sampled wavelength
// used in conjunction with the Wavelengths to represent a spectrum
public struct Spectrum_SampledSet<let N: u32> : I_Spectrum, IDifferentiable {
	public f32 amounts[N];

	PonderDifferentiable
	__init(f32 amounts[N]) {
		this.amounts = amounts;
	}

	public static func fromRgb(
		rgb: f32v3,
		λ: Wavelengths,
	) -> This {
		if (N == skWavelengthSet) {
			return reinterpret<This>(implSpectrumSigmoidUpsample(rgb, λ));
		} else {
			// unimplemented
			return This({});
		}
	}

	public static func fromBlackbody<uint M>(
		λ: Wavelengths,
		amounts: f32[M]
	) -> This {
		static_assert(M == N, "mismatched wavelength set size");
		var a: f32[N];
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			a[i] = amounts[i];
		}
		return This(a);
	}

	public func asRgb(λ: Wavelengths) -> f32v3 {
		var xyz: f32v3 = f32v3(0.0f, 0.0f, 0.0f);
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			// convert wavelength to xyz using CIE 1931 color matching functions
			if (λ.pdfs[i] == 0.0f) { continue; }
			xyz += implCieXyz(λ.lambdas[i]) * (amounts[i] / λ.pdfs[i]);
		}
		// average over all N lanes, normalized
		xyz /= f32(N) * skCieYIntegral;
		return vkof::xyzToLinearRec709(xyz);
	}

	// linear rec.709 primaries, d65 white -> cie xyz
	public static func illuminantD65(λ: Wavelengths) -> This {
		var amounts: f32[N];
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			amounts[i] = vkof::cieD65(λ.lambdas[i]);
		}
		return This(amounts);
	}

	public func asXyz(λ: Wavelengths) -> f32v3 {
		return spectrumAsXyz(reinterpret<SpectrumSampled>(this), λ);
	}

	public func exp() -> This {
		return reinterpret<This>(spectrumExp(reinterpret<SpectrumSampled>(this)));
	}
	public func average() -> f32 {
		return spectrumAverage(reinterpret<SpectrumSampled>(this));
	}
	public func hero() -> f32 {
		return amounts[0];
	}
	public static func zero() -> This {
		var a: f32[N];
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			a[i] = 0.0f;
		}
		return This(a);
	}
	public static func one() -> This {
		var a: f32[N];
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			a[i] = 1.0f;
		}
		return This(a);
	}
	public static func channelCount() -> u32 {
		return N;
	}
	public func channel(index: u32) -> f32 {
		return amounts[index];
	}
	public static func fromRgbUnbounded(rgb: f32v3, λ: Wavelengths) -> This {
		let scale: f32 = max(max(rgb.r, rgb.g), rgb.b);
		if (scale <= 0.0f) { return zero(); }
		let normalized: This = fromRgb(rgb / scale, λ);
		var a: f32[N];
		[ForceUnroll]
		for (u32 i = 0u; i < N; ++i) {
			a[i] = normalized.amounts[i] * scale;
		}
		return This(a);
	}

	public func mul(other: This) -> This {
		return reinterpret<This>(
			reinterpret<SpectrumSampled>(this)
			* reinterpret<SpectrumSampled>(other)
		);
	}

	public func add(other: This) -> This {
		return reinterpret<This>(
			reinterpret<SpectrumSampled>(this)
			+ reinterpret<SpectrumSampled>(other)
		);
	}
};
public typealias SpectrumSampled = Spectrum_SampledSet<skWavelengthSet>;

// -----------------------------------------------------------------------------
// -- differentiable operations
// -----------------------------------------------------------------------------

PonderDifferentiable
public func spectrumExp(s: SpectrumSampled) -> SpectrumSampled {
	var a: f32[skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < skWavelengthSet; ++i) {
		a[i] = ::exp(s.amounts[i]);
	}
	return SpectrumSampled(a);
}

PonderDifferentiable
public func spectrumExp(s: Spectrum_Rgb) -> Spectrum_Rgb {
	return Spectrum_Rgb(::exp(s.value));
}

PonderDifferentiable
public func spectrumAverage(s: SpectrumSampled) -> f32 {
	var sum: f32 = 0.0f;
	[ForceUnroll]
	for (u32 i = 0u; i < skWavelengthSet; ++i) {
		sum += s.amounts[i];
	}
	return sum / (f32)(skWavelengthSet);
}

PonderDifferentiable
public func spectrumAverage(s: Spectrum_Rgb) -> f32 {
	return (s.value.x + s.value.y + s.value.z) / 3.0f;
}

// normalized so Y = 1 for d65 white
PonderDifferentiable
public func spectrumAsXyz(
	s: SpectrumSampled,
	PonderNoDiff λ: Wavelengths,
) -> f32v3 {
	var xyz: f32v3 = f32v3(0.0f, 0.0f, 0.0f);
	[ForceUnroll]
	for (u32 i = 0u; i < skWavelengthSet; ++i) {
		if (λ.pdfs[i] == 0.0f) { continue; }
		xyz += PonderNoDiff implCieXyz(λ.lambdas[i]) * (s.amounts[i] / λ.pdfs[i]);
	}
	return xyz / (f32(skWavelengthSet) * skCieYIntegral);
}

// linear rec.709 primaries, d65 white
PonderDifferentiable
public func spectrumAsXyz(
	s: Spectrum_Rgb,
	PonderNoDiff λ: Wavelengths,
) -> f32v3 {
	return f32v3(
		0.4124564f * s.value.x + 0.3575761f * s.value.y + 0.1804375f * s.value.z,
		0.2126729f * s.value.x + 0.7151522f * s.value.y + 0.0721750f * s.value.z,
		0.0193339f * s.value.x + 0.1191920f * s.value.y + 0.9503041f * s.value.z
	);
}

// -----------------------------------------------------------------------------
// -- spectrum freeform utilitiy
// -----------------------------------------------------------------------------

// terminates the secondary lanes of a wavefront path
// when a path hits something wavelength-dependent, the other lanes are
// terminated adn the hero lane continues.
public func wavelengthsTerminateSecondary(λ : Wavelengths) -> Wavelengths {
	let allZero: bool = InlineExpr {
		[ForceUnroll]
		for (u32 i = 0u; i < 3u; ++i) {
			if (λ.pdfs[1+i] > 0.0f) { return false; }
		}
		return true;
	};
	if (allZero) { return λ; }
	return (
		Wavelengths(
			λ.lambdas,
			{λ.pdfs[0] / f32(skWavelengthSet), 0.0f, 0.0f, 0.0f}
		)
	);
}

} // namespace vkof

// -- free form operators

PonderDifferentiable
public vkof::SpectrumSampled operator+(
	vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a.amounts[i] + b.amounts[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator*(
	vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a.amounts[i] * b.amounts[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator*(
	vkof::SpectrumSampled a,
	f32 b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a.amounts[i] * b;
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator/(
	vkof::SpectrumSampled a,
	f32 b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a.amounts[i] / b;
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator*(
	vkof::SpectrumSampled a,
	f32 b[vkof::skWavelengthSet]
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a.amounts[i] * b[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator-(
	vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a.amounts[i] - b.amounts[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator-(
	f32 a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a - b.amounts[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator*(
	f32 a,
	vkof::SpectrumSampled b
) {
	return b * a;
}
PonderDifferentiable
public void operator+=(
	inout vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	a = a + b;
}
PonderDifferentiable
public void operator*=(
	inout vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	a = a * b;
}
PonderDifferentiable
public void operator*=(
	inout vkof::SpectrumSampled a,
	f32 b
) {
	a = a * b;
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator+(
	vkof::Spectrum_Rgb a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a.value + b.value);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator*(
	vkof::Spectrum_Rgb a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a.value * b.value);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator*(
	vkof::Spectrum_Rgb a,
	f32 b
) {
	return vkof::Spectrum_Rgb(a.value * b);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator/(
	vkof::Spectrum_Rgb a,
	f32 b
) {
	return vkof::Spectrum_Rgb(a.value / b);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator-(
	vkof::Spectrum_Rgb a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a.value - b.value);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator-(
	f32 a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a - b.value);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator*(
	f32 a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a * b.value);
}
PonderDifferentiable
public void operator+=(
	inout vkof::Spectrum_Rgb a,
	vkof::Spectrum_Rgb b
) {
	a = a + b;
}
PonderDifferentiable
public void operator*=(
	inout vkof::Spectrum_Rgb a,
	vkof::Spectrum_Rgb b
) {
	a = a * b;
}
PonderDifferentiable
public void operator*=(
	inout vkof::Spectrum_Rgb a,
	f32 b
) {
	a = a * b;
}

PonderDifferentiable
public vkof::SpectrumSampled max(
	vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = max(a.amounts[i], b.amounts[i]);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled min(
	vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = min(a.amounts[i], b.amounts[i]);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled max(
	vkof::SpectrumSampled a,
	f32 b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = max(a.amounts[i], b);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled min(vkof::SpectrumSampled a, f32 b) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = min(a.amounts[i], b);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::Spectrum_Rgb max(vkof::Spectrum_Rgb a, vkof::Spectrum_Rgb b) {
	return vkof::Spectrum_Rgb(max(a.value, b.value));
}
PonderDifferentiable
public vkof::Spectrum_Rgb min(vkof::Spectrum_Rgb a, vkof::Spectrum_Rgb b) {
	return vkof::Spectrum_Rgb(min(a.value, b.value));
}
PonderDifferentiable
public vkof::Spectrum_Rgb max(vkof::Spectrum_Rgb a, f32 b) {
	return vkof::Spectrum_Rgb(max(a.value, f32v3(b)));
}
PonderDifferentiable
public vkof::Spectrum_Rgb min(vkof::Spectrum_Rgb a, f32 b) {
	return vkof::Spectrum_Rgb(min(a.value, f32v3(b)));
}

// -- per-channel division, scalar addition, sqrt, clamp

PonderDifferentiable
public vkof::SpectrumSampled operator/(
	vkof::SpectrumSampled a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a.amounts[i] / b.amounts[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled operator+(
	f32 a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a + b.amounts[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled sqrt(vkof::SpectrumSampled a) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = sqrt(a.amounts[i]);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled clamp(vkof::SpectrumSampled a, f32 lo, f32 hi) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = clamp(a.amounts[i], lo, hi);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator/(
	vkof::Spectrum_Rgb a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a.value / b.value);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator+(
	f32 a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a + b.value);
}
PonderDifferentiable
public vkof::Spectrum_Rgb sqrt(vkof::Spectrum_Rgb a) {
	return vkof::Spectrum_Rgb(sqrt(a.value));
}
PonderDifferentiable
public vkof::Spectrum_Rgb clamp(vkof::Spectrum_Rgb a, f32 lo, f32 hi) {
	return vkof::Spectrum_Rgb(clamp(a.value, f32v3(lo), f32v3(hi)));
}
PonderDifferentiable
public vkof::SpectrumSampled operator/(
	f32 a,
	vkof::SpectrumSampled b
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = a / b.amounts[i];
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::SpectrumSampled pow(vkof::SpectrumSampled a, f32 e) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = pow(a.amounts[i], e);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::Spectrum_Rgb operator/(
	f32 a,
	vkof::Spectrum_Rgb b
) {
	return vkof::Spectrum_Rgb(a / b.value);
}
PonderDifferentiable
public vkof::Spectrum_Rgb pow(vkof::Spectrum_Rgb a, f32 e) {
	return vkof::Spectrum_Rgb(pow(a.value, f32v3(e)));
}
PonderDifferentiable
public vkof::SpectrumSampled lerp(
	vkof::SpectrumSampled a,
	vkof::SpectrumSampled b,
	f32 t
) {
	var c: f32[vkof::skWavelengthSet];
	[ForceUnroll]
	for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
		c[i] = lerp(a.amounts[i], b.amounts[i], t);
	}
	return vkof::SpectrumSampled(c);
}
PonderDifferentiable
public vkof::Spectrum_Rgb lerp(
	vkof::Spectrum_Rgb a,
	vkof::Spectrum_Rgb b,
	f32 t
) {
	return vkof::Spectrum_Rgb(lerp(a.value, b.value, f32v3(t)));
}

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

func implSpectrumSampledFromRgb(
	rgb: f32v3 ,
	λ: vkof::Wavelengths
) -> vkof::SpectrumSampled {
	return vkof::SpectrumSampled.fromRgb(rgb, λ);
}
