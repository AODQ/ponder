implementing util;

import types;

namespace vkof {

// a bounded [0, 1] reflectance color, e.g. base color
public struct ColorReflectance {
	// linear rec.709 primary, d65 white
	public f32v3 rgb;

	public __init(f32v3 rgb) {
		this.rgb = rgb;
	}

	public func toSpectrum(λ: Wavelengths) -> vkof::Spectrum {
		return vkof::Spectrum.fromRgb(rgb, λ);
	}
};

// unbounded emission / radiance, e.g. light source
public struct ColorEmission {
	f32v3 rgb;
	f32 luminance;

	public __init(f32v3 rgb, f32 luminance) {
		this.rgb = rgb;
		this.luminance = luminance;
	}

	public func toSpectrum(λ: Wavelengths) -> vkof::Spectrum {
		return (
			vkof::Spectrum.fromRgbUnbounded(rgb, λ)
			* vkof::Spectrum.illuminantD65(λ)
			* luminance
		);
	}
};

} // namespace vkof
