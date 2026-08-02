implementing material;

import types;
import util;
import model;

namespace vkof {

// only resolves diffuse material properties
public struct MaterialBackend_Diffuse : I_MaterialBackend {
	public struct Params {
		__init(ColorReflectance albedo) {
			this.albedo = albedo;
		}
		public ColorReflectance albedo;
	};

	public static func resolve(
		mtrl: MaterialRef,
		inout geometry: SurfaceGeometry,
	) -> Params {
		let baseColor: f32v3 = (
			materialComponentEvaluate(
				/*mtrl=*/ mtrl,
				/*component=*/ &mtrl.material.baseColor,
				/*uv=*/ geometry.uv,
				/*lod=*/ geometry.textureLod
			)
		);
		let baseWeight: f32 = (
			materialComponentEvaluate(
				/*mtrl=*/ mtrl,
				/*component=*/ &mtrl.material.baseWeight,
				/*uv=*/ geometry.uv,
				/*lod=*/ geometry.textureLod
			)
		);
		return Params(/*albedo=*/ ColorReflectance(baseColor * baseWeight));
	}
};

} // namespace vkof
