implementing material;

import types;
import util;
import model;

namespace vkof {

// backend interface for material evaluation
// doesn't apply any bsdf, instead resolving parameters into a common material
// data structure. handles resolving material constants, textures, etc
public interface I_MaterialBackend {
	associatedtype Params;
	static func resolve(
		mtrl: MaterialRef,
		inout geometry: SurfaceGeometry,
	) -> Params;
};

} // namespace vkof
