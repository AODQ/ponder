implementing material;


namespace vkof {

// resolves material properties for OpenPBR materials; colors are uplifted
// once here, so not an I_MaterialBackend as resolve needs λ
public struct MaterialBackend_OpenPbr {
	public struct Params {
		public f32 baseWeight;
		public vkof::Spectrum baseColor;
		public f32 baseMetalness;
		public f32 baseDiffuseRoughness;
		public f32 specularWeight;
		public vkof::Spectrum specularColor;
		public f32 specularRoughness;
		public f32 specularRoughnessAnisotropy;
		// not part of openpbr (which rotates via the geometry tangent instead);
		// gltf khr_materials_anisotropy rotation, radians counter-clockwise
		// from the tangent
		public f32 specularRoughnessAnisotropyRotation;
		public f32 specularIor;
		public f32 transmissionWeight;
		public vkof::Spectrum transmissionColor;
		public f32 transmissionDepth;
		public f32v3 transmissionScatter;
		public f32 transmissionScatterAnisotropy;
		public f32 transmissionDispersionScale;
		public f32 transmissionDispersionAbbeNumber;
		public f32 subsurfaceWeight;
		public vkof::Spectrum subsurfaceColor;
		public f32 subsurfaceRadius;
		public f32v3 subsurfaceRadiusScale;
		public f32 subsurfaceScatterAnisotropy;
		public f32 coatWeight;
		public vkof::Spectrum coatColor;
		public f32 coatRoughness;
		public f32 coatRoughnessAnisotropy;
		public f32 coatIor;
		public f32 coatDarkening;
		public f32 fuzzWeight;
		public vkof::Spectrum fuzzColor;
		public f32 fuzzRoughness;
		public f32 thinFilmWeight;
		public f32 thinFilmThickness;
		public f32 thinFilmIor;
		// public f32 geometryOpacity;
		public f32 geometryThinWalled;

		// tangent-space, unnormalized; +z without a normal map
		public f32v3 geometryNormal;
		public f32v3 geometryCoatNormal;
		// f32v4 geometryTangentTexture;
		// f32v4 geometryCoatTangentTexture;

		// // -- glTF extras
		// f32 alphaCutoff;
	};

	public static func resolve(
		mtrl: MaterialRef,
		inout geometry: SurfaceGeometry,
		λ: Wavelengths,
	) -> Params {
		Params params;
		#define Evaluate( field ) \
			params.field = ( \
				materialComponentEvaluate( \
					/*mtrl=*/ mtrl, \
					/*component=*/ &mtrl.material.field, \
					/*uv=*/ geometry.uv, \
					/*lod=*/ geometry.textureLod \
				) \
			) \
		/**/
		#define EvaluateSpectrum( field ) \
			params.field = ( \
				vkof::Spectrum.fromRgb( \
					materialComponentEvaluate( \
						/*mtrl=*/ mtrl, \
						/*component=*/ &mtrl.material.field, \
						/*uv=*/ geometry.uv, \
						/*lod=*/ geometry.textureLod \
					), \
					λ \
				) \
			) \
		/**/

		// -- evaluate all openpbr material properties
		Evaluate( baseWeight );
		EvaluateSpectrum( baseColor );
		Evaluate( baseMetalness );
		Evaluate( baseDiffuseRoughness );
		Evaluate( specularWeight );
		EvaluateSpectrum( specularColor );
		Evaluate( specularRoughness );
		Evaluate( specularRoughnessAnisotropy );
		Evaluate( specularRoughnessAnisotropyRotation );
		Evaluate( specularIor );
		Evaluate( transmissionWeight );
		EvaluateSpectrum( transmissionColor );
		Evaluate( transmissionDepth );
		Evaluate( transmissionScatter );
		Evaluate( transmissionScatterAnisotropy );
		Evaluate( transmissionDispersionScale );
		Evaluate( transmissionDispersionAbbeNumber );
		Evaluate( subsurfaceWeight );
		EvaluateSpectrum( subsurfaceColor );
		Evaluate( subsurfaceRadius );
		Evaluate( subsurfaceRadiusScale );
		Evaluate( subsurfaceScatterAnisotropy );
		Evaluate( coatWeight );
		EvaluateSpectrum( coatColor );
		Evaluate( coatRoughness );
		Evaluate( coatRoughnessAnisotropy );
		Evaluate( coatIor );
		Evaluate( coatDarkening );
		Evaluate( fuzzWeight );
		EvaluateSpectrum( fuzzColor );
		Evaluate( fuzzRoughness );
		Evaluate( thinFilmWeight );
		Evaluate( thinFilmThickness );
		Evaluate( thinFilmIor );
		// Evaluate( geometryOpacity );
		Evaluate( geometryThinWalled );
		params.geometryNormal = (
			materialNormalEvaluate(
				/*mtrl=*/ mtrl,
				/*texture=*/ mtrl.material.geometryNormalTexture,
				/*uvTransform=*/ mtrl.material.geometryNormalUvTransform,
				/*uv=*/ geometry.uv,
				/*lod=*/ geometry.textureLod
			)
		);
		params.geometryCoatNormal = (
			materialNormalEvaluate(
				/*mtrl=*/ mtrl,
				/*texture=*/ mtrl.material.geometryCoatNormalTexture,
				/*uvTransform=*/ mtrl.material.geometryCoatNormalUvTransform,
				/*uv=*/ geometry.uv,
				/*lod=*/ geometry.textureLod
			)
		);
		#undef Evaluate
		#undef EvaluateSpectrum

		// -- apply roughness floor
		let skMinRoughness = 1e-3f;
		params.specularRoughness = max(params.specularRoughness, skMinRoughness);
		params.coatRoughness = max(params.coatRoughness, skMinRoughness);
		params.fuzzRoughness = max(params.fuzzRoughness, skMinRoughness);

		return params;
	}
};

} // namespace vkof
