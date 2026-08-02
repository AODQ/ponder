implementing util;

namespace vkof {

// ray aabb intersection
public func intersectRayAabb(
	origin: f32v3,
	direction: f32v3,
	bounds: f32v3x2,
) -> f32v2 {
	let invDir: f32v3 = 1.0f / direction;
	let t0s: f32v3 = (bounds[0] - origin) * invDir;
	let t1s: f32v3 = (bounds[1] - origin) * invDir;
	let tsmaller: f32v3 = min(t0s, t1s);
	let tbigger: f32v3 = max(t0s, t1s);
	let tmin: f32 = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
	let tmax: f32 = min(min(tbigger.x, tbigger.y), tbigger.z);
	return f32v2(tmin, tmax);
}

public func intersectPointAabb(
	pt: f32v3,
	bounds: f32v3x2,
) -> bool {
	return (
		pt.x >= bounds[0].x && pt.x <= bounds[1].x
		&& pt.y >= bounds[0].y && pt.y <= bounds[1].y
		&& pt.z >= bounds[0].z && pt.z <= bounds[1].z
	);
}

}
