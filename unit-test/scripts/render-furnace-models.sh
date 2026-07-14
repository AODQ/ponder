#!/usr/bin/env bash
# renders each furnace-tested model through the real viewer binary (not the
# vkof-test harness's furnace_model_render.comp reimplementation) and writes
# the screenshots to unit-test/output/, alongside the numeric test's own
# furnace-*.png outputs. this is a visual/end-to-end companion, not a
# replacement: it exercises resolve.comp exactly as the real app runs it
# (bluenoise seeding, antialiasing, the production shader entry point) but
# makes no assertions of its own -- test-furnace-model-render.cpp
# (unit-test/src) owns the actual NaN/firefly pass-fail check.
#
# usage: unit-test/scripts/render-furnace-models.sh [build-dir]
#   build-dir defaults to build/release (see Makefile / feedback_test_on_release)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"
build_dir="${1:-${repo_dir}/build/release}"
viewer_bin="${build_dir}/viewer"
output_dir="${repo_dir}/unit-test/output"

if [[ ! -x "${viewer_bin}" ]]; then
	echo "viewer binary not found at ${viewer_bin}" >&2
	echo "build it first: cmake --build ${build_dir} --target viewer" >&2
	exit 1
fi

mkdir -p "${output_dir}"

# model path (relative to repo_dir/assets/Models), output filename, camera
# az,el,dist,fovY -- matches test-furnace-model-render.cpp's FurnaceModelConfig
# entries so the two renders are of the same view
models=(
	"Avocado/glTF/Avocado.gltf;furnace-avocado-viewer.png;0.6,0.35,0,0.7"
	"DragonAttenuation/glTF/DragonAttenuation.gltf;furnace-dragon-attenuation-viewer.png;0.5,0.25,0,0.8"
	"DispersionTest/glTF/DispersionTest.gltf;furnace-dispersion-test-viewer.png;0.0,0.15,0,1.0"
)

status=0
for entry in "${models[@]}"; do
	IFS=';' read -r rel_path out_name camera <<< "${entry}"
	model_path="${repo_dir}/assets/Models/${rel_path}"
	out_path="${output_dir}/${out_name}"

	if [[ ! -f "${model_path}" ]]; then
		echo "skipping ${rel_path}: not found (see assets/ setup in" \
			"project_furnace_model_render_test memory / re-copy from" \
			"cull/assets)" >&2
		status=1
		continue
	fi

	echo "rendering ${rel_path} -> ${out_name}"
	"${viewer_bin}" "${model_path}" \
		--screenshot "${out_path}" \
		--camera "${camera}" \
		--resolution 1920x1080 \
		--spp 128 \
		--env-mode furnace \
		--env-intensity 0.1
done

exit "${status}"
