#!/bin/bash
cd /home/toad/repo/ponder
MODS="util-fresnel util-roughness util-mis util-random util-shading-frame util-energy-compensation \
util-material-openpbr-tables util-material-openpbr-thinfilm util-material-openpbr-microfacet \
util-material-openpbr-diffuse util-material-openpbr-metallic util-material-openpbr-coat \
util-material-openpbr-fuzz util-material-openpbr-transmission util-material-openpbr-subsurface \
util-material-openpbr util-material-openpbr-ibl util-dielectric-specular util-fog util-blackbody \
util-tonemap util-nan-probe util-environment-map util-env-prefilter \
util-mesh util-mesh-raytrace util-emitter util-regir util-regir-fill \
util-material-openpbr-load"
# hand-ported modules are not regenerated
HAND="util-vdb util-fresnel util-roughness util-mis util-random util-shading-frame util-energy-compensation util-material-openpbr-tables util-material-openpbr-shared"
for pass in 1 2 3; do
  for m in $MODS; do
    case " $HAND " in *" $m "*) continue;; esac
    [ -f lib/ponder/shaders/$m.glsl ] || continue
    python3 lib/tools/glsl2slang.py lib/ponder/shaders/$m.glsl > lib/ponder/shaders/$m.slang 2>/dev/null
  done
done
echo "--- compile status:"
FAIL=0
for m in $MODS; do
  [ -f lib/ponder/shaders/$m.slang ] || continue
  printf 'import "%s";\nstruct PC { u32v4 pad[8]; u64 outVa; };\n[[vk::push_constant]] PC pc;\n[shader("compute")]\n[numthreads(64,1,1)]\nvoid main(const u32v3 tid : SV_DispatchThreadID) { ((f32 *)pc.outVa)[tid.x] = 0.0f; }\n' "$m" > unit-test/shaders/slang/_p.slang
  printf "%-42s " "$m"
  slangc -target spirv -stage compute -entry main -matrix-layout-column-major -force-glsl-scalar-layout -Ilib/ponder/shaders -o unit-test/shaders/slang/bin/p.spv unit-test/shaders/slang/_p.slang 2>/tmp/slangport_err_$m
  N=$(grep -cE "^error" /tmp/slangport_err_$m)
  if [ "$N" = "0" ]; then echo OK; else echo "errors: $N"; FAIL=$((FAIL+1)); fi
done
rm -f unit-test/shaders/slang/bin/p.spv
echo "modules failing: $FAIL"
