#ifdef VKOF_AFTERMATH

#include "aftermath.hpp"

#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <thread>
#include <vector>

static constexpr char const * skCrashDir = "crashes";

static VkDevice sDevice = VK_NULL_HANDLE;
static VkQueue sGraphicsQueue = VK_NULL_HANDLE;
static PFN_vkGetQueueCheckpointDataNV sPfnGetCheckpoints = nullptr;

// -----------------------------------------------------------------------------
// -- on_gpu_crash_dump
// -----------------------------------------------------------------------------

static void on_gpu_crash_dump(
	void const * pGpuCrashDump,
	uint32_t const gpuCrashDumpSize,
	void * pUserData
) {
	std::filesystem::create_directories(skCrashDir);
	printf(
		"[aftermath] crash dump received (%u bytes)\n",
		gpuCrashDumpSize
	);
	std::string const dumpPath = std::string(skCrashDir) + "/aftermath-crash.nv-gpudmp";
	FILE * f = fopen(dumpPath.c_str(), "wb");
	if (!f) {
		printf("[aftermath] ERROR: could not open %s\n", dumpPath.c_str());
		return;
	}
	fwrite(pGpuCrashDump, 1u, gpuCrashDumpSize, f);
	fclose(f);
	printf("[aftermath] written to %s\n", dumpPath.c_str());
	exit(0);
}

static void on_shader_debug_info(
	void const * pShaderDebugInfo,
	uint32_t const shaderDebugInfoSize,
	void * pUserData
) {
	std::filesystem::create_directories(skCrashDir);
	GFSDK_Aftermath_ShaderDebugInfoIdentifier id = {};
	GFSDK_Aftermath_GetShaderDebugInfoIdentifier(
		GFSDK_Aftermath_Version_API,
		pShaderDebugInfo,
		shaderDebugInfoSize,
		&id
	);
	char path[128];
	snprintf(
		path, sizeof(path),
		"%s/%016llx-%016llx.nvdbg",
		skCrashDir,
		(unsigned long long)id.id[0],
		(unsigned long long)id.id[1]
	);
	FILE * f = fopen(path, "wb");
	if (!f) { return; }
	fwrite(pShaderDebugInfo, 1u, shaderDebugInfoSize, f);
	fclose(f);
}

// -----------------------------------------------------------------------------
// -- on_description
// -----------------------------------------------------------------------------

static void on_description(
	PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription add,
	void * pUserData
) {
	add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "cull");
}

static void on_resolve_marker(
	void const * pMarkerData,
	uint32_t const markerDataSize,
	void * pUserData,
	PFN_GFSDK_Aftermath_ResolveMarker resolveMarker
) {
}

void vkof_aftermath_enable() {
	GFSDK_Aftermath_Result const r = GFSDK_Aftermath_EnableGpuCrashDumps(
		GFSDK_Aftermath_Version_API,
		GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
		GFSDK_Aftermath_GpuCrashDumpFeatureFlags_Default,
		on_gpu_crash_dump,
		on_shader_debug_info,
		on_description,
		on_resolve_marker,
		nullptr
	);
	if (GFSDK_Aftermath_SUCCEED(r)) {
		printf("[aftermath] GPU crash dump collection enabled\n");
	} else {
		printf(
			"[aftermath] WARN: failed to enable crash dumps (0x%08x)\n",
			(unsigned)r
		);
	}
}

// -----------------------------------------------------------------------------
// -- vkof_aftermath_disable
// -----------------------------------------------------------------------------

void vkof_aftermath_disable() {
	GFSDK_Aftermath_DisableGpuCrashDumps();
}

void vkof_aftermath_set_handles(
	VkDevice device,
	VkQueue graphics_queue,
	PFN_vkGetQueueCheckpointDataNV pfn_get_checkpoints
) {
	sDevice = device;
	sGraphicsQueue = graphics_queue;
	sPfnGetCheckpoints = pfn_get_checkpoints;
}

void vkof_aftermath_register_spirv(void const * pData, uint32_t size) {
	GFSDK_Aftermath_SpirvCode const spirv = {
		.pData = pData,
		.size = size,
	};
	GFSDK_Aftermath_ShaderBinaryHash hash = {};
	GFSDK_Aftermath_Result const r = GFSDK_Aftermath_GetShaderHashSpirv(
		GFSDK_Aftermath_Version_API,
		&spirv,
		&hash
	);
	if (!GFSDK_Aftermath_SUCCEED(r)) {
		printf("[aftermath] WARN: GetShaderHashSpirv failed (0x%08x)\n", (unsigned)r);
		return;
	}
	std::filesystem::create_directories(skCrashDir);
	char path[128];
	snprintf(path, sizeof(path), "%s/%016llx.spirv", skCrashDir, (unsigned long long)hash.hash);
	FILE * f = fopen(path, "wb");
	if (!f) {
		printf("[aftermath] WARN: could not write %s\n", path);
		return;
	}
	fwrite(pData, 1u, size, f);
	fclose(f);
	printf("[aftermath] shader binary -> %s\n", path);
}

// -----------------------------------------------------------------------------
// -- vkof_aftermath_on_device_lost
// -----------------------------------------------------------------------------

void vkof_aftermath_on_device_lost() {
	printf("[aftermath] device lost, waiting for crash dump...\n");

	GFSDK_Aftermath_CrashDump_Status status = (
		GFSDK_Aftermath_CrashDump_Status_Unknown
	);
	for (int i = 0; i < 300; ++i) {
		GFSDK_Aftermath_GetCrashDumpStatus(&status);
		if (
			status == GFSDK_Aftermath_CrashDump_Status_Finished ||
			status == GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed
		) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	if (status == GFSDK_Aftermath_CrashDump_Status_Finished) {
		printf("[aftermath] crash dump complete\n");
	} else {
		printf(
			"[aftermath] crash dump status at exit: %d\n",
			(int)status
		);
	}

	if (sPfnGetCheckpoints == nullptr || sGraphicsQueue == VK_NULL_HANDLE) {
		return;
	}

	uint32_t count = 0u;
	sPfnGetCheckpoints(sGraphicsQueue, &count, nullptr);
	if (count == 0u) { return; }

	std::vector<VkCheckpointDataNV> checkpoints(count);
	for (auto & c : checkpoints) {
		c.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
		c.pNext = nullptr;
	}
	sPfnGetCheckpoints(sGraphicsQueue, &count, checkpoints.data());

	printf("[aftermath] last %u GPU breadcrumb(s):\n", count);
	for (uint32_t i = 0u; i < count; ++i) {
		printf(
			"[aftermath]   [%u] stage=0x%08x node=%llu\n",
			i,
			(unsigned)checkpoints[i].stage,
			(unsigned long long)(uintptr_t)checkpoints[i].pCheckpointMarker
		);
	}
}

#endif // VKOF_AFTERMATH
