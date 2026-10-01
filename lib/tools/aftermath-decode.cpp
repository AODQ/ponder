// tools/aftermath-decode
// Resolves shader binary hashes referenced by the dump against the
// crashes/*.spirv files vkof writes at pipeline-compile time

#include <vulkan/vulkan.h>

#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

#include <srat/core-types.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::vector<u8> read_file(std::filesystem::path const & path) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) { return {}; }
	std::streamsize const size = f.tellg();
	f.seekg(0);
	std::vector<u8> data((usize)size);
	f.read(reinterpret_cast<char *>(data.data()), size);
	return data;
}

// -----------------------------------------------------------------------------
// -- shader_lookup_cb
// -----------------------------------------------------------------------------

void shader_lookup_cb(
	GFSDK_Aftermath_ShaderBinaryHash const * const pShaderHash,
	PFN_GFSDK_Aftermath_SetData const setShaderBinary,
	void * const pUserData
) {
	auto const & crashDir = *reinterpret_cast<std::string const *>(pUserData);
	char name[32];
	snprintf(
		name, sizeof(name), "%016llx.spirv",
		(unsigned long long)pShaderHash->hash
	);
	std::vector<u8> const data = read_file(std::filesystem::path(crashDir) / name);
	if (data.empty()) {
		printf("[aftermath-decode] no shader binary for hash %s\n", name);
		return;
	}
	setShaderBinary(data.data(), (u32)data.size());
}

void shader_debug_info_lookup_cb(
	GFSDK_Aftermath_ShaderDebugInfoIdentifier const * const pIdentifier,
	PFN_GFSDK_Aftermath_SetData const setShaderDebugInfo,
	void * const pUserData
) {
	auto const & crashDir = *reinterpret_cast<std::string const *>(pUserData);
	char name[48];
	snprintf(
		name, sizeof(name), "%016llx-%016llx.nvdbg",
		(unsigned long long)pIdentifier->id[0],
		(unsigned long long)pIdentifier->id[1]
	);
	std::vector<u8> const data = read_file(std::filesystem::path(crashDir) / name);
	if (data.empty()) { return; }
	setShaderDebugInfo(data.data(), (u32)data.size());
}

} // namespace

// -----------------------------------------------------------------------------
// -- entry point
// -----------------------------------------------------------------------------

int main(int const argc, char const * const * const argv) {
	if (argc > 2 && std::string(argv[1]) == "--hash") {
		std::vector<u8> const spirv = read_file(argv[2]);
		if (spirv.empty()) {
			printf("[aftermath-decode] could not read %s\n", argv[2]);
			return 1;
		}
		GFSDK_Aftermath_SpirvCode const code = {
			.pData = spirv.data(),
			.size = (u32)spirv.size(),
		};
		GFSDK_Aftermath_ShaderBinaryHash hash = {};
		GFSDK_Aftermath_GetShaderHashSpirv(
			GFSDK_Aftermath_Version_API, &code, &hash
		);
		printf("%016llx\n", (unsigned long long)hash.hash);
		return 0;
	}

	std::string const dumpPath = (
		argc > 1 ? argv[1] : "crashes/aftermath-crash.nv-gpudmp"
	);
	std::string const crashDir = (
		std::filesystem::path(dumpPath).parent_path().string()
	);
	std::string const outPath = argc > 2 ? argv[2] : dumpPath + ".json";

	std::vector<u8> const dump = read_file(dumpPath);
	if (dump.empty()) {
		printf("[aftermath-decode] could not read %s\n", dumpPath.c_str());
		return 1;
	}

	GFSDK_Aftermath_GpuCrashDump_Decoder decoder = {};
	GFSDK_Aftermath_Result const createResult = (
		GFSDK_Aftermath_GpuCrashDump_CreateDecoder(
			GFSDK_Aftermath_Version_API,
			dump.data(), (u32)dump.size(),
			&decoder
		)
	);
	if (!GFSDK_Aftermath_SUCCEED(createResult)) {
		printf(
			"[aftermath-decode] CreateDecoder failed (0x%08x)\n",
			(unsigned)createResult
		);
		return 1;
	}

	u32 jsonSize = 0u;
	GFSDK_Aftermath_Result const jsonResult = (
		GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
			decoder,
			GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
			GFSDK_Aftermath_GpuCrashDumpFormatterFlags_NONE,
			shader_debug_info_lookup_cb,
			shader_lookup_cb,
			nullptr,
			(void *)&crashDir,
			&jsonSize
		)
	);
	if (!GFSDK_Aftermath_SUCCEED(jsonResult) || jsonSize == 0u) {
		printf(
			"[aftermath-decode] GenerateJSON failed (0x%08x)\n",
			(unsigned)jsonResult
		);
		GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder);
		return 1;
	}

	std::vector<char> json(jsonSize);
	GFSDK_Aftermath_GpuCrashDump_GetJSON(decoder, jsonSize, json.data());
	GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder);

	std::ofstream out(outPath, std::ios::binary);
	out.write(json.data(), (std::streamsize)json.size());
	printf(
		"[aftermath-decode] wrote %s (%u bytes)\n",
		outPath.c_str(), jsonSize
	);
	return 0;
}
