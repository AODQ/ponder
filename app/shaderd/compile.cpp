#include "compile.hpp"

#include <slang.h>
#include <slang-com-ptr.h>

#include <map>
#include <mutex>
#include <unordered_set>

namespace fs = std::filesystem;
namespace sc = vkof::shader_cache;

using Slang::ComPtr;

namespace
{
	struct CachedModule {
		std::string name;
		std::string path;
		ComPtr<slang::IBlob> blob;
		std::vector<sc::Dependency> dependencies;
	};

	// load order of one define set, dependencies before importers
	struct VariantModules {
		std::vector<CachedModule> modules;
	};
}

struct shaderd::ModuleCache {
	std::mutex mutex;
	std::map<std::string, VariantModules> variants;
};

void shaderd::ModuleCacheDeleter::operator()(ModuleCache * const cache) const {
	delete cache;
}

shaderd::ModuleCachePtr shaderd::module_cache_create() {
	return ModuleCachePtr(new ModuleCache {});
}

// -----------------------------------------------------------------------------
// -- global sessions
// -----------------------------------------------------------------------------

// global sessions aren't thread-safe; each compile checks one out
static std::mutex sGlobalSessionMutex;
static std::vector<ComPtr<slang::IGlobalSession>> sGlobalSessions;

static ComPtr<slang::IGlobalSession> global_session_acquire() {
	{
		std::lock_guard<std::mutex> const lock(sGlobalSessionMutex);
		if (!sGlobalSessions.empty()) {
			ComPtr<slang::IGlobalSession> session = sGlobalSessions.back();
			sGlobalSessions.pop_back();
			return session;
		}
	}
	ComPtr<slang::IGlobalSession> session;
	slang::createGlobalSession(session.writeRef());
	return session;
}

static void global_session_release(ComPtr<slang::IGlobalSession> session) {
	std::lock_guard<std::mutex> const lock(sGlobalSessionMutex);
	sGlobalSessions.emplace_back(session);
}

u32 shaderd::global_session_pool_trim(u32 const keep) {
	std::vector<ComPtr<slang::IGlobalSession>> released;
	{
		std::lock_guard<std::mutex> const lock(sGlobalSessionMutex);
		while (sGlobalSessions.size() > keep) {
			released.emplace_back(std::move(sGlobalSessions.back()));
			sGlobalSessions.pop_back();
		}
	}
	return (u32)released.size();
}

std::string shaderd::slang_version() {
	ComPtr<slang::IGlobalSession> const session = global_session_acquire();
	std::string const version = session->getBuildTagString();
	global_session_release(session);
	return version;
}

// -----------------------------------------------------------------------------
// -- helpers
// -----------------------------------------------------------------------------

std::string shaderd::dependency_path(
	fs::path const & root,
	fs::path const & path
) {
	std::error_code ec;
	fs::path const canonical = fs::weakly_canonical(path, ec);
	fs::path const relative = canonical.lexically_relative(root);
	if (relative.empty() || *relative.begin() == "..") {
		return canonical.generic_string();
	}
	return relative.generic_string();
}

static u64 snapshot_hash(
	fs::path const & root,
	shaderd::HashSnapshot const & snapshot,
	std::string const & path
) {
	auto const it = snapshot.find(path);
	return it != snapshot.end() ? it->second : sc::file_hash(root / path);
}

static bool dependencies_clean(
	fs::path const & root,
	shaderd::HashSnapshot const & snapshot,
	std::vector<sc::Dependency> const & dependencies
) {
	for (sc::Dependency const & dependency : dependencies) {
		if (snapshot_hash(root, snapshot, dependency.path) != dependency.hash) {
			return false;
		}
	}
	return true;
}

static std::string blob_string(slang::IBlob * const blob) {
	if (!blob) { return {}; }
	return std::string(
		(char const *)blob->getBufferPointer(), blob->getBufferSize()
	);
}

static std::string variant_modules_key(sc::Variant const & variant) {
	std::string key = variant.optimize ? "O1;" : "O0;";
	for (std::string const & define : variant.defines) {
		key += define + ";";
	}
	return key;
}

static ComPtr<slang::ISession> session_create(
	slang::IGlobalSession * const global,
	fs::path const & root,
	sc::Variant const & variant
) {
	// -- macros, NAME or NAME=VALUE
	std::vector<std::string> names;
	std::vector<std::string> values;
	for (std::string const & define : variant.defines) {
		size_t const equals = define.find('=');
		names.emplace_back(define.substr(0u, equals));
		values.emplace_back(
			equals == std::string::npos ? "1" : define.substr(equals + 1u)
		);
	}
	std::vector<slang::PreprocessorMacroDesc> macros;
	for (size_t i = 0u; i < names.size(); ++i) {
		macros.emplace_back(slang::PreprocessorMacroDesc {
			.name = names[i].c_str(),
			.value = values[i].c_str(),
		});
	}

	// -- matches vkof's slangc flags
	slang::CompilerOptionEntry options[2] = {};
	options[0].name = slang::CompilerOptionName::Optimization;
	options[0].value.intValue0 = (
		variant.optimize
		? SLANG_OPTIMIZATION_LEVEL_DEFAULT
		: SLANG_OPTIMIZATION_LEVEL_NONE
	);
	options[1].name = slang::CompilerOptionName::GLSLForceScalarLayout;
	options[1].value.intValue0 = 1;

	slang::TargetDesc target = {};
	target.format = SLANG_SPIRV;
	target.forceGLSLScalarBufferLayout = true;

	std::string const searchPath = root.string();
	char const * const searchPaths[] = { searchPath.c_str() };

	slang::SessionDesc desc = {};
	desc.targets = &target;
	desc.targetCount = 1;
	desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
	desc.searchPaths = searchPaths;
	desc.searchPathCount = 1;
	desc.preprocessorMacros = macros.data();
	desc.preprocessorMacroCount = (SlangInt)macros.size();
	desc.compilerOptionEntries = options;
	desc.compilerOptionEntryCount = 2u;

	ComPtr<slang::ISession> session;
	global->createSession(desc, session.writeRef());
	return session;
}

// -----------------------------------------------------------------------------
// -- compile
// -----------------------------------------------------------------------------

static shaderd::CompileResult compile_with(
	slang::IGlobalSession * const global,
	fs::path const & root,
	sc::Variant const & variant,
	shaderd::HashSnapshot const & snapshot,
	shaderd::ModuleCache & cache
) {
	shaderd::CompileResult result {
		.ok = false,
		.diagnostics = {},
		.spirv = {},
		.dependencies = {},
		.modulesReused = 0u,
		.modulesCompiled = 0u,
	};
	ComPtr<slang::ISession> const session = (
		session_create(global, root, variant)
	);
	if (!session) {
		result.diagnostics = "failed to create slang session";
		return result;
	}
	std::string const modulesKey = variant_modules_key(variant);

	// -- reuse the modules whose files haven't changed
	std::vector<CachedModule> reusable;
	{
		std::lock_guard<std::mutex> const lock(cache.mutex);
		for (CachedModule const & module : cache.variants[modulesKey].modules) {
			if (dependencies_clean(root, snapshot, module.dependencies)) {
				reusable.emplace_back(module);
			}
		}
	}
	std::unordered_set<std::string> reused;
	for (CachedModule const & module : reusable) {
		ComPtr<slang::IBlob> diagnostics;
		slang::IModule * const loaded = session->loadModuleFromIRBlob(
			module.name.c_str(),
			module.path.c_str(),
			module.blob,
			diagnostics.writeRef()
		);
		if (!loaded) { break; }
		reused.insert(module.name);
	}
	result.modulesReused = (u32)reused.size();

	// -- entry from source
	fs::path const entryPath = root / variant.entry;
	ComPtr<slang::IBlob> diagnostics;
	slang::IModule * const entryModule = session->loadModule(
		entryPath.string().c_str(), diagnostics.writeRef()
	);
	result.diagnostics += blob_string(diagnostics);
	if (!entryModule) { return result; }

	ComPtr<slang::IEntryPoint> entryPoint;
	diagnostics.setNull();
	entryModule->findAndCheckEntryPoint(
		"main", SLANG_STAGE_COMPUTE, entryPoint.writeRef(), diagnostics.writeRef()
	);
	result.diagnostics += blob_string(diagnostics);
	if (!entryPoint) {
		result.diagnostics += "no compute entry point 'main'\n";
		return result;
	}

	// -- link and generate spir-v
	slang::IComponentType * const components[] = { entryModule, entryPoint };
	ComPtr<slang::IComponentType> composite;
	ComPtr<slang::IComponentType> linked;
	ComPtr<slang::IBlob> code;
	diagnostics.setNull();
	session->createCompositeComponentType(
		components, 2, composite.writeRef(), diagnostics.writeRef()
	);
	result.diagnostics += blob_string(diagnostics);
	if (composite) {
		diagnostics.setNull();
		composite->link(linked.writeRef(), diagnostics.writeRef());
		result.diagnostics += blob_string(diagnostics);
	}
	if (linked) {
		diagnostics.setNull();
		linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
		result.diagnostics += blob_string(diagnostics);
	}

	// -- dependencies, hashed as of the batch's snapshot
	for (int i = 0; i < entryModule->getDependencyFileCount(); ++i) {
		std::string const path = shaderd::dependency_path(
			root, entryModule->getDependencyFilePath(i)
		);
		result.dependencies.emplace_back(sc::Dependency {
			.path = path,
			.hash = snapshot_hash(root, snapshot, path),
		});
	}
	if (!code) { return result; }
	result.ok = true;
	result.spirv = blob_string(code);

	// -- keep the newly compiled modules for later compiles
	std::string const entryName = entryModule->getName();
	std::vector<CachedModule> modules;
	for (SlangInt i = 0; i < session->getLoadedModuleCount(); ++i) {
		slang::IModule * const module = session->getLoadedModule(i);
		std::string const name = module->getName();
		if (name == entryName) { continue; }
		CachedModule cached {
			.name = name,
			.path = module->getFilePath() ? module->getFilePath() : "",
			.blob = {},
			.dependencies = {},
		};
		if (reused.contains(name)) {
			for (CachedModule const & r : reusable) {
				if (r.name == name) { cached = r; }
			}
			modules.emplace_back(cached);
			continue;
		}
		if (SLANG_FAILED(module->serialize(cached.blob.writeRef()))) {
			continue;
		}
		for (int j = 0; j < module->getDependencyFileCount(); ++j) {
			std::string const path = shaderd::dependency_path(
				root, module->getDependencyFilePath(j)
			);
			cached.dependencies.emplace_back(sc::Dependency {
				.path = path,
				.hash = snapshot_hash(root, snapshot, path),
			});
		}
		modules.emplace_back(cached);
		++result.modulesCompiled;
	}
	{
		std::lock_guard<std::mutex> const lock(cache.mutex);
		cache.variants[modulesKey].modules = std::move(modules);
	}
	return result;
}

shaderd::CompileResult shaderd::compile(
	fs::path const & root,
	sc::Variant const & variant,
	HashSnapshot const & snapshot,
	ModuleCache & cache
) {
	ComPtr<slang::IGlobalSession> const global = global_session_acquire();
	CompileResult result = compile_with(global, root, variant, snapshot, cache);
	global_session_release(global);
	return result;
}
