.PHONY: release debug install install-debug release-optimize install-opt clean

release:
	cmake -G "Ninja" -B build/release -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=True -DCMAKE_INSTALL_PREFIX=install/release
	cmake --build build/release
	cmake --install build/release

debug:
	cmake -G "Ninja" -B build/debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=True -DCMAKE_INSTALL_PREFIX=install/debug
	cmake --build build/debug
	cmake --install build/debug

release-optimize:
	cmake -G "Ninja" -B build/release-optimize -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=True -DCMAKE_INSTALL_PREFIX=install/release-optimize
	cmake --build build/release-optimize
	cmake --install build/release-optimize

clean:

	rm -rf build/ install/
