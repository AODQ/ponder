#pragma once

#include "monitor.hpp"

#include <atomic>

namespace shaderd
{
	bool stdout_is_terminal();

	// runs until the user quits or stop is set
	void tui_run(Monitor & monitor, std::atomic<bool> const & stop);
}
