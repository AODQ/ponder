implementing medium;

import types;
import util;

namespace vkof {

public static const u32 skMediumStackCapacity = 4u;

// the media a path is inside, for nested volumes
public struct MediumStack {
	// the ambient medium is always at the bottom
	MediumHandle handles[skMediumStackCapacity];
	u32 count;

	public __init() {
		[ForceUnroll]
		for (u32 i = 0u; i < skMediumStackCapacity; ++i) {
			handles[i] = MediumHandle::ambient();
		}
		count = 1u;
	}

	// the medium the path is in based on priority in th stack
	public func current() -> MediumHandle {
		var best: MediumHandle = MediumHandle::ambient();
		var bestPriority: u32 = 0u;
		for (u32 i = 0u; i < count; ++i) {
			let priority: u32 = mediumPriority(handles[i]);
			if (priority >= bestPriority) {
				best = handles[i];
				bestPriority = priority;
			}
		}
		return best;
	}

	// returns true if the medium is in the stack
	public func contains(handle: MediumHandle) -> bool {
		for (u32 i = 0u; i < count; ++i) {
			if (handles[i].equals(handle)) { return true; }
		}
		return false;
	}

	// -- boundary events

	// path is entering a medium
	// ignored if the medium is already in the stack or the stack is full
	[mutating] public func enter(handle: MediumHandle) -> void {
		if (contains(handle) || count >= skMediumStackCapacity) { return; }
		handles[count] = handle;
		count += 1u;
	}

	// path is exiting a medium
	// ignored if the medium is not in the stack
	[mutating] public func exit(handle: MediumHandle) -> void {
		// -- find the latest entry of the medium, never the ambient one
		var found: u32 = 0u;
		for (u32 i = 1u; i < count; ++i) {
			if (handles[i].equals(handle)) { found = i; }
		}
		if (found == 0u) { return; }
		// -- close the gap
		for (u32 i = found; i + 1u < count; ++i) {
			handles[i] = handles[i + 1u];
		}
		count -= 1u;
	}
};

} // namespace vkof
