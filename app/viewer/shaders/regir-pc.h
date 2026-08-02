#ifdef __cplusplus
#pragma once
#endif

#ifndef REGIR_PC_H
#define REGIR_PC_H

#include "shared/global_pc.h"

#ifndef __cplusplus
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
} pc;
#endif

#endif // REGIR_PC_H
