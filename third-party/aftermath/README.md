# NVIDIA Aftermath SDK

Download the Linux SDK from:
  https://developer.nvidia.com/nsight-aftermath

After downloading, place files here:

```
third-party/aftermath/include/GFSDK_Aftermath.h
third-party/aftermath/include/GFSDK_Aftermath_Defines.h
third-party/aftermath/include/GFSDK_Aftermath_GpuCrashDump.h
third-party/aftermath/include/GFSDK_Aftermath_GpuCrashDumpDecoding.h
third-party/aftermath/lib/libGFSDK_Aftermath_Lib.so
```

Then configure with:

```sh
cmake -B build/release -DVKOF_AFTERMATH=ON ...
```

On crash, `aftermath-crash.nv-gpudmp` is written to the working directory.
Open it with Nsight Graphics for full shader-level decoding.
