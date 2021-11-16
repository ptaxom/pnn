#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_variables)]

use std::{
    error::Error,
    fmt,
    self,
};
use core::fmt::Debug;
use std::{os::raw::{c_void, }};

#[derive(Debug)]
pub enum cudaError{
    InvalidValue,
    MemoryAllocation,
    InitializationError,
    CudartUnloading,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    InvalidConfiguration,
    InvalidPitchValue,
    InvalidSymbol,
    InvalidHostPointer,
    InvalidDevicePointer,
    InvalidTexture,
    InvalidTextureBinding,
    InvalidChannelDescriptor,
    InvalidMemcpyDirection,
    AddressOfConstant,
    TextureFetchFailed,
    TextureNotBound,
    SynchronizationError,
    InvalidFilterSetting,
    InvalidNormSetting,
    MixedDeviceExecution,
    NotYetImplemented,
    MemoryValueTooLarge,
    StubLibrary,
    InsufficientDriver,
    CallRequiresNewerDriver,
    InvalidSurface,
    DuplicateVariableName,
    DuplicateTextureName,
    DuplicateSurfaceName,
    DevicesUnavailable,
    IncompatibleDriverContext,
    MissingConfiguration,
    PriorLaunchFailure,
    LaunchMaxDepthExceeded,
    LaunchFileScopedTex,
    LaunchFileScopedSurf,
    SyncDepthExceeded,
    LaunchPendingCountExceeded,
    InvalidDeviceFunction,
    NoDevice,
    InvalidDevice,
    DeviceNotLicensed,
    SoftwareValidityNotEstablished,
    StartupFailure,
    InvalidKernelImage,
    DeviceUninitialized,
    MapBufferObjectFailed,
    UnmapBufferObjectFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoKernelImageForDevice,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    ECCUncorrectable,
    UnsupportedLimit,
    DeviceAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    JitCompilerNotFound,
    UnsupportedPtxVersion,
    JitCompilationDisabled,
    UnsupportedExecAffinity,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidResourceHandle,
    IllegalState,
    SymbolNotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    SetOnActiveProcess,
    ContextIsDestroyed,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    LaunchFailure,
    CooperativeLaunchTooLarge,
    NotPermitted,
    NotSupported,
    SystemNotReady,
    SystemDriverMismatch,
    CompatNotSupportedOnDevice,
    MpsConnectionFailed,
    MpsRpcFailure,
    MpsServerNotReady,
    MpsMaxClientsReached,
    MpsMaxConnectionsReached,
    StreamCaptureUnsupported,
    StreamCaptureInvalidated,
    StreamCaptureMerge,
    StreamCaptureUnmatched,
    StreamCaptureUnjoined,
    StreamCaptureIsolation,
    StreamCaptureImplicit,
    CapturedEvent,
    StreamCaptureWrongThread,
    Timeout,
    GraphExecUpdateFailure,
    ExternalDevice,
    Unknown,
    ApiFailureBase
}


impl fmt::Display for cudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_desc = match self {
            cudaError::InvalidValue => "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.",
            cudaError::MemoryAllocation => "The API call failed because it was unable to allocate enough memory to perform the requested operation.",
            cudaError::InitializationError => "The API call failed because the CUDA driver and runtime could not be initialized.",
            cudaError::CudartUnloading => "This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a pointin time after CUDA driver has been unloaded.",
            cudaError::ProfilerDisabled => "This indicates profiler is not initialized for this run. This can happen when the application is running with external profilingtools like visual profiler.",
            cudaError::ProfilerNotInitialized => "Deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via cudaProfilerStart or cudaProfilerStop without initialization.",
            cudaError::ProfilerAlreadyStarted => "Deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStart() when profiling is already enabled.",
            cudaError::ProfilerAlreadyStopped => "Deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStop() when profiling is already disabled.",
            cudaError::InvalidConfiguration => "This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requestingmore shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks.See cudaDeviceProp for more device limitations.",
            cudaError::InvalidPitchValue => "This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable rangefor pitch.",
            cudaError::InvalidSymbol => "This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.",
            cudaError::InvalidHostPointer => "Deprecated This error return is deprecated as of CUDA 10.1.  This indicates that at least one host pointer passed to the API call is not a valid host pointer.",
            cudaError::InvalidDevicePointer => "Deprecated This error return is deprecated as of CUDA 10.1.  This indicates that at least one device pointer passed to the API call is not a valid device pointer.",
            cudaError::InvalidTexture => "This indicates that the texture passed to the API call is not a valid texture.",
            cudaError::InvalidTextureBinding => "This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture.",
            cudaError::InvalidChannelDescriptor => "This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of theformats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.",
            cudaError::InvalidMemcpyDirection => "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind.",
            cudaError::AddressOfConstant => "Deprecated This error return is deprecated as of CUDA 3.1. Variables in constant memory may now have their address taken by the runtime      via cudaGetSymbolAddress().  This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release.",
            cudaError::TextureFetchFailed => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture   operations.",
            cudaError::TextureNotBound => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations.",
            cudaError::SynchronizationError => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a synchronization operation had failed. This was previously used for some device emulation functions.",
            cudaError::InvalidFilterSetting => "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.",
            cudaError::InvalidNormSetting => "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.",
            cudaError::MixedDeviceExecution => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  Mixing of device and device emulation code was not allowed.",
            cudaError::NotYetImplemented => "Deprecated This error return is deprecated as of CUDA 4.1.  This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error.",
            cudaError::MemoryValueTooLarge => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that an emulated device pointer exceeded the 32-bit address range.",
            cudaError::StubLibrary => "This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stubrather than a real driver loaded will result in CUDA API returning this error.",
            cudaError::InsufficientDriver => "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration.Users should install an updated NVIDIA display driver to allow the application to run.",
            cudaError::CallRequiresNewerDriver => "This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updatedNVIDIA CUDA driver to allow the API call to succeed.",
            cudaError::InvalidSurface => "This indicates that the surface passed to the API call is not a valid surface.",
            cudaError::DuplicateVariableName => "This indicates that multiple global or constant variables (across separate CUDA source files in the application) share thesame string name.",
            cudaError::DuplicateTextureName => "This indicates that multiple textures (across separate CUDA source files in the application) share the same string name.",
            cudaError::DuplicateSurfaceName => "This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name.",
            cudaError::DevicesUnavailable => "This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due touse of cudaComputeModeExclusive, cudaComputeModeProhibited or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailabledue to memory constraints on a device that already has active CUDA work being performed.",
            cudaError::IncompatibleDriverContext => "This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are usingCUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver contextmay be incompatible either because the Driver context was created using an older version of the API, because the Runtime APIcall expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed.Please see Interactions with the CUDA Driver API for more information.",
            cudaError::MissingConfiguration => "The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the cudaConfigureCall() function.",
            cudaError::PriorLaunchFailure => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches.",
            cudaError::LaunchMaxDepthExceeded => "This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed themaximum supported number of nested grid launches.",
            cudaError::LaunchFileScopedTex => "This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported bythe device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.",
            cudaError::LaunchFileScopedSurf => "This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported bythe device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.",
            cudaError::SyncDepthExceeded => "This error indicates that a call to cudaDeviceSynchronize made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levelsof grids) or user specified device limit cudaLimitDevRuntimeSyncDepth. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which cudaDeviceSynchronize will be called must be specified with the cudaLimitDevRuntimeSyncDepth limit to the cudaDeviceSetLimit api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth requirethe runtime to reserve large amounts of device memory that cannot be used for user allocations.",
            cudaError::LaunchPendingCountExceeded => "This error indicates that a device runtime grid launch failed because the launch would exceed the limit cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed successfully, cudaDeviceSetLimit must be called to set the cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raisingthe limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for userallocations.",
            cudaError::InvalidDeviceFunction => "The requested device function does not exist or is not compiled for the proper device architecture.",
            cudaError::NoDevice => "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.",
            cudaError::InvalidDevice => "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the actionrequested is invalid for the specified device.",
            cudaError::DeviceNotLicensed => "This indicates that the device doesn't have a valid Grid License.",
            cudaError::SoftwareValidityNotEstablished => "By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validityof both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validityof either the runtime or the driver could not be established.",
            cudaError::StartupFailure => "This indicates an internal startup failure in the CUDA runtime.",
            cudaError::InvalidKernelImage => "This indicates that the device kernel image is invalid.",
            cudaError::DeviceUninitialized => "This most frequently indicates that there is no context bound to the current thread. This can also be returned if the contextpassed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls).See cuCtxGetApiVersion() for more details.",
            cudaError::MapBufferObjectFailed => "This indicates that the buffer object could not be mapped.",
            cudaError::UnmapBufferObjectFailed => "This indicates that the buffer object could not be unmapped.",
            cudaError::ArrayIsMapped => "This indicates that the specified array is currently mapped and thus cannot be destroyed.",
            cudaError::AlreadyMapped => "This indicates that the resource is already mapped.",
            cudaError::NoKernelImageForDevice => "This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifiescode generation options for a particular CUDA source file that do not include the corresponding device configuration.",
            cudaError::AlreadyAcquired => "This indicates that a resource has already been acquired.",
            cudaError::NotMapped => "This indicates that a resource is not mapped.",
            cudaError::NotMappedAsArray => "This indicates that a mapped resource is not available for access as an array.",
            cudaError::NotMappedAsPointer => "This indicates that a mapped resource is not available for access as a pointer.",
            cudaError::ECCUncorrectable => "This indicates that an uncorrectable ECC error was detected during execution.",
            cudaError::UnsupportedLimit => "This indicates that the cudaLimit passed to the API call is not supported by the active device.",
            cudaError::DeviceAlreadyInUse => "This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.",
            cudaError::PeerAccessUnsupported => "This error indicates that P2P access is not supported across the given devices.",
            cudaError::InvalidPtx => "A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binaryfor the current device.",
            cudaError::InvalidGraphicsContext => "This indicates an error with the OpenGL or DirectX context.",
            cudaError::NvlinkUncorrectable => "This indicates that an uncorrectable NVLink error was detected during the execution.",
            cudaError::JitCompilerNotFound => "This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. Theruntime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.",
            cudaError::UnsupportedPtxVersion => "This indicates that the provided PTX was compiled with an unsupported toolchain. The most common reason for this, is the PTXwas generated by a compiler newer than what is supported by the CUDA driver and PTX JIT compiler.",
            cudaError::JitCompilationDisabled => "This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compilingPTX if an application does not contain a suitable binary for the current device.",
            cudaError::UnsupportedExecAffinity => "This indicates that the provided execution affinity is not supported by the device.",
            cudaError::InvalidSource => "This indicates that the device kernel source is invalid.",
            cudaError::FileNotFound => "This indicates that the file specified was not found.",
            cudaError::SharedObjectSymbolNotFound => "This indicates that a link to a shared object failed to resolve.",
            cudaError::SharedObjectInitFailed => "This indicates that initialization of a shared object failed.",
            cudaError::OperatingSystem => "This error indicates that an OS call failed.",
            cudaError::InvalidResourceHandle => "This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t.",
            cudaError::IllegalState => "This indicates that a resource required by the API call is not in a valid state to perform the requested operation.",
            cudaError::SymbolNotFound => "This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver functionnames, texture names, and surface names.",
            cudaError::NotReady => "This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error,but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery().",
            cudaError::IllegalAddress => "The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistentstate and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.",
            cudaError::LaunchOutOfResources => "This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar tocudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernellaunch specifies too many threads for the kernel's register count.",
            cudaError::LaunchTimeout => "This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the deviceproperty kernelExecTimeoutEnabled for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error.To continue using CUDA, the process must be terminated and relaunched.",
            cudaError::LaunchIncompatibleTexturing => "This error indicates a kernel launch that uses an incompatible texturing mode.",
            cudaError::PeerAccessAlreadyEnabled => "This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.",
            cudaError::PeerAccessNotEnabled => "This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().",
            cudaError::SetOnActiveProcess => "This indicates that the user has called cudaSetValidDevices(), cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(), or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernelsare examples of non-device management operations). This error can also be returned if using runtime/driver interoperabilityand there is an existing CUcontext active on the host thread.",
            cudaError::ContextIsDestroyed => "This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.",
            cudaError::Assert => "An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations areinvalid. To continue using CUDA, the process must be terminated and relaunched.",
            cudaError::TooManyPeers => "This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of thedevices passed to cudaEnablePeerAccess().",
            cudaError::HostMemoryAlreadyRegistered => "This error indicates that the memory range passed to cudaHostRegister() has already been registered.",
            cudaError::HostMemoryNotRegistered => "This error indicates that the pointer passed to cudaHostUnregister() does not correspond to any currently registered memory region.",
            cudaError::HardwareStackError => "Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stacksize limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continueusing CUDA, the process must be terminated and relaunched.",
            cudaError::IllegalInstruction => "The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state andany further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.",
            cudaError::MisalignedAddress => "The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in aninconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminatedand relaunched.",
            cudaError::InvalidAddressSpace => "While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain addressspaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leavesthe process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the processmust be terminated and relaunched.",
            cudaError::InvalidPc => "The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA workwill return the same error. To continue using CUDA, the process must be terminated and relaunched.",
            cudaError::LaunchFailure => "An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointerand accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases canbe found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work willreturn the same error. To continue using CUDA, the process must be terminated and relaunched.",
            cudaError::CooperativeLaunchTooLarge => "This error indicates that the number of blocks launched per grid for a kernel that was launched via either cudaLaunchCooperativeKernel or cudaLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cudaOccupancyMaxActiveBlocksPerMultiprocessor or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount.",
            cudaError::NotPermitted => "This error indicates the attempted operation is not permitted.",
            cudaError::NotSupported => "This error indicates the attempted operation is not supported on the current system or device.",
            cudaError::SystemNotReady => "This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configurationis in a valid state and all required driver daemons are actively running. More information about this error can be found inthe system specific user guide.",
            cudaError::SystemDriverMismatch => "This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to thecompatibility documentation for supported versions.",
            cudaError::CompatNotSupportedOnDevice => "This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDAdoes not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensurethat only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.",
            cudaError::MpsConnectionFailed => "This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.",
            cudaError::MpsRpcFailure => "This error indicates that the remote procedural call between the MPS server and the MPS client failed.",
            cudaError::MpsServerNotReady => "This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when theMPS server is in the process of recovering from a fatal failure.",
            cudaError::MpsMaxClientsReached => "This error indicates that the hardware resources required to create MPS client have been exhausted.",
            cudaError::MpsMaxConnectionsReached => "This error indicates the the hardware resources required to device connections have been exhausted.",
            cudaError::StreamCaptureUnsupported => "The operation is not permitted when the stream is capturing.",
            cudaError::StreamCaptureInvalidated => "The current capture sequence on the stream has been invalidated due to a previous error.",
            cudaError::StreamCaptureMerge => "The operation would have resulted in a merge of two independent capture sequences.",
            cudaError::StreamCaptureUnmatched => "The capture was not initiated in this stream.",
            cudaError::StreamCaptureUnjoined => "The capture sequence contains a fork that was not joined to the primary stream.",
            cudaError::StreamCaptureIsolation => "A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependenciesare allowed to cross the boundary.",
            cudaError::StreamCaptureImplicit => "The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.",
            cudaError::CapturedEvent => "The operation is not permitted on an event which was last recorded in a capturing stream.",
            cudaError::StreamCaptureWrongThread => "A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture in a different thread.",
            cudaError:: Timeout => "This indicates that the wait operation has timed out.",
            cudaError::GraphExecUpdateFailure => "This error indicates that the graph update was not performed because it included changes which violated constraints specificto instantiated graph update.",
            cudaError::ExternalDevice => "This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device'ssignal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption.This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA,the process must be terminated and relaunched.",
            cudaError::Unknown => "This indicates that an unknown internal error has occurred.",
            cudaError::ApiFailureBase => "Deprecated This error return is deprecated as of CUDA 4.1.  Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not   return such errors.",};
        write!(f, "{}", err_desc)
    }
}

impl From<u32> for cudaError {
    fn from(error: u32) -> Self {
        match error {
            1 => cudaError::InvalidValue,
            2 => cudaError::MemoryAllocation,
            3 => cudaError::InitializationError,
            4 => cudaError::CudartUnloading,
            5 => cudaError::ProfilerDisabled,
            6 => cudaError::ProfilerNotInitialized,
            7 => cudaError::ProfilerAlreadyStarted,
            8 => cudaError::ProfilerAlreadyStopped,
            9 => cudaError::InvalidConfiguration,
            12 => cudaError::InvalidPitchValue,
            13 => cudaError::InvalidSymbol,
            16 => cudaError::InvalidHostPointer,
            17 => cudaError::InvalidDevicePointer,
            18 => cudaError::InvalidTexture,
            19 => cudaError::InvalidTextureBinding,
            20 => cudaError::InvalidChannelDescriptor,
            21 => cudaError::InvalidMemcpyDirection,
            22 => cudaError::AddressOfConstant,
            23 => cudaError::TextureFetchFailed,
            24 => cudaError::TextureNotBound,
            25 => cudaError::SynchronizationError,
            26 => cudaError::InvalidFilterSetting,
            27 => cudaError::InvalidNormSetting,
            28 => cudaError::MixedDeviceExecution,
            31 => cudaError::NotYetImplemented,
            32 => cudaError::MemoryValueTooLarge,
            34 => cudaError::StubLibrary,
            35 => cudaError::InsufficientDriver,
            36 => cudaError::CallRequiresNewerDriver,
            37 => cudaError::InvalidSurface,
            43 => cudaError::DuplicateVariableName,
            44 => cudaError::DuplicateTextureName,
            45 => cudaError::DuplicateSurfaceName,
            46 => cudaError::DevicesUnavailable,
            49 => cudaError::IncompatibleDriverContext,
            52 => cudaError::MissingConfiguration,
            53 => cudaError::PriorLaunchFailure,
            65 => cudaError::LaunchMaxDepthExceeded,
            66 => cudaError::LaunchFileScopedTex,
            67 => cudaError::LaunchFileScopedSurf,
            68 => cudaError::SyncDepthExceeded,
            69 => cudaError::LaunchPendingCountExceeded,
            98 => cudaError::InvalidDeviceFunction,
            100 => cudaError::NoDevice,
            101 => cudaError::InvalidDevice,
            102 => cudaError::DeviceNotLicensed,
            103 => cudaError::SoftwareValidityNotEstablished,
            127 => cudaError::StartupFailure,
            200 => cudaError::InvalidKernelImage,
            201 => cudaError::DeviceUninitialized,
            205 => cudaError::MapBufferObjectFailed,
            206 => cudaError::UnmapBufferObjectFailed,
            207 => cudaError::ArrayIsMapped,
            208 => cudaError::AlreadyMapped,
            209 => cudaError::NoKernelImageForDevice,
            210 => cudaError::AlreadyAcquired,
            211 => cudaError::NotMapped,
            212 => cudaError::NotMappedAsArray,
            213 => cudaError::NotMappedAsPointer,
            214 => cudaError::ECCUncorrectable,
            215 => cudaError::UnsupportedLimit,
            216 => cudaError::DeviceAlreadyInUse,
            217 => cudaError::PeerAccessUnsupported,
            218 => cudaError::InvalidPtx,
            219 => cudaError::InvalidGraphicsContext,
            220 => cudaError::NvlinkUncorrectable,
            221 => cudaError::JitCompilerNotFound,
            222 => cudaError::UnsupportedPtxVersion,
            223 => cudaError::JitCompilationDisabled,
            224 => cudaError::UnsupportedExecAffinity,
            300 => cudaError::InvalidSource,
            301 => cudaError::FileNotFound,
            302 => cudaError::SharedObjectSymbolNotFound,
            303 => cudaError::SharedObjectInitFailed,
            304 => cudaError::OperatingSystem,
            400 => cudaError::InvalidResourceHandle,
            401 => cudaError::IllegalState,
            500 => cudaError::SymbolNotFound,
            600 => cudaError::NotReady,
            700 => cudaError::IllegalAddress,
            701 => cudaError::LaunchOutOfResources,
            702 => cudaError::LaunchTimeout,
            703 => cudaError::LaunchIncompatibleTexturing,
            704 => cudaError::PeerAccessAlreadyEnabled,
            705 => cudaError::PeerAccessNotEnabled,
            708 => cudaError::SetOnActiveProcess,
            709 => cudaError::ContextIsDestroyed,
            710 => cudaError::Assert,
            711 => cudaError::TooManyPeers,
            712 => cudaError::HostMemoryAlreadyRegistered,
            713 => cudaError::HostMemoryNotRegistered,
            714 => cudaError::HardwareStackError,
            715 => cudaError::IllegalInstruction,
            716 => cudaError::MisalignedAddress,
            717 => cudaError::InvalidAddressSpace,
            718 => cudaError::InvalidPc,
            719 => cudaError::LaunchFailure,
            720 => cudaError::CooperativeLaunchTooLarge,
            800 => cudaError::NotPermitted,
            801 => cudaError::NotSupported,
            802 => cudaError::SystemNotReady,
            803 => cudaError::SystemDriverMismatch,
            804 => cudaError::CompatNotSupportedOnDevice,
            805 => cudaError::MpsConnectionFailed,
            806 => cudaError::MpsRpcFailure,
            807 => cudaError::MpsServerNotReady,
            808 => cudaError::MpsMaxClientsReached,
            809 => cudaError::MpsMaxConnectionsReached,
            900 => cudaError::StreamCaptureUnsupported,
            901 => cudaError::StreamCaptureInvalidated,
            902 => cudaError::StreamCaptureMerge,
            903 => cudaError::StreamCaptureUnmatched,
            904 => cudaError::StreamCaptureUnjoined,
            905 => cudaError::StreamCaptureIsolation,
            906 => cudaError::StreamCaptureImplicit,
            907 => cudaError::CapturedEvent,
            908 => cudaError::StreamCaptureWrongThread,
            909 => cudaError::Timeout,
            910 => cudaError::GraphExecUpdateFailure,
            911 => cudaError::ExternalDevice,
            999 => cudaError::Unknown,
            10000 => cudaError::ApiFailureBase,
            _ => cudaError::Unknown,
        }
    }
}

impl Error for cudaError {}

#[derive(Debug)]
pub enum cudnnError {
    NOT_INITIALIZED,
    ALLOC_FAILED,
    BAD_PARAM,
    INTERNAL_ERROR,
    INVALID_VALUE,
    ARCH_MISMATCH,
    MAPPING_ERROR,
    EXECUTION_FAILED,
    NOT_SUPPORTED,
    LICENSE_ERROR,
    RUNTIME_PREREQUISITE_MISSING,
    RUNTIME_IN_PROGRESS,
    RUNTIME_FP_OVERFLOW,
    VERSION_MISMATCH,
    UNKNOWN
}

impl fmt::Display for cudnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_desc = match self {
            cudnnError::NOT_INITIALIZED => "The cuDNN library was not initialized properly. This error is usually returned when a call to cudnnCreate() fails or when cudnnCreate() has not been called prior to calling another cuDNN routine. In the former case, it is usually due to an error in the CUDA Runtime API called by cudnnCreate() or by an error in the hardware setup.",
            cudnnError::ALLOC_FAILED => "Resource allocation failed inside the cuDNN library. This is usually caused by an internal cudaMalloc() failure.\nTo correct, prior to the function call, deallocate previously allocated memory as much as possible.",
            cudnnError::BAD_PARAM => "An incorrect value or parameter was passed to the function.\nTo correct, ensure that all the parameters being passed have valid values.",
            cudnnError::INTERNAL_ERROR => "An internal cuDNN operation failed.",
            cudnnError::INVALID_VALUE => "Function specific errror. Refer docs at https://docs.nvidia.com/deeplearning/cudnn/api/index.html",
            cudnnError::ARCH_MISMATCH => "The function requires a feature absent from the current GPU device. Note that cuDNN only supports devices with compute capabilities greater than or equal to 3.0.\nTo correct, compile and run the application on a device with appropriate compute capability.",
            cudnnError::MAPPING_ERROR => "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\nTo correct, prior to the function call, unbind any previously bound textures.\nOtherwise, this may indicate an internal error/bug in the library.",
            cudnnError::EXECUTION_FAILED => "Function specific errror. Refer docs at https://docs.nvidia.com/deeplearning/cudnn/api/index.html",
            cudnnError::NOT_SUPPORTED => "Function specific errror. Refer docs at https://docs.nvidia.com/deeplearning/cudnn/api/index.html",
            cudnnError::LICENSE_ERROR => "The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.",
            cudnnError::RUNTIME_PREREQUISITE_MISSING => "A prerequisite runtime library cannot be found.",
            cudnnError::RUNTIME_IN_PROGRESS => "Some tasks in the user stream are not completed.",
            cudnnError::RUNTIME_FP_OVERFLOW => "Numerical overflow occurred during the GPU kernel execution.",
            cudnnError::VERSION_MISMATCH => "The version of this DLL file does not match that of a cuDNN DLLs on which it depends.",
            cudnnError::UNKNOWN => "This error is defined by rust bindings. If it occurs, means that returned unexpected int value."
        };
        write!(f, "{}", err_desc)
    }
}

impl From<u32> for cudnnError {
    fn from(error: u32) -> Self {
        match error {
            1  => cudnnError::NOT_INITIALIZED,
            2  => cudnnError::ALLOC_FAILED,
            3  => cudnnError::BAD_PARAM,
            4  => cudnnError::INTERNAL_ERROR,
            5  => cudnnError::INVALID_VALUE,
            6  => cudnnError::ARCH_MISMATCH,
            7  => cudnnError::MAPPING_ERROR,
            8  => cudnnError::EXECUTION_FAILED,
            9  => cudnnError::NOT_SUPPORTED,
            10 => cudnnError::LICENSE_ERROR,
            11 => cudnnError::RUNTIME_PREREQUISITE_MISSING,
            12 => cudnnError::RUNTIME_IN_PROGRESS,
            13 => cudnnError::RUNTIME_FP_OVERFLOW,
            14 => cudnnError::VERSION_MISMATCH,
            _  => cudnnError::UNKNOWN
        }
    }
}

impl Error for cudnnError {}

pub fn cudaMalloc(size: usize) -> Result<*mut c_void, cudaError> {
    use pnn_sys::cudaMalloc;

    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let res = cudaMalloc(&mut ptr as *mut *mut c_void, size);
        match  res{
            0 => Ok(ptr),
            x => Err(cudaError::from(x))
        }
    }
}

pub fn cudaMallocHost(size: usize) -> Result<*mut c_void, cudaError> {
    use pnn_sys::cudaMallocHost;

    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let res = cudaMallocHost(&mut ptr as *mut *mut c_void, size);
        match  res{
            0 => Ok(ptr),
            x => Err(cudaError::from(x))
        }
    }
}

pub fn cudaFree(ptr: *mut c_void) -> Result<(), cudaError> {
    use pnn_sys::cudaFree;

    unsafe {
        let res = cudaFree(ptr);
        match  res{
            0 => Ok(()),
            x => Err(cudaError::from(x))
        }
    }
}

pub use pnn_sys::{cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorFormat_t};

pub fn cudnnCreate() -> Result<cudnnHandle_t, cudnnError> {
    use pnn_sys::cudnnCreate;

    unsafe {
        let mut ptr: cudnnHandle_t = std::ptr::null_mut() as cudnnHandle_t;
        let res = cudnnCreate(&mut ptr as *mut cudnnHandle_t);
        match  res{
            0 => Ok(ptr),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnDestroy(handle: cudnnHandle_t) -> Result<(), cudnnError> {
    use pnn_sys::cudnnDestroy;

    unsafe {
        let res = cudnnDestroy(handle);
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnCreateTensorDescriptor() -> Result<cudnnTensorDescriptor_t, cudnnError> {
    use pnn_sys::cudnnCreateTensorDescriptor;

    unsafe {
        let mut ptr: cudnnTensorDescriptor_t = std::ptr::null_mut() as cudnnTensorDescriptor_t;
        let res = cudnnCreateTensorDescriptor(&mut ptr as *mut cudnnTensorDescriptor_t);
        match  res{
            0 => Ok(ptr),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> Result<(), cudnnError> {
    use pnn_sys::cudnnDestroyTensorDescriptor;

    unsafe {
        let res = cudnnDestroyTensorDescriptor(tensorDesc);
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum cudnnDataType {
    FLOAT = 0,
    DOUBLE = 1,
    HALF = 2,
    INT8 = 3,
    INT32 = 4,
    INT8x4 = 5,
    UINT8 = 6,
    UINT8x4 = 7,
    INT8x32 = 8,
    BFLOAT16 = 9,
    INT64 = 10
}

pub fn cudnnSetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    data_type: cudnnDataType,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Result<(), cudnnError> {
    use pnn_sys::{
        cudnnSetTensor4dDescriptor, 
        cudnnTensorFormat_t_CUDNN_TENSOR_NCHW, 
        cudnnDataType_t
    };

    unsafe {
        let res = cudnnSetTensor4dDescriptor(
            tensorDesc,
            cudnnTensorFormat_t_CUDNN_TENSOR_NCHW, 
            data_type as cudnnDataType_t,
            n as ::std::os::raw::c_int,
            c as ::std::os::raw::c_int,
            h as ::std::os::raw::c_int,
            w as ::std::os::raw::c_int
        );
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}

pub enum cudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

pub fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> Result<(), cudaError> {
    use pnn_sys::cudaMemcpy;

    unsafe {
        let res = cudaMemcpy(dst, src, count, kind as u32);
        match  res{
            0 => Ok(()),
            x => Err(cudaError::from(x))
        }
    }
}

// Implement C = A * alpha + C * beta
pub fn cudnnAddTensor(
    handle: cudnnHandle_t,
    alpha: f32,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::std::os::raw::c_void,
    beta: f32,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::std::os::raw::c_void,
) -> Result<(), cudnnError> {
    use pnn_sys::{
        cudnnAddTensor,
    };

    unsafe {
        let res = cudnnAddTensor(
            handle,
            std::ptr::addr_of!(alpha) as *const c_void, 
            aDesc,
            A,
            std::ptr::addr_of!(beta) as *const c_void,
            cDesc,
            C
        );
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}


pub use pnn_sys::{cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t};

pub fn cudnnCreateFilterDescriptor() -> Result<cudnnFilterDescriptor_t, cudnnError> {
    use pnn_sys::cudnnCreateFilterDescriptor;

    unsafe {
        let mut ptr: cudnnFilterDescriptor_t = std::ptr::null_mut() as cudnnFilterDescriptor_t;
        let res = cudnnCreateFilterDescriptor(&mut ptr as *mut cudnnFilterDescriptor_t);
        match  res{
            0 => Ok(ptr),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> Result<(), cudnnError> {
    use pnn_sys::cudnnDestroyFilterDescriptor;

    unsafe {
        let res = cudnnDestroyFilterDescriptor(filterDesc);
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnSetFilter4dDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    data_type: cudnnDataType,
    k: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Result<(), cudnnError> {
    use pnn_sys::{
        cudnnSetFilter4dDescriptor, 
        cudnnTensorFormat_t_CUDNN_TENSOR_NCHW, 
        cudnnDataType_t
    };

    unsafe {
        let res = cudnnSetFilter4dDescriptor(
            filterDesc,
            cudnnTensorFormat_t_CUDNN_TENSOR_NCHW, 
            data_type as cudnnDataType_t,
            k as ::std::os::raw::c_int,
            c as ::std::os::raw::c_int,
            h as ::std::os::raw::c_int,
            w as ::std::os::raw::c_int
        );
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnCreateConvolutionDescriptor() -> Result<cudnnConvolutionDescriptor_t, cudnnError> {
    use pnn_sys::cudnnCreateConvolutionDescriptor;

    unsafe {
        let mut ptr = std::ptr::null_mut() as cudnnConvolutionDescriptor_t;
        let res = cudnnCreateConvolutionDescriptor(&mut ptr as *mut cudnnConvolutionDescriptor_t);
        match  res{
            0 => Ok(ptr),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> Result<(), cudnnError> {
    use pnn_sys::cudnnDestroyConvolutionDescriptor;

    unsafe {
        let res = cudnnDestroyConvolutionDescriptor(convDesc);
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}

pub fn cudnnSetConvolution2dDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: usize,
    pad_w: usize,
    u: usize,
    v: usize,
    dilation_h: usize,
    dilation_w: usize,
    computeType: cudnnDataType,
) -> Result<(), cudnnError> {
    use pnn_sys::{
        cudnnSetConvolution2dDescriptor, 
        cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION, 
        cudnnDataType_t
    };

    unsafe {
        let res = cudnnSetConvolution2dDescriptor(
            convDesc,
            pad_h as ::std::os::raw::c_int,
            pad_w as ::std::os::raw::c_int,
            u as ::std::os::raw::c_int,
            v as ::std::os::raw::c_int,
            dilation_h as ::std::os::raw::c_int,
            dilation_w as ::std::os::raw::c_int,
            cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION, // Use CC by default
            computeType as cudnnDataType_t,
        );
        match  res{
            0 => Ok(()),
            x => Err(cudnnError::from(x))
        }
    }
}