use std::{
    error::Error,
    fmt,
    self,
};
use core::fmt::Debug;
use std::{os::raw::{c_void, }};

#[derive(Debug)]
pub enum CudaError{
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


impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_desc = match self {
            InvalidValue => "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.",
            MemoryAllocation => "The API call failed because it was unable to allocate enough memory to perform the requested operation.",
            InitializationError => "The API call failed because the CUDA driver and runtime could not be initialized.",
            CudartUnloading => "This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a pointin time after CUDA driver has been unloaded.",
            ProfilerDisabled => "This indicates profiler is not initialized for this run. This can happen when the application is running with external profilingtools like visual profiler.",
            ProfilerNotInitialized => "Deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via cudaProfilerStart or cudaProfilerStop without initialization.",
            ProfilerAlreadyStarted => "Deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStart() when profiling is already enabled.",
            ProfilerAlreadyStopped => "Deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStop() when profiling is already disabled.",
            InvalidConfiguration => "This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requestingmore shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks.See cudaDeviceProp for more device limitations.",
            InvalidPitchValue => "This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable rangefor pitch.",
            InvalidSymbol => "This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.",
            InvalidHostPointer => "Deprecated This error return is deprecated as of CUDA 10.1.  This indicates that at least one host pointer passed to the API call is not a valid host pointer.",
            InvalidDevicePointer => "Deprecated This error return is deprecated as of CUDA 10.1.  This indicates that at least one device pointer passed to the API call is not a valid device pointer.",
            InvalidTexture => "This indicates that the texture passed to the API call is not a valid texture.",
            InvalidTextureBinding => "This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture.",
            InvalidChannelDescriptor => "This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of theformats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.",
            InvalidMemcpyDirection => "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind.",
            AddressOfConstant => "Deprecated This error return is deprecated as of CUDA 3.1. Variables in constant memory may now have their address taken by the runtime      via cudaGetSymbolAddress().  This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release.",
            TextureFetchFailed => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture   operations.",
            TextureNotBound => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations.",
            SynchronizationError => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a synchronization operation had failed. This was previously used for some device emulation functions.",
            InvalidFilterSetting => "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.",
            InvalidNormSetting => "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.",
            MixedDeviceExecution => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  Mixing of device and device emulation code was not allowed.",
            NotYetImplemented => "Deprecated This error return is deprecated as of CUDA 4.1.  This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error.",
            MemoryValueTooLarge => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that an emulated device pointer exceeded the 32-bit address range.",
            StubLibrary => "This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stubrather than a real driver loaded will result in CUDA API returning this error.",
            InsufficientDriver => "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration.Users should install an updated NVIDIA display driver to allow the application to run.",
            CallRequiresNewerDriver => "This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updatedNVIDIA CUDA driver to allow the API call to succeed.",
            InvalidSurface => "This indicates that the surface passed to the API call is not a valid surface.",
            DuplicateVariableName => "This indicates that multiple global or constant variables (across separate CUDA source files in the application) share thesame string name.",
            DuplicateTextureName => "This indicates that multiple textures (across separate CUDA source files in the application) share the same string name.",
            DuplicateSurfaceName => "This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name.",
            DevicesUnavailable => "This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due touse of cudaComputeModeExclusive, cudaComputeModeProhibited or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailabledue to memory constraints on a device that already has active CUDA work being performed.",
            IncompatibleDriverContext => "This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are usingCUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver contextmay be incompatible either because the Driver context was created using an older version of the API, because the Runtime APIcall expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed.Please see Interactions with the CUDA Driver API for more information.",
            MissingConfiguration => "The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the cudaConfigureCall() function.",
            PriorLaunchFailure => "Deprecated This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.  This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches.",
            LaunchMaxDepthExceeded => "This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed themaximum supported number of nested grid launches.",
            LaunchFileScopedTex => "This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported bythe device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.",
            LaunchFileScopedSurf => "This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported bythe device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.",
            SyncDepthExceeded => "This error indicates that a call to cudaDeviceSynchronize made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levelsof grids) or user specified device limit cudaLimitDevRuntimeSyncDepth. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which cudaDeviceSynchronize will be called must be specified with the cudaLimitDevRuntimeSyncDepth limit to the cudaDeviceSetLimit api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth requirethe runtime to reserve large amounts of device memory that cannot be used for user allocations.",
            LaunchPendingCountExceeded => "This error indicates that a device runtime grid launch failed because the launch would exceed the limit cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed successfully, cudaDeviceSetLimit must be called to set the cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raisingthe limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for userallocations.",
            InvalidDeviceFunction => "The requested device function does not exist or is not compiled for the proper device architecture.",
            NoDevice => "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.",
            InvalidDevice => "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the actionrequested is invalid for the specified device.",
            DeviceNotLicensed => "This indicates that the device doesn't have a valid Grid License.",
            SoftwareValidityNotEstablished => "By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validityof both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validityof either the runtime or the driver could not be established.",
            StartupFailure => "This indicates an internal startup failure in the CUDA runtime.",
            InvalidKernelImage => "This indicates that the device kernel image is invalid.",
            DeviceUninitialized => "This most frequently indicates that there is no context bound to the current thread. This can also be returned if the contextpassed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls).See cuCtxGetApiVersion() for more details.",
            MapBufferObjectFailed => "This indicates that the buffer object could not be mapped.",
            UnmapBufferObjectFailed => "This indicates that the buffer object could not be unmapped.",
            ArrayIsMapped => "This indicates that the specified array is currently mapped and thus cannot be destroyed.",
            AlreadyMapped => "This indicates that the resource is already mapped.",
            NoKernelImageForDevice => "This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifiescode generation options for a particular CUDA source file that do not include the corresponding device configuration.",
            AlreadyAcquired => "This indicates that a resource has already been acquired.",
            NotMapped => "This indicates that a resource is not mapped.",
            NotMappedAsArray => "This indicates that a mapped resource is not available for access as an array.",
            NotMappedAsPointer => "This indicates that a mapped resource is not available for access as a pointer.",
            ECCUncorrectable => "This indicates that an uncorrectable ECC error was detected during execution.",
            UnsupportedLimit => "This indicates that the cudaLimit passed to the API call is not supported by the active device.",
            DeviceAlreadyInUse => "This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.",
            PeerAccessUnsupported => "This error indicates that P2P access is not supported across the given devices.",
            InvalidPtx => "A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binaryfor the current device.",
            InvalidGraphicsContext => "This indicates an error with the OpenGL or DirectX context.",
            NvlinkUncorrectable => "This indicates that an uncorrectable NVLink error was detected during the execution.",
            JitCompilerNotFound => "This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. Theruntime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.",
            UnsupportedPtxVersion => "This indicates that the provided PTX was compiled with an unsupported toolchain. The most common reason for this, is the PTXwas generated by a compiler newer than what is supported by the CUDA driver and PTX JIT compiler.",
            JitCompilationDisabled => "This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compilingPTX if an application does not contain a suitable binary for the current device.",
            UnsupportedExecAffinity => "This indicates that the provided execution affinity is not supported by the device.",
            InvalidSource => "This indicates that the device kernel source is invalid.",
            FileNotFound => "This indicates that the file specified was not found.",
            SharedObjectSymbolNotFound => "This indicates that a link to a shared object failed to resolve.",
            SharedObjectInitFailed => "This indicates that initialization of a shared object failed.",
            OperatingSystem => "This error indicates that an OS call failed.",
            InvalidResourceHandle => "This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t.",
            IllegalState => "This indicates that a resource required by the API call is not in a valid state to perform the requested operation.",
            SymbolNotFound => "This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver functionnames, texture names, and surface names.",
            NotReady => "This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error,but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery().",
            IllegalAddress => "The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistentstate and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.",
            LaunchOutOfResources => "This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar tocudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernellaunch specifies too many threads for the kernel's register count.",
            LaunchTimeout => "This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the deviceproperty kernelExecTimeoutEnabled for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error.To continue using CUDA, the process must be terminated and relaunched.",
            LaunchIncompatibleTexturing => "This error indicates a kernel launch that uses an incompatible texturing mode.",
            PeerAccessAlreadyEnabled => "This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.",
            PeerAccessNotEnabled => "This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().",
            SetOnActiveProcess => "This indicates that the user has called cudaSetValidDevices(), cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(), or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernelsare examples of non-device management operations). This error can also be returned if using runtime/driver interoperabilityand there is an existing CUcontext active on the host thread.",
            ContextIsDestroyed => "This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.",
            Assert => "An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations areinvalid. To continue using CUDA, the process must be terminated and relaunched.",
            TooManyPeers => "This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of thedevices passed to cudaEnablePeerAccess().",
            HostMemoryAlreadyRegistered => "This error indicates that the memory range passed to cudaHostRegister() has already been registered.",
            HostMemoryNotRegistered => "This error indicates that the pointer passed to cudaHostUnregister() does not correspond to any currently registered memory region.",
            HardwareStackError => "Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stacksize limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continueusing CUDA, the process must be terminated and relaunched.",
            IllegalInstruction => "The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state andany further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.",
            MisalignedAddress => "The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in aninconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminatedand relaunched.",
            InvalidAddressSpace => "While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain addressspaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leavesthe process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the processmust be terminated and relaunched.",
            InvalidPc => "The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA workwill return the same error. To continue using CUDA, the process must be terminated and relaunched.",
            LaunchFailure => "An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointerand accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases canbe found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work willreturn the same error. To continue using CUDA, the process must be terminated and relaunched.",
            CooperativeLaunchTooLarge => "This error indicates that the number of blocks launched per grid for a kernel that was launched via either cudaLaunchCooperativeKernel or cudaLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cudaOccupancyMaxActiveBlocksPerMultiprocessor or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount.",
            NotPermitted => "This error indicates the attempted operation is not permitted.",
            NotSupported => "This error indicates the attempted operation is not supported on the current system or device.",
            SystemNotReady => "This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configurationis in a valid state and all required driver daemons are actively running. More information about this error can be found inthe system specific user guide.",
            SystemDriverMismatch => "This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to thecompatibility documentation for supported versions.",
            CompatNotSupportedOnDevice => "This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDAdoes not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensurethat only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.",
            MpsConnectionFailed => "This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.",
            MpsRpcFailure => "This error indicates that the remote procedural call between the MPS server and the MPS client failed.",
            MpsServerNotReady => "This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when theMPS server is in the process of recovering from a fatal failure.",
            MpsMaxClientsReached => "This error indicates that the hardware resources required to create MPS client have been exhausted.",
            MpsMaxConnectionsReached => "This error indicates the the hardware resources required to device connections have been exhausted.",
            StreamCaptureUnsupported => "The operation is not permitted when the stream is capturing.",
            StreamCaptureInvalidated => "The current capture sequence on the stream has been invalidated due to a previous error.",
            StreamCaptureMerge => "The operation would have resulted in a merge of two independent capture sequences.",
            StreamCaptureUnmatched => "The capture was not initiated in this stream.",
            StreamCaptureUnjoined => "The capture sequence contains a fork that was not joined to the primary stream.",
            StreamCaptureIsolation => "A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependenciesare allowed to cross the boundary.",
            StreamCaptureImplicit => "The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.",
            CapturedEvent => "The operation is not permitted on an event which was last recorded in a capturing stream.",
            StreamCaptureWrongThread => "A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture in a different thread.",
            Timeout => "This indicates that the wait operation has timed out.",
            GraphExecUpdateFailure => "This error indicates that the graph update was not performed because it included changes which violated constraints specificto instantiated graph update.",
            ExternalDevice => "This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device'ssignal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption.This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA,the process must be terminated and relaunched.",
            Unknown => "This indicates that an unknown internal error has occurred.",
            ApiFailureBase => "Deprecated This error return is deprecated as of CUDA 4.1.  Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not   return such errors.",};
        write!(f, "{}", err_desc)
    }
}

impl From<u32> for CudaError {
    fn from(error: u32) -> Self {
        match error {
            1 => CudaError::InvalidValue,
            2 => CudaError::MemoryAllocation,
            3 => CudaError::InitializationError,
            4 => CudaError::CudartUnloading,
            5 => CudaError::ProfilerDisabled,
            6 => CudaError::ProfilerNotInitialized,
            7 => CudaError::ProfilerAlreadyStarted,
            8 => CudaError::ProfilerAlreadyStopped,
            9 => CudaError::InvalidConfiguration,
            12 => CudaError::InvalidPitchValue,
            13 => CudaError::InvalidSymbol,
            16 => CudaError::InvalidHostPointer,
            17 => CudaError::InvalidDevicePointer,
            18 => CudaError::InvalidTexture,
            19 => CudaError::InvalidTextureBinding,
            20 => CudaError::InvalidChannelDescriptor,
            21 => CudaError::InvalidMemcpyDirection,
            22 => CudaError::AddressOfConstant,
            23 => CudaError::TextureFetchFailed,
            24 => CudaError::TextureNotBound,
            25 => CudaError::SynchronizationError,
            26 => CudaError::InvalidFilterSetting,
            27 => CudaError::InvalidNormSetting,
            28 => CudaError::MixedDeviceExecution,
            31 => CudaError::NotYetImplemented,
            32 => CudaError::MemoryValueTooLarge,
            34 => CudaError::StubLibrary,
            35 => CudaError::InsufficientDriver,
            36 => CudaError::CallRequiresNewerDriver,
            37 => CudaError::InvalidSurface,
            43 => CudaError::DuplicateVariableName,
            44 => CudaError::DuplicateTextureName,
            45 => CudaError::DuplicateSurfaceName,
            46 => CudaError::DevicesUnavailable,
            49 => CudaError::IncompatibleDriverContext,
            52 => CudaError::MissingConfiguration,
            53 => CudaError::PriorLaunchFailure,
            65 => CudaError::LaunchMaxDepthExceeded,
            66 => CudaError::LaunchFileScopedTex,
            67 => CudaError::LaunchFileScopedSurf,
            68 => CudaError::SyncDepthExceeded,
            69 => CudaError::LaunchPendingCountExceeded,
            98 => CudaError::InvalidDeviceFunction,
            100 => CudaError::NoDevice,
            101 => CudaError::InvalidDevice,
            102 => CudaError::DeviceNotLicensed,
            103 => CudaError::SoftwareValidityNotEstablished,
            127 => CudaError::StartupFailure,
            200 => CudaError::InvalidKernelImage,
            201 => CudaError::DeviceUninitialized,
            205 => CudaError::MapBufferObjectFailed,
            206 => CudaError::UnmapBufferObjectFailed,
            207 => CudaError::ArrayIsMapped,
            208 => CudaError::AlreadyMapped,
            209 => CudaError::NoKernelImageForDevice,
            210 => CudaError::AlreadyAcquired,
            211 => CudaError::NotMapped,
            212 => CudaError::NotMappedAsArray,
            213 => CudaError::NotMappedAsPointer,
            214 => CudaError::ECCUncorrectable,
            215 => CudaError::UnsupportedLimit,
            216 => CudaError::DeviceAlreadyInUse,
            217 => CudaError::PeerAccessUnsupported,
            218 => CudaError::InvalidPtx,
            219 => CudaError::InvalidGraphicsContext,
            220 => CudaError::NvlinkUncorrectable,
            221 => CudaError::JitCompilerNotFound,
            222 => CudaError::UnsupportedPtxVersion,
            223 => CudaError::JitCompilationDisabled,
            224 => CudaError::UnsupportedExecAffinity,
            300 => CudaError::InvalidSource,
            301 => CudaError::FileNotFound,
            302 => CudaError::SharedObjectSymbolNotFound,
            303 => CudaError::SharedObjectInitFailed,
            304 => CudaError::OperatingSystem,
            400 => CudaError::InvalidResourceHandle,
            401 => CudaError::IllegalState,
            500 => CudaError::SymbolNotFound,
            600 => CudaError::NotReady,
            700 => CudaError::IllegalAddress,
            701 => CudaError::LaunchOutOfResources,
            702 => CudaError::LaunchTimeout,
            703 => CudaError::LaunchIncompatibleTexturing,
            704 => CudaError::PeerAccessAlreadyEnabled,
            705 => CudaError::PeerAccessNotEnabled,
            708 => CudaError::SetOnActiveProcess,
            709 => CudaError::ContextIsDestroyed,
            710 => CudaError::Assert,
            711 => CudaError::TooManyPeers,
            712 => CudaError::HostMemoryAlreadyRegistered,
            713 => CudaError::HostMemoryNotRegistered,
            714 => CudaError::HardwareStackError,
            715 => CudaError::IllegalInstruction,
            716 => CudaError::MisalignedAddress,
            717 => CudaError::InvalidAddressSpace,
            718 => CudaError::InvalidPc,
            719 => CudaError::LaunchFailure,
            720 => CudaError::CooperativeLaunchTooLarge,
            800 => CudaError::NotPermitted,
            801 => CudaError::NotSupported,
            802 => CudaError::SystemNotReady,
            803 => CudaError::SystemDriverMismatch,
            804 => CudaError::CompatNotSupportedOnDevice,
            805 => CudaError::MpsConnectionFailed,
            806 => CudaError::MpsRpcFailure,
            807 => CudaError::MpsServerNotReady,
            808 => CudaError::MpsMaxClientsReached,
            809 => CudaError::MpsMaxConnectionsReached,
            900 => CudaError::StreamCaptureUnsupported,
            901 => CudaError::StreamCaptureInvalidated,
            902 => CudaError::StreamCaptureMerge,
            903 => CudaError::StreamCaptureUnmatched,
            904 => CudaError::StreamCaptureUnjoined,
            905 => CudaError::StreamCaptureIsolation,
            906 => CudaError::StreamCaptureImplicit,
            907 => CudaError::CapturedEvent,
            908 => CudaError::StreamCaptureWrongThread,
            909 => CudaError::Timeout,
            910 => CudaError::GraphExecUpdateFailure,
            911 => CudaError::ExternalDevice,
            999 => CudaError::Unknown,
            10000 => CudaError::ApiFailureBase,
            _ => CudaError::Unknown,
        }
    }
}

impl Error for CudaError {}

pub fn cudaMalloc(size: usize) -> Result<*mut c_void, CudaError> {
    use pnn_sys::cudaMalloc;

    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let res = cudaMalloc(&mut ptr as *mut *mut c_void, size);
        match  res{
            0 => Ok(ptr),
            x => Err(CudaError::from(x))
        }
    }
}

pub fn cudaFree(ptr: *mut c_void) -> Result<(), CudaError> {
    use pnn_sys::cudaFree;
    extern crate libc;

    unsafe {
        let res = cudaFree(ptr);
        match  res{
            0 => Ok(()),
            x => Err(CudaError::from(x))
        }
    }
}