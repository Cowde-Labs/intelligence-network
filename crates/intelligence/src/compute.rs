//! Backend-neutral compute boundary for V5.
//!
//! The distributed runtime deals in bounded tasks, generations and state
//! hashes.  This module is the only place where a worker selects a local
//! compute implementation.  The reference CPU implementation is complete;
//! optional accelerator adapters expose discovery and an explicit native
//! execution boundary without adding a framework dependency.

use blake3::Hasher;
use intelligence_protocol::{
    BackendCapabilities, BackendHealth, BackendKind, CapabilityChallenge,
    CapabilityChallengeResult, CapabilityEvidenceRecord, ComputeFeature, ComputeRequirements,
    ComputeTaskKind, NumericFormat, RequestId,
};
use std::{collections::HashMap, time::Instant};
use thiserror::Error;

const MAX_TASK_ELEMENTS: u64 = 128 * 128;
const MAX_TASK_BYTES: u64 = 8 * 1024 * 1024;
const MAX_QUEUE: u16 = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeOperation {
    MatrixVector { rows: u16, cols: u16 },
    MatrixTransposeVector { rows: u16, cols: u16 },
    Affine { coefficient: i64, bias: i64 },
    Gradient { coefficient: i64 },
    ElementwiseMultiply,
    Challenge,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeTask {
    pub operation: ComputeOperation,
    pub requirements: ComputeRequirements,
    pub input_elements: u64,
    pub output_elements: u64,
    pub graph_generation: u64,
    pub model_generation: u64,
    pub shard_id: u16,
    pub deadline_ms: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeInput {
    pub values: Vec<i64>,
    pub format: NumericFormat,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTask {
    pub task: ComputeTask,
    pub backend: BackendKind,
    pub reserved_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOutput {
    pub values: Vec<i64>,
    pub format: NumericFormat,
    pub backend: BackendKind,
    pub elapsed_micros: u64,
    pub converted: bool,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum ComputeError {
    #[error("backend is unavailable")]
    Unavailable,
    #[error("backend does not support the requested task")]
    Unsupported,
    #[error("compute task is malformed: {0}")]
    InvalidTask(String),
    #[error("compute task exceeds safe memory")]
    OutOfMemory,
    #[error("compute task queue is full")]
    QueueFull,
    #[error("compute task timed out")]
    Timeout,
    #[error("native backend failed: {0}")]
    NativeFailure(String),
    #[error("checked arithmetic overflow")]
    Overflow,
}

pub trait ComputeBackend: Send {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> BackendCapabilities;
    fn validate(&self, task: &ComputeTask) -> Result<(), ComputeError>;
    fn prepare(&mut self, task: &ComputeTask) -> Result<PreparedTask, ComputeError>;
    fn execute(
        &mut self,
        task: &PreparedTask,
        inputs: &[ComputeInput],
    ) -> Result<ComputeOutput, ComputeError>;
    fn synchronize(&mut self) -> Result<(), ComputeError>;
    fn health(&self) -> BackendHealth;
}

fn checked_elements(elements: u64) -> Result<u64, ComputeError> {
    if elements == 0 || elements > MAX_TASK_ELEMENTS {
        return Err(ComputeError::InvalidTask(
            "tensor element count is outside the bounded task limit".to_string(),
        ));
    }
    elements.checked_mul(8).ok_or(ComputeError::Overflow)
}

fn validate_task_shape(task: &ComputeTask) -> Result<(), ComputeError> {
    if task.graph_generation == 0
        || task.model_generation == 0
        || task.deadline_ms == 0
        || task.deadline_ms > 120_000
        || task.requirements.required_memory_bytes == 0
    {
        return Err(ComputeError::InvalidTask(
            "generation, deadline, or memory requirement is invalid".to_string(),
        ));
    }
    let input_bytes = checked_elements(task.input_elements)?;
    let output_bytes = checked_elements(task.output_elements)?;
    let total = input_bytes
        .checked_add(output_bytes)
        .and_then(|value| value.checked_add(task.requirements.required_memory_bytes))
        .ok_or(ComputeError::Overflow)?;
    if total > MAX_TASK_BYTES {
        return Err(ComputeError::OutOfMemory);
    }
    match task.operation {
        ComputeOperation::MatrixVector { rows, cols } => {
            if rows == 0
                || cols == 0
                || u64::from(cols) != task.input_elements
                || u64::from(rows) != task.output_elements
            {
                return Err(ComputeError::InvalidTask(
                    "matrix-vector shape does not match task metadata".to_string(),
                ));
            }
        }
        ComputeOperation::MatrixTransposeVector { rows, cols } => {
            if rows == 0
                || cols == 0
                || u64::from(rows) != task.input_elements
                || u64::from(cols) != task.output_elements
            {
                return Err(ComputeError::InvalidTask(
                    "transposed matrix-vector shape does not match task metadata".to_string(),
                ));
            }
        }
        ComputeOperation::Affine { .. } | ComputeOperation::Gradient { .. } => {
            if task.input_elements != task.output_elements {
                return Err(ComputeError::InvalidTask(
                    "elementwise shape does not match task metadata".to_string(),
                ));
            }
        }
        ComputeOperation::ElementwiseMultiply => {}
        ComputeOperation::Challenge => {}
    }
    Ok(())
}

fn validate_requirements(
    capabilities: &BackendCapabilities,
    task: &ComputeTask,
) -> Result<(), ComputeError> {
    validate_task_shape(task)?;
    if !capabilities.runtime_available || capabilities.health == BackendHealth::Unavailable {
        return Err(ComputeError::Unavailable);
    }
    if task.requirements.required_backend != Some(capabilities.kind)
        && !task
            .requirements
            .allowed_backends
            .contains(&capabilities.kind)
    {
        return Err(ComputeError::Unsupported);
    }
    let safe_memory = capabilities
        .available_memory_bytes
        .saturating_mul(u64::from(
            1000_u16.saturating_sub(capabilities.safety_margin_permille),
        ))
        / 1000;
    if task.requirements.required_memory_bytes > safe_memory
        || task.input_elements.saturating_add(task.output_elements)
            > capabilities.max_tensor_elements
        || task
            .requirements
            .required_formats
            .iter()
            .any(|format| !capabilities.formats.contains(format))
        || task
            .requirements
            .required_features
            .iter()
            .any(|feature| !capabilities.features.contains(feature))
    {
        return Err(ComputeError::Unsupported);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CpuBackend {
    capabilities: BackendCapabilities,
    queued: u16,
    reserved_bytes: u64,
}

impl CpuBackend {
    fn new() -> Self {
        let memory = host_available_memory();
        Self {
            capabilities: BackendCapabilities {
                kind: BackendKind::Cpu,
                runtime_version: "native-reference-1".to_string(),
                device_count: 1,
                device_memory_bytes: memory,
                available_memory_bytes: memory,
                formats: vec![NumericFormat::F32],
                max_tensor_elements: MAX_TASK_ELEMENTS,
                features: vec![
                    ComputeFeature::MatrixMultiply,
                    ComputeFeature::TensorParallel,
                    ComputeFeature::PipelineStage,
                    ComputeFeature::DeterministicKernel,
                ],
                device_architecture: Some(std::env::consts::ARCH.to_string()),
                driver_available: true,
                runtime_available: true,
                peer_to_peer: false,
                unified_memory: true,
                max_concurrent_tasks: 2,
                safety_margin_permille: 100,
                health: BackendHealth::Ready,
                physical_verified: true,
                observed_successes: 0,
                observed_failures: 0,
            },
            queued: 0,
            reserved_bytes: 0,
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn validate(&self, task: &ComputeTask) -> Result<(), ComputeError> {
        validate_requirements(&self.capabilities, task)
    }

    fn prepare(&mut self, task: &ComputeTask) -> Result<PreparedTask, ComputeError> {
        self.validate(task)?;
        let reserved_bytes = task
            .requirements
            .required_memory_bytes
            .checked_add(
                task.input_elements
                    .checked_add(task.output_elements)
                    .ok_or(ComputeError::Overflow)?
                    .checked_mul(8)
                    .ok_or(ComputeError::Overflow)?,
            )
            .ok_or(ComputeError::Overflow)?;
        let safe_memory = self
            .capabilities
            .available_memory_bytes
            .saturating_mul(u64::from(
                1000_u16.saturating_sub(self.capabilities.safety_margin_permille),
            ))
            / 1000;
        if self.reserved_bytes.saturating_add(reserved_bytes) > safe_memory {
            return Err(ComputeError::OutOfMemory);
        }
        if self.queued
            >= self
                .capabilities
                .max_concurrent_tasks
                .saturating_add(MAX_QUEUE)
        {
            return Err(ComputeError::QueueFull);
        }
        self.queued = self.queued.saturating_add(1);
        self.reserved_bytes = self.reserved_bytes.saturating_add(reserved_bytes);
        Ok(PreparedTask {
            task: task.clone(),
            backend: self.kind(),
            reserved_bytes,
        })
    }

    fn execute(
        &mut self,
        prepared: &PreparedTask,
        inputs: &[ComputeInput],
    ) -> Result<ComputeOutput, ComputeError> {
        let started = Instant::now();
        let task = &prepared.task;
        if inputs
            .iter()
            .any(|input| input.format != NumericFormat::F32)
        {
            return Err(ComputeError::Unsupported);
        }
        let values = match task.operation {
            ComputeOperation::MatrixVector { rows, cols } => {
                if inputs.len() != 2
                    || inputs[0].values.len() != usize::from(rows * cols)
                    || inputs[1].values.len() != usize::from(cols)
                {
                    return Err(ComputeError::InvalidTask(
                        "matrix-vector buffers do not match checked dimensions".to_string(),
                    ));
                }
                (0..usize::from(rows))
                    .map(|row| {
                        let start = row * usize::from(cols);
                        inputs[0].values[start..start + usize::from(cols)]
                            .iter()
                            .zip(&inputs[1].values)
                            .map(|(left, right)| i128::from(*left) * i128::from(*right))
                            .sum::<i128>()
                            .clamp(-1_000_000_000, 1_000_000_000) as i64
                    })
                    .collect()
            }
            ComputeOperation::Affine { coefficient, bias } => {
                if inputs.len() != 1 {
                    return Err(ComputeError::InvalidTask(
                        "affine task requires one input".to_string(),
                    ));
                }
                inputs[0]
                    .values
                    .iter()
                    .map(|value| {
                        (i128::from(*value) * i128::from(coefficient) + i128::from(bias))
                            .clamp(-1_000_000_000, 1_000_000_000) as i64
                    })
                    .collect()
            }
            ComputeOperation::MatrixTransposeVector { rows, cols } => {
                if inputs.len() != 2
                    || inputs[0].values.len() != usize::from(rows * cols)
                    || inputs[1].values.len() != usize::from(rows)
                {
                    return Err(ComputeError::InvalidTask(
                        "transposed matrix-vector buffers do not match checked dimensions"
                            .to_string(),
                    ));
                }
                (0..usize::from(cols))
                    .map(|col| {
                        (0..usize::from(rows))
                            .map(|row| {
                                inputs[0].values[row * usize::from(cols) + col]
                                    .saturating_mul(inputs[1].values[row])
                            })
                            .sum::<i64>()
                    })
                    .collect()
            }
            ComputeOperation::Gradient { coefficient } => {
                if inputs.len() != 1 {
                    return Err(ComputeError::InvalidTask(
                        "gradient task requires one input".to_string(),
                    ));
                }
                inputs[0]
                    .values
                    .iter()
                    .map(|value| {
                        (i128::from(*value) * i128::from(coefficient))
                            .clamp(-1_000_000_000, 1_000_000_000) as i64
                    })
                    .collect()
            }
            ComputeOperation::ElementwiseMultiply => {
                if inputs.len() != 2 || inputs[0].values.len() != inputs[1].values.len() {
                    return Err(ComputeError::InvalidTask(
                        "elementwise multiply buffers do not match".to_string(),
                    ));
                }
                inputs[0]
                    .values
                    .iter()
                    .zip(&inputs[1].values)
                    .map(|(left, right)| left.saturating_mul(*right))
                    .collect()
            }
            ComputeOperation::Challenge => {
                if inputs.len() != 1 {
                    return Err(ComputeError::InvalidTask(
                        "challenge task requires one input".to_string(),
                    ));
                }
                inputs[0]
                    .values
                    .iter()
                    .map(|value| value.saturating_mul(2).saturating_add(1))
                    .collect()
            }
        };
        self.queued = self.queued.saturating_sub(1);
        self.reserved_bytes = self.reserved_bytes.saturating_sub(prepared.reserved_bytes);
        Ok(ComputeOutput {
            values,
            format: NumericFormat::F32,
            backend: self.kind(),
            elapsed_micros: started.elapsed().as_micros().max(1) as u64,
            converted: false,
        })
    }

    fn synchronize(&mut self) -> Result<(), ComputeError> {
        self.queued = 0;
        self.reserved_bytes = 0;
        Ok(())
    }

    fn health(&self) -> BackendHealth {
        self.capabilities.health
    }
}

/// Runtime-loaded CUDA Driver API. This feature is optional and never part of
/// the default CPU build.
#[cfg(all(feature = "cuda", target_os = "linux"))]
mod cuda_native {
    use super::{ComputeError, ComputeOperation};
    use std::{
        ffi::{CString, c_char, c_int, c_void},
        ptr,
    };

    type CuResult = i32;
    type CuDevice = i32;
    type CuDevicePtr = u64;
    type CuContext = *mut c_void;
    type CuModule = *mut c_void;
    type CuFunction = *mut c_void;
    const CUDA_SUCCESS: CuResult = 0;
    const RTLD_NOW: c_int = 2;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    type CuInit = unsafe extern "C" fn(u32) -> CuResult;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
    type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
    type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
    type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
    type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, u32, CuDevice) -> CuResult;
    type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;
    type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
    type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
    type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
    type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
    type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
    type CuLaunchKernel = unsafe extern "C" fn(
        CuFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult;
    type CuCtxSynchronize = unsafe extern "C" fn() -> CuResult;

    const PTX: &[u8] = br#"
.version 7.0
.target sm_50
.address_size 64
.visible .entry v5_challenge(
 .param .u64 p0, .param .u64 p1, .param .u32 p2
) {
 .reg .pred %p; .reg .b32 %r<3>; .reg .b64 %rd<7>;
 ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
 ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2; ld.param.u32 %r2,[p2];
 mov.u32 %r1,%tid.x; setp.ge.u32 %p,%r1,%r2; @%p bra done;
 mul.wide.u32 %rd3,%r1,8; add.u64 %rd4,%rd1,%rd3; ld.global.s64 %rd5,[%rd4];
 shl.b64 %rd5,%rd5,1; add.s64 %rd6,%rd5,1; add.u64 %rd4,%rd2,%rd3;
 st.global.s64 [%rd4],%rd6;
done: ret;
}
.visible .entry v5_affine(
 .param .u64 p0, .param .u64 p1, .param .u32 p2,
 .param .s64 p3, .param .s64 p4
) {
 .reg .pred %p; .reg .b32 %r<3>; .reg .b64 %rd<9>;
 ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
 ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2; ld.param.u32 %r2,[p2];
 ld.param.s64 %rd5,[p3]; ld.param.s64 %rd6,[p4]; mov.u32 %r1,%tid.x;
 setp.ge.u32 %p,%r1,%r2; @%p bra done; mul.wide.u32 %rd3,%r1,8;
 add.u64 %rd4,%rd1,%rd3; ld.global.s64 %rd7,[%rd4]; mul.lo.s64 %rd7,%rd7,%rd5;
 add.s64 %rd7,%rd7,%rd6; add.u64 %rd4,%rd2,%rd3; st.global.s64 [%rd4],%rd7;
done: ret;
}
.visible .entry v5_elementwise(
 .param .u64 p0, .param .u64 p1, .param .u64 p2, .param .u32 p3
) {
 .reg .pred %p; .reg .b32 %r<3>; .reg .b64 %rd<9>;
 ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
 ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
 ld.param.u64 %rd3,[p2]; cvta.to.global.u64 %rd3,%rd3;
 ld.param.u32 %r2,[p3]; mov.u32 %r1,%tid.x; setp.ge.u32 %p,%r1,%r2; @%p bra done;
 mul.wide.u32 %rd4,%r1,8; add.u64 %rd5,%rd1,%rd4; add.u64 %rd6,%rd2,%rd4;
 add.u64 %rd7,%rd3,%rd4; ld.global.s64 %rd8,[%rd5]; ld.global.s64 %rd5,[%rd6];
 mul.lo.s64 %rd8,%rd8,%rd5; st.global.s64 [%rd7],%rd8;
done: ret;
}
.visible .entry v5_matvec(
 .param .u64 p0, .param .u64 p1, .param .u64 p2,
 .param .u32 p3, .param .u32 p4
) {
 .reg .pred %p; .reg .b32 %r<8>; .reg .s64 %rd<14>;
 ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
 ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
 ld.param.u64 %rd3,[p2]; cvta.to.global.u64 %rd3,%rd3;
 ld.param.u32 %r2,[p3]; ld.param.u32 %r3,[p4]; mov.u32 %r1,%tid.x;
 setp.ge.u32 %p,%r1,%r2; @%p bra done; mov.s64 %rd5,0; mov.u32 %r4,0;
 loop: setp.ge.u32 %p,%r4,%r3; @%p bra store; mad.lo.u32 %r5,%r1,%r3,%r4;
 mul.wide.u32 %rd6,%r5,8; add.u64 %rd7,%rd1,%rd6;
 ld.global.s64 %rd8,[%rd7]; mul.wide.u32 %rd9,%r4,8; add.u64 %rd10,%rd2,%rd9;
 ld.global.s64 %rd11,[%rd10]; mul.lo.s64 %rd12,%rd8,%rd11; add.s64 %rd5,%rd5,%rd12;
 add.u32 %r4,%r4,1; bra loop;
store: mul.wide.u32 %rd6,%r1,8; add.u64 %rd7,%rd3,%rd6; st.global.s64 [%rd7],%rd5;
done: ret;
}
.visible .entry v5_transpose(
 .param .u64 p0, .param .u64 p1, .param .u64 p2,
 .param .u32 p3, .param .u32 p4
) {
 .reg .pred %p; .reg .b32 %r<8>; .reg .s64 %rd<14>;
 ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
 ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
 ld.param.u64 %rd3,[p2]; cvta.to.global.u64 %rd3,%rd3;
 ld.param.u32 %r2,[p3]; ld.param.u32 %r3,[p4]; mov.u32 %r1,%tid.x;
 setp.ge.u32 %p,%r1,%r3; @%p bra done; mov.s64 %rd5,0; mov.u32 %r4,0;
 loop: setp.ge.u32 %p,%r4,%r2; @%p bra store; mad.lo.u32 %r5,%r4,%r3,%r1;
 mul.wide.u32 %rd6,%r5,8; add.u64 %rd7,%rd1,%rd6;
 ld.global.s64 %rd8,[%rd7]; mul.wide.u32 %rd9,%r4,8; add.u64 %rd10,%rd2,%rd9;
 ld.global.s64 %rd11,[%rd10]; mul.lo.s64 %rd12,%rd8,%rd11; add.s64 %rd5,%rd5,%rd12;
 add.u32 %r4,%r4,1; bra loop;
store: mul.wide.u32 %rd6,%r1,8; add.u64 %rd7,%rd3,%rd6; st.global.s64 [%rd7],%rd5;
done: ret;
}
"#;

    fn cuda_error(error: CuResult, operation: &str) -> ComputeError {
        ComputeError::NativeFailure(format!("CUDA {operation} returned error {error}"))
    }

    unsafe fn load_symbol<T: Copy>(handle: *mut c_void, name: &str) -> Option<T> {
        let name = CString::new(name).ok()?;
        // SAFETY: handle is a live loader handle and name is NUL terminated.
        let symbol = unsafe { dlsym(handle, name.as_ptr()) };
        if symbol.is_null() {
            None
        } else {
            // SAFETY: the requested T is the exact exported C ABI type.
            Some(unsafe { std::mem::transmute_copy(&symbol) })
        }
    }

    pub struct CudaRuntime {
        handle: *mut c_void,
        context: CuContext,
        module: CuModule,
        device_memory: u64,
        device_name: String,
        cu_ctx_destroy: CuCtxDestroy,
        cu_module_unload: CuModuleUnload,
        cu_mem_alloc: CuMemAlloc,
        cu_mem_free: CuMemFree,
        cu_memcpy_htod: CuMemcpyHtoD,
        cu_memcpy_dtoh: CuMemcpyDtoH,
        cu_launch_kernel: CuLaunchKernel,
        cu_ctx_synchronize: CuCtxSynchronize,
        challenge: CuFunction,
        affine: CuFunction,
        elementwise: CuFunction,
        matvec: CuFunction,
        transpose: CuFunction,
    }

    impl std::fmt::Debug for CudaRuntime {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("CudaRuntime")
                .field("device_name", &self.device_name)
                .field("device_memory", &self.device_memory)
                .finish_non_exhaustive()
        }
    }

    // SAFETY: the worker-local registry serializes access to one CUDA context.
    unsafe impl Send for CudaRuntime {}

    impl CudaRuntime {
        pub fn discover() -> Option<Self> {
            let library = CString::new("libcuda.so.1").ok()?;
            // SAFETY: only the operator-provided driver library is opened.
            let handle = unsafe { dlopen(library.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                return None;
            }
            let mut runtime = match Self::load(handle) {
                Ok(runtime) => runtime,
                Err(_) => {
                    // SAFETY: handle came from dlopen and is not retained.
                    unsafe { dlclose(handle) };
                    return None;
                }
            };
            let result =
                runtime.execute_values(&ComputeOperation::Challenge, &[vec![1_i64, 2, 3, 4]], 4);
            if result.as_deref() != Ok(&[3_i64, 5, 7, 9][..]) {
                drop(runtime);
                return None;
            }
            Some(runtime)
        }

        fn load(handle: *mut c_void) -> Result<Self, ComputeError> {
            // SAFETY: each symbol is cast to its matching CUDA ABI type.
            let cu_init: CuInit =
                unsafe { load_symbol(handle, "cuInit") }.ok_or(ComputeError::Unavailable)?;
            let cu_device_get_count: CuDeviceGetCount =
                unsafe { load_symbol(handle, "cuDeviceGetCount") }
                    .ok_or(ComputeError::Unavailable)?;
            let cu_device_get: CuDeviceGet =
                unsafe { load_symbol(handle, "cuDeviceGet") }.ok_or(ComputeError::Unavailable)?;
            let cu_device_get_name: CuDeviceGetName =
                unsafe { load_symbol(handle, "cuDeviceGetName") }
                    .ok_or(ComputeError::Unavailable)?;
            let cu_device_total_mem: CuDeviceTotalMem =
                unsafe { load_symbol(handle, "cuDeviceTotalMem_v2") }
                    .or_else(|| unsafe { load_symbol(handle, "cuDeviceTotalMem") })
                    .ok_or(ComputeError::Unavailable)?;
            let cu_ctx_create: CuCtxCreate = unsafe { load_symbol(handle, "cuCtxCreate_v2") }
                .or_else(|| unsafe { load_symbol(handle, "cuCtxCreate") })
                .ok_or(ComputeError::Unavailable)?;
            let cu_ctx_destroy: CuCtxDestroy = unsafe { load_symbol(handle, "cuCtxDestroy_v2") }
                .or_else(|| unsafe { load_symbol(handle, "cuCtxDestroy") })
                .ok_or(ComputeError::Unavailable)?;
            let cu_module_load_data: CuModuleLoadData =
                unsafe { load_symbol(handle, "cuModuleLoadData") }
                    .ok_or(ComputeError::Unavailable)?;
            let cu_module_unload: CuModuleUnload = unsafe { load_symbol(handle, "cuModuleUnload") }
                .ok_or(ComputeError::Unavailable)?;
            let cu_module_get_function: CuModuleGetFunction =
                unsafe { load_symbol(handle, "cuModuleGetFunction") }
                    .ok_or(ComputeError::Unavailable)?;
            let cu_mem_alloc: CuMemAlloc = unsafe { load_symbol(handle, "cuMemAlloc_v2") }
                .or_else(|| unsafe { load_symbol(handle, "cuMemAlloc") })
                .ok_or(ComputeError::Unavailable)?;
            let cu_mem_free: CuMemFree = unsafe { load_symbol(handle, "cuMemFree_v2") }
                .or_else(|| unsafe { load_symbol(handle, "cuMemFree") })
                .ok_or(ComputeError::Unavailable)?;
            let cu_memcpy_htod: CuMemcpyHtoD = unsafe { load_symbol(handle, "cuMemcpyHtoD_v2") }
                .or_else(|| unsafe { load_symbol(handle, "cuMemcpyHtoD") })
                .ok_or(ComputeError::Unavailable)?;
            let cu_memcpy_dtoh: CuMemcpyDtoH = unsafe { load_symbol(handle, "cuMemcpyDtoH_v2") }
                .or_else(|| unsafe { load_symbol(handle, "cuMemcpyDtoH") })
                .ok_or(ComputeError::Unavailable)?;
            let cu_launch_kernel: CuLaunchKernel = unsafe { load_symbol(handle, "cuLaunchKernel") }
                .ok_or(ComputeError::Unavailable)?;
            let cu_ctx_synchronize: CuCtxSynchronize =
                unsafe { load_symbol(handle, "cuCtxSynchronize") }
                    .ok_or(ComputeError::Unavailable)?;
            // SAFETY: valid output pointers and Driver API arguments.
            unsafe {
                if cu_init(0) != CUDA_SUCCESS {
                    return Err(ComputeError::Unavailable);
                }
            }
            let mut count = 0;
            // SAFETY: count is a valid output pointer.
            unsafe {
                if cu_device_get_count(&mut count) != CUDA_SUCCESS || count <= 0 {
                    return Err(ComputeError::Unavailable);
                }
            }
            let mut device = 0;
            // SAFETY: ordinal zero is valid after the count query.
            unsafe {
                if cu_device_get(&mut device, 0) != CUDA_SUCCESS {
                    return Err(ComputeError::Unavailable);
                }
            }
            let mut total_memory = 0usize;
            // SAFETY: total_memory is a valid output pointer.
            unsafe {
                if cu_device_total_mem(&mut total_memory, device) != CUDA_SUCCESS
                    || total_memory == 0
                {
                    return Err(ComputeError::Unavailable);
                }
            }
            let mut name_bytes = [0_i8; 256];
            // SAFETY: fixed-size name buffer and length are valid.
            unsafe {
                if cu_device_get_name(name_bytes.as_mut_ptr(), name_bytes.len() as c_int, device)
                    != CUDA_SUCCESS
                {
                    return Err(ComputeError::Unavailable);
                }
            }
            let name_len = name_bytes
                .iter()
                .position(|value| *value == 0)
                .unwrap_or(name_bytes.len());
            let device_name = name_bytes[..name_len]
                .iter()
                .map(|value| *value as u8 as char)
                .collect::<String>();
            let mut context = ptr::null_mut();
            // SAFETY: context is a valid output pointer and device is valid.
            unsafe {
                if cu_ctx_create(&mut context, 0, device) != CUDA_SUCCESS || context.is_null() {
                    return Err(ComputeError::Unavailable);
                }
            }
            let mut ptx = PTX.to_vec();
            ptx.push(0);
            let mut module = ptr::null_mut();
            // SAFETY: PTX is an immutable, NUL-terminated approved image.
            let module_result =
                unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) };
            if module_result != CUDA_SUCCESS || module.is_null() {
                // SAFETY: context was successfully created and is owned here.
                unsafe { cu_ctx_destroy(context) };
                return Err(cuda_error(module_result, "cuModuleLoadData"));
            }
            let function = |name: &str| -> Result<CuFunction, ComputeError> {
                let name = CString::new(name).map_err(|_| ComputeError::Unavailable)?;
                let mut function = ptr::null_mut();
                // SAFETY: module is live and output/name are valid.
                let result =
                    unsafe { cu_module_get_function(&mut function, module, name.as_ptr()) };
                if result != CUDA_SUCCESS || function.is_null() {
                    Err(cuda_error(result, "cuModuleGetFunction"))
                } else {
                    Ok(function)
                }
            };
            let functions = (
                function("v5_challenge"),
                function("v5_affine"),
                function("v5_elementwise"),
                function("v5_matvec"),
                function("v5_transpose"),
            );
            let (challenge, affine, elementwise, matvec, transpose) = match functions {
                (Ok(challenge), Ok(affine), Ok(elementwise), Ok(matvec), Ok(transpose)) => {
                    (challenge, affine, elementwise, matvec, transpose)
                }
                _ => {
                    // SAFETY: module/context were created by this instance.
                    unsafe {
                        cu_module_unload(module);
                        cu_ctx_destroy(context);
                    }
                    return Err(ComputeError::Unavailable);
                }
            };
            Ok(Self {
                handle,
                context,
                module,
                device_memory: total_memory as u64,
                device_name,
                cu_ctx_destroy,
                cu_module_unload,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_htod,
                cu_memcpy_dtoh,
                cu_launch_kernel,
                cu_ctx_synchronize,
                challenge,
                affine,
                elementwise,
                matvec,
                transpose,
            })
        }

        fn execute_values(
            &mut self,
            operation: &ComputeOperation,
            inputs: &[Vec<i64>],
            output_elements: u64,
        ) -> Result<Vec<i64>, ComputeError> {
            let output_len =
                usize::try_from(output_elements).map_err(|_| ComputeError::Overflow)?;
            let expected_matrix_elements = |rows: u16, cols: u16| {
                usize::from(rows)
                    .checked_mul(usize::from(cols))
                    .ok_or(ComputeError::Overflow)
            };
            match operation {
                ComputeOperation::Challenge
                | ComputeOperation::Affine { .. }
                | ComputeOperation::Gradient { .. } => {
                    if inputs.len() != 1 || inputs[0].len() != output_len {
                        return Err(ComputeError::InvalidTask(
                            "CUDA unary buffers do not match checked dimensions".to_string(),
                        ));
                    }
                }
                ComputeOperation::ElementwiseMultiply => {
                    if inputs.len() != 2
                        || inputs[0].len() != output_len
                        || inputs[1].len() != output_len
                    {
                        return Err(ComputeError::InvalidTask(
                            "CUDA elementwise buffers do not match checked dimensions".to_string(),
                        ));
                    }
                }
                ComputeOperation::MatrixVector { rows, cols } => {
                    if inputs.len() != 2
                        || inputs[0].len() != expected_matrix_elements(*rows, *cols)?
                        || inputs[1].len() != usize::from(*cols)
                        || output_len != usize::from(*rows)
                    {
                        return Err(ComputeError::InvalidTask(
                            "CUDA matrix-vector buffers do not match checked dimensions"
                                .to_string(),
                        ));
                    }
                }
                ComputeOperation::MatrixTransposeVector { rows, cols } => {
                    if inputs.len() != 2
                        || inputs[0].len() != expected_matrix_elements(*rows, *cols)?
                        || inputs[1].len() != usize::from(*rows)
                        || output_len != usize::from(*cols)
                    {
                        return Err(ComputeError::InvalidTask(
                            "CUDA transpose buffers do not match checked dimensions".to_string(),
                        ));
                    }
                }
            }
            let mut device_inputs = Vec::with_capacity(inputs.len());
            let mut device_output = 0;
            let result = (|| {
                for values in inputs {
                    let bytes = values
                        .len()
                        .checked_mul(std::mem::size_of::<i64>())
                        .ok_or(ComputeError::Overflow)?;
                    let mut device_pointer = 0;
                    // SAFETY: checked size, valid context and output pointer.
                    let result = unsafe { (self.cu_mem_alloc)(&mut device_pointer, bytes) };
                    if result != CUDA_SUCCESS {
                        return Err(cuda_error(result, "cuMemAlloc"));
                    }
                    // SAFETY: host slice remains live for this synchronous copy.
                    let result = unsafe {
                        (self.cu_memcpy_htod)(
                            device_pointer,
                            values.as_ptr().cast::<c_void>(),
                            bytes,
                        )
                    };
                    if result != CUDA_SUCCESS {
                        // SAFETY: this pointer was allocated by this context.
                        unsafe { (self.cu_mem_free)(device_pointer) };
                        return Err(cuda_error(result, "cuMemcpyHtoD"));
                    }
                    device_inputs.push(device_pointer);
                }
                let output_bytes = output_len
                    .checked_mul(std::mem::size_of::<i64>())
                    .ok_or(ComputeError::Overflow)?;
                // SAFETY: checked size and valid output pointer.
                let result = unsafe { (self.cu_mem_alloc)(&mut device_output, output_bytes) };
                if result != CUDA_SUCCESS {
                    return Err(cuda_error(result, "cuMemAlloc"));
                }
                let input_lengths = inputs.iter().map(Vec::len).collect::<Vec<_>>();
                self.launch(operation, &device_inputs, &input_lengths, device_output)?;
                let mut output = vec![0_i64; output_len];
                // SAFETY: output is correctly sized for the synchronous copy.
                let result = unsafe {
                    (self.cu_memcpy_dtoh)(
                        output.as_mut_ptr().cast::<c_void>(),
                        device_output,
                        output_bytes,
                    )
                };
                if result != CUDA_SUCCESS {
                    return Err(cuda_error(result, "cuMemcpyDtoH"));
                }
                Ok(output)
            })();
            // SAFETY: every nonzero pointer was allocated by this context.
            for pointer in device_inputs {
                unsafe { (self.cu_mem_free)(pointer) };
            }
            if device_output != 0 {
                unsafe { (self.cu_mem_free)(device_output) };
            }
            result
        }

        fn launch_raw(
            &mut self,
            function: CuFunction,
            count: u32,
            params: &mut [*mut c_void],
        ) -> Result<(), ComputeError> {
            if count == 0 {
                return Err(ComputeError::InvalidTask(
                    "CUDA task has zero launch elements".to_string(),
                ));
            }
            // SAFETY: function, dimensions and parameter pointers are valid
            // for the approved PTX entry point; the context is serialized.
            let result = unsafe {
                (self.cu_launch_kernel)(
                    function,
                    count.div_ceil(128),
                    1,
                    1,
                    128,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            if result != CUDA_SUCCESS {
                return Err(cuda_error(result, "cuLaunchKernel"));
            }
            // SAFETY: synchronization bounds the lifetime of parameter values.
            let result = unsafe { (self.cu_ctx_synchronize)() };
            if result != CUDA_SUCCESS {
                return Err(cuda_error(result, "cuCtxSynchronize"));
            }
            Ok(())
        }

        fn launch(
            &mut self,
            operation: &ComputeOperation,
            inputs: &[CuDevicePtr],
            input_lengths: &[usize],
            output: CuDevicePtr,
        ) -> Result<(), ComputeError> {
            let pointer = |value: &mut CuDevicePtr| (value as *mut CuDevicePtr).cast::<c_void>();
            let scalar_u32 = |value: &mut u32| (value as *mut u32).cast::<c_void>();
            let scalar_i64 = |value: &mut i64| (value as *mut i64).cast::<c_void>();
            match operation {
                ComputeOperation::Challenge => {
                    let mut input = *inputs.first().ok_or(ComputeError::InvalidTask(
                        "CUDA challenge input is missing".to_string(),
                    ))?;
                    let mut output = output;
                    let mut count =
                        u32::try_from(input_lengths[0]).map_err(|_| ComputeError::Overflow)?;
                    let mut params = vec![
                        pointer(&mut input),
                        pointer(&mut output),
                        scalar_u32(&mut count),
                    ];
                    self.launch_raw(self.challenge, count, &mut params)
                }
                ComputeOperation::Affine { coefficient, bias } => {
                    let mut input = *inputs.first().ok_or(ComputeError::InvalidTask(
                        "CUDA affine input is missing".to_string(),
                    ))?;
                    let mut output = output;
                    let mut count =
                        u32::try_from(input_lengths[0]).map_err(|_| ComputeError::Overflow)?;
                    let mut coefficient = *coefficient;
                    let mut bias = *bias;
                    let mut params = vec![
                        pointer(&mut input),
                        pointer(&mut output),
                        scalar_u32(&mut count),
                        scalar_i64(&mut coefficient),
                        scalar_i64(&mut bias),
                    ];
                    self.launch_raw(self.affine, count, &mut params)
                }
                ComputeOperation::Gradient { coefficient } => {
                    let mut input = *inputs.first().ok_or(ComputeError::InvalidTask(
                        "CUDA gradient input is missing".to_string(),
                    ))?;
                    let mut output = output;
                    let mut count =
                        u32::try_from(input_lengths[0]).map_err(|_| ComputeError::Overflow)?;
                    let mut coefficient = *coefficient;
                    let mut bias = 0_i64;
                    let mut params = vec![
                        pointer(&mut input),
                        pointer(&mut output),
                        scalar_u32(&mut count),
                        scalar_i64(&mut coefficient),
                        scalar_i64(&mut bias),
                    ];
                    self.launch_raw(self.affine, count, &mut params)
                }
                ComputeOperation::ElementwiseMultiply => {
                    let mut left = *inputs.first().ok_or(ComputeError::InvalidTask(
                        "CUDA elementwise left input is missing".to_string(),
                    ))?;
                    let mut right = *inputs.get(1).ok_or(ComputeError::InvalidTask(
                        "CUDA elementwise right input is missing".to_string(),
                    ))?;
                    let mut output = output;
                    let mut count =
                        u32::try_from(input_lengths[0]).map_err(|_| ComputeError::Overflow)?;
                    let mut params = vec![
                        pointer(&mut left),
                        pointer(&mut right),
                        pointer(&mut output),
                        scalar_u32(&mut count),
                    ];
                    self.launch_raw(self.elementwise, count, &mut params)
                }
                ComputeOperation::MatrixVector { rows, cols } => {
                    let mut weights = *inputs.first().ok_or(ComputeError::InvalidTask(
                        "CUDA matrix weights are missing".to_string(),
                    ))?;
                    let mut input = *inputs.get(1).ok_or(ComputeError::InvalidTask(
                        "CUDA matrix input is missing".to_string(),
                    ))?;
                    let mut output = output;
                    let mut rows = u32::from(*rows);
                    let mut cols = u32::from(*cols);
                    let mut params = vec![
                        pointer(&mut weights),
                        pointer(&mut input),
                        pointer(&mut output),
                        scalar_u32(&mut rows),
                        scalar_u32(&mut cols),
                    ];
                    self.launch_raw(self.matvec, rows, &mut params)
                }
                ComputeOperation::MatrixTransposeVector { rows, cols } => {
                    let mut weights = *inputs.first().ok_or(ComputeError::InvalidTask(
                        "CUDA transpose weights are missing".to_string(),
                    ))?;
                    let mut input = *inputs.get(1).ok_or(ComputeError::InvalidTask(
                        "CUDA transpose input is missing".to_string(),
                    ))?;
                    let mut output = output;
                    let mut rows = u32::from(*rows);
                    let mut cols = u32::from(*cols);
                    let mut params = vec![
                        pointer(&mut weights),
                        pointer(&mut input),
                        pointer(&mut output),
                        scalar_u32(&mut rows),
                        scalar_u32(&mut cols),
                    ];
                    self.launch_raw(self.transpose, cols, &mut params)
                }
            }
        }

        pub fn execute(
            &mut self,
            operation: &ComputeOperation,
            inputs: &[Vec<i64>],
            output_elements: u64,
        ) -> Result<Vec<i64>, ComputeError> {
            self.execute_values(operation, inputs, output_elements)
        }

        pub fn memory(&self) -> u64 {
            self.device_memory
        }

        pub fn name(&self) -> &str {
            &self.device_name
        }
    }

    impl Drop for CudaRuntime {
        fn drop(&mut self) {
            // SAFETY: handles are owned by this instance and released in
            // dependency order. Shutdown is best effort.
            unsafe {
                (self.cu_module_unload)(self.module);
                (self.cu_ctx_destroy)(self.context);
                dlclose(self.handle);
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
#[derive(Debug)]
struct OptionalAcceleratorBackend {
    capabilities: BackendCapabilities,
    emulate: bool,
    queued: u16,
    reserved_bytes: u64,
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    native_cuda: Option<cuda_native::CudaRuntime>,
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
impl OptionalAcceleratorBackend {
    fn new(kind: BackendKind, runtime_available: bool, runtime_version: &str) -> Self {
        let emulate = std::env::var_os("INTELLIGENCE_V5_EMULATE_ACCELERATORS").is_some();
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let native_cuda = if kind == BackendKind::Cuda && runtime_available {
            cuda_native::CudaRuntime::discover()
        } else {
            None
        };
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let native_ready = native_cuda.is_some();
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let native_ready = false;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let native_memory = native_cuda
            .as_ref()
            .map(cuda_native::CudaRuntime::memory)
            .unwrap_or(0);
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let native_memory = 0;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let native_name = native_cuda
            .as_ref()
            .map(|runtime| runtime.name().to_string());
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let native_name = None;

        // Driver presence alone is not a usable backend.  The CUDA path is
        // advertised only after the bounded native PTX challenge succeeds;
        // other accelerator families remain explicit emulation/build
        // boundaries until their native runtime is available.
        let ready = native_ready || emulate;
        let available = if native_ready {
            native_memory
        } else if emulate {
            512 * 1024 * 1024
        } else {
            1
        };
        Self {
            capabilities: BackendCapabilities {
                kind,
                runtime_version: runtime_version.to_string(),
                device_count: 1,
                device_memory_bytes: available,
                available_memory_bytes: available,
                formats: vec![NumericFormat::F32],
                max_tensor_elements: MAX_TASK_ELEMENTS,
                features: vec![
                    ComputeFeature::MatrixMultiply,
                    ComputeFeature::TensorParallel,
                    ComputeFeature::PipelineStage,
                    ComputeFeature::DeterministicKernel,
                ],
                device_architecture: native_name,
                driver_available: runtime_available,
                runtime_available: ready,
                peer_to_peer: false,
                unified_memory: false,
                max_concurrent_tasks: 1,
                safety_margin_permille: 150,
                health: if ready {
                    BackendHealth::Ready
                } else {
                    BackendHealth::Unavailable
                },
                physical_verified: native_ready,
                observed_successes: 0,
                observed_failures: 0,
            },
            emulate,
            queued: 0,
            reserved_bytes: 0,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            native_cuda,
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
impl ComputeBackend for OptionalAcceleratorBackend {
    fn kind(&self) -> BackendKind {
        self.capabilities.kind
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn validate(&self, task: &ComputeTask) -> Result<(), ComputeError> {
        validate_requirements(&self.capabilities, task)
    }

    fn prepare(&mut self, task: &ComputeTask) -> Result<PreparedTask, ComputeError> {
        self.validate(task)?;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let native_ready = self.native_cuda.is_some();
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let native_ready = false;
        if !self.emulate && !native_ready {
            return Err(ComputeError::Unavailable);
        }
        if self.queued >= self.capabilities.max_concurrent_tasks {
            return Err(ComputeError::QueueFull);
        }
        let reserved_bytes = task
            .requirements
            .required_memory_bytes
            .checked_add(
                task.input_elements
                    .checked_add(task.output_elements)
                    .ok_or(ComputeError::Overflow)?
                    .checked_mul(8)
                    .ok_or(ComputeError::Overflow)?,
            )
            .ok_or(ComputeError::Overflow)?;
        let safe_memory = self
            .capabilities
            .available_memory_bytes
            .saturating_mul(u64::from(
                1000_u16.saturating_sub(self.capabilities.safety_margin_permille),
            ))
            / 1000;
        if self.reserved_bytes.saturating_add(reserved_bytes) > safe_memory {
            return Err(ComputeError::OutOfMemory);
        }
        self.queued = self.queued.saturating_add(1);
        self.reserved_bytes = self.reserved_bytes.saturating_add(reserved_bytes);
        Ok(PreparedTask {
            task: task.clone(),
            backend: self.kind(),
            reserved_bytes,
        })
    }

    fn execute(
        &mut self,
        prepared: &PreparedTask,
        inputs: &[ComputeInput],
    ) -> Result<ComputeOutput, ComputeError> {
        let started = Instant::now();
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(native) = self.native_cuda.as_mut() {
            let values = inputs
                .iter()
                .map(|input| input.values.clone())
                .collect::<Vec<_>>();
            let output = native.execute(
                &prepared.task.operation,
                &values,
                prepared.task.output_elements,
            )?;
            self.queued = self.queued.saturating_sub(1);
            self.reserved_bytes = self.reserved_bytes.saturating_sub(prepared.reserved_bytes);
            return Ok(ComputeOutput {
                values: output,
                format: NumericFormat::F32,
                backend: self.kind(),
                elapsed_micros: started.elapsed().as_micros().max(1) as u64,
                converted: false,
            });
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let native_ready = self.native_cuda.is_some();
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let native_ready = false;
        if !self.emulate || native_ready {
            return Err(ComputeError::Unavailable);
        }
        // Portable feature builds deliberately use the same checked reference
        // operation while the native runtime adapter is unavailable.  This is
        // an explicit emulation path and never marks physical verification.
        let mut cpu = CpuBackend::new();
        // Keep the emulation execution inside the same validated task boundary
        // while replacing only the local backend selector.  The original task
        // remains the caller's CUDA/ROCm/Metal contract; the reference CPU
        // implementation must not reject it merely because it is emulating
        // the approved operation.
        let mut reference_task = prepared.task.clone();
        reference_task.requirements.required_backend = Some(BackendKind::Cpu);
        reference_task.requirements.allowed_backends = vec![BackendKind::Cpu];
        let mut output = cpu.execute(
            &PreparedTask {
                task: reference_task,
                backend: BackendKind::Cpu,
                reserved_bytes: prepared.reserved_bytes,
            },
            inputs,
        )?;
        self.queued = self.queued.saturating_sub(1);
        self.reserved_bytes = self.reserved_bytes.saturating_sub(prepared.reserved_bytes);
        output.backend = self.kind();
        output.elapsed_micros = started.elapsed().as_micros().max(1) as u64;
        Ok(output)
    }

    fn synchronize(&mut self) -> Result<(), ComputeError> {
        self.queued = 0;
        self.reserved_bytes = 0;
        Ok(())
    }

    fn health(&self) -> BackendHealth {
        self.capabilities.health
    }
}

pub struct BackendRegistry {
    backends: HashMap<BackendKind, Box<dyn ComputeBackend>>,
    evidence: HashMap<BackendKind, CapabilityEvidenceRecord>,
    health_overrides: HashMap<BackendKind, BackendHealth>,
}

impl std::fmt::Debug for BackendRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BackendRegistry")
            .field("backends", &self.capabilities())
            .field("evidence", &self.evidence)
            .field("health_overrides", &self.health_overrides)
            .finish()
    }
}

impl BackendRegistry {
    pub fn discover() -> Self {
        let mut registry = Self {
            backends: HashMap::new(),
            evidence: HashMap::new(),
            health_overrides: HashMap::new(),
        };
        registry.register(Box::new(CpuBackend::new()));
        #[cfg(feature = "cuda")]
        registry.register(Box::new(OptionalAcceleratorBackend::new(
            BackendKind::Cuda,
            std::path::Path::new("/dev/nvidiactl").exists()
                || std::path::Path::new("/usr/lib/libcuda.so.1").exists(),
            "cuda-driver-dynamic",
        )));
        #[cfg(feature = "rocm")]
        registry.register(Box::new(OptionalAcceleratorBackend::new(
            BackendKind::Rocm,
            std::path::Path::new("/dev/kfd").exists()
                && (std::path::Path::new("/opt/rocm").exists()
                    || std::path::Path::new("/usr/lib/libamdhip64.so").exists()),
            "rocm-hip-dynamic",
        )));
        #[cfg(feature = "metal")]
        registry.register(Box::new(OptionalAcceleratorBackend::new(
            BackendKind::Metal,
            cfg!(target_os = "macos"),
            "metal-system",
        )));
        registry
    }

    pub fn register(&mut self, backend: Box<dyn ComputeBackend>) {
        let kind = backend.kind();
        self.health_overrides.remove(&kind);
        self.backends.insert(kind, backend);
    }

    pub fn capabilities(&self) -> Vec<BackendCapabilities> {
        let mut values = self
            .backends
            .values()
            .map(|backend| {
                let mut capability = backend.capabilities();
                if let Some(health) = self.health_overrides.get(&capability.kind) {
                    capability.health = *health;
                    if *health == BackendHealth::Unavailable {
                        capability.runtime_available = false;
                    }
                }
                if let Some(evidence) = self.evidence.get(&capability.kind) {
                    capability.observed_successes = evidence.successful_challenges;
                    capability.observed_failures = evidence.failed_challenges;
                }
                capability
            })
            .collect::<Vec<_>>();
        values.sort_by_key(|value| value.kind as u8);
        values
    }

    pub fn advertised_capabilities(&self) -> Vec<BackendCapabilities> {
        self.capabilities()
            .into_iter()
            .filter(|capability| {
                capability.runtime_available && capability.health != BackendHealth::Unavailable
            })
            .collect()
    }

    pub fn health(&self, kind: BackendKind) -> BackendHealth {
        if let Some(health) = self.health_overrides.get(&kind) {
            return *health;
        }
        self.backends
            .get(&kind)
            .map(|backend| backend.health())
            .unwrap_or(BackendHealth::Unavailable)
    }

    pub fn validate(&self, kind: BackendKind, task: &ComputeTask) -> Result<(), ComputeError> {
        self.backends
            .get(&kind)
            .ok_or(ComputeError::Unavailable)?
            .validate(task)
    }

    pub fn execute(
        &mut self,
        kind: BackendKind,
        task: &ComputeTask,
        inputs: &[ComputeInput],
    ) -> Result<ComputeOutput, ComputeError> {
        if self.health(kind) == BackendHealth::Unavailable {
            return Err(ComputeError::Unavailable);
        }
        let result = {
            let backend = self
                .backends
                .get_mut(&kind)
                .ok_or(ComputeError::Unavailable)?;
            backend.validate(task)?;
            let prepared = backend.prepare(task)?;
            let result = match backend.execute(&prepared, inputs) {
                Ok(output)
                    if output.elapsed_micros
                        > u64::from(task.deadline_ms).saturating_mul(1_000) =>
                {
                    Err(ComputeError::Timeout)
                }
                result => result,
            };
            if result.is_err() {
                let _ = backend.synchronize();
            }
            result
        };
        let backend_fault = matches!(
            &result,
            Err(ComputeError::Unavailable)
                | Err(ComputeError::NativeFailure(_))
                | Err(ComputeError::OutOfMemory)
                | Err(ComputeError::Timeout)
        );
        if backend_fault {
            self.health_overrides
                .insert(kind, BackendHealth::Unavailable);
        }
        result
    }

    /// Apply a bounded worker-local health transition.  The override is
    /// included in the next signed capability advertisement and is not a
    /// global reputation record.
    pub fn set_health(&mut self, kind: BackendKind, health: BackendHealth) {
        self.health_overrides.insert(kind, health);
    }

    pub fn synchronize(&mut self, kind: BackendKind) -> Result<(), ComputeError> {
        self.backends
            .get_mut(&kind)
            .ok_or(ComputeError::Unavailable)?
            .synchronize()
    }

    pub fn challenge(
        &mut self,
        challenge: &CapabilityChallenge,
        now_secs: u64,
    ) -> CapabilityChallengeResult {
        let started = Instant::now();
        let input_elements = challenge
            .input_elements
            .min(MAX_TASK_ELEMENTS as u32)
            .max(1);
        let requirements = ComputeRequirements {
            task_kind: ComputeTaskKind::CapabilityChallenge,
            required_backend: Some(challenge.backend),
            allowed_backends: vec![challenge.backend],
            required_formats: vec![challenge.format],
            required_memory_bytes: u64::from(input_elements).saturating_mul(8),
            max_tensor_elements: u64::from(input_elements),
            required_features: vec![ComputeFeature::DeterministicKernel],
            kernel_id: "v5.capability.challenge.v1".to_string(),
            kernel_version: 1,
            fallback_backends: Vec::new(),
            fallback_allowed: false,
        };
        let task = ComputeTask {
            operation: ComputeOperation::Challenge,
            requirements,
            input_elements: u64::from(input_elements),
            output_elements: u64::from(input_elements),
            graph_generation: 1,
            model_generation: 1,
            shard_id: 0,
            deadline_ms: challenge.deadline_ms,
        };
        let values = (0..input_elements)
            .map(|index| (challenge.seed as i64).wrapping_add(i64::from(index)))
            .collect::<Vec<_>>();
        let result = self.execute(
            challenge.backend,
            &task,
            &[ComputeInput {
                values,
                format: challenge.format,
            }],
        );
        match result {
            Ok(output) => {
                let expected = (0..input_elements)
                    .map(|index| {
                        (challenge.seed as i64)
                            .wrapping_add(i64::from(index))
                            .saturating_mul(2)
                            .saturating_add(1)
                    })
                    .collect::<Vec<_>>();
                if output.values != expected {
                    CapabilityChallengeResult {
                        challenge_id: challenge.challenge_id,
                        worker: challenge.worker,
                        backend: challenge.backend,
                        success: false,
                        output_hash: intelligence_protocol::ArtifactId::default(),
                        elapsed_micros: started.elapsed().as_micros().max(1) as u64,
                        allocated_bytes: u64::from(input_elements).saturating_mul(8),
                        health: self.health(challenge.backend),
                        error: Some("capability challenge output mismatch".to_string()),
                    }
                } else {
                    let mut hasher = Hasher::new();
                    for value in output.values {
                        hasher.update(&value.to_le_bytes());
                    }
                    let hash = intelligence_protocol::ArtifactId::from_bytes(
                        *hasher.finalize().as_bytes(),
                    );
                    CapabilityChallengeResult {
                        challenge_id: challenge.challenge_id,
                        worker: challenge.worker,
                        backend: challenge.backend,
                        success: true,
                        output_hash: hash,
                        elapsed_micros: started.elapsed().as_micros().max(1) as u64,
                        allocated_bytes: u64::from(input_elements).saturating_mul(8),
                        health: self.health(challenge.backend),
                        error: None,
                    }
                }
            }
            Err(error) => CapabilityChallengeResult {
                challenge_id: challenge.challenge_id,
                worker: challenge.worker,
                backend: challenge.backend,
                success: false,
                output_hash: intelligence_protocol::ArtifactId::default(),
                elapsed_micros: started.elapsed().as_micros().max(1) as u64,
                allocated_bytes: 0,
                health: self.health(challenge.backend),
                error: Some(error.to_string()),
            },
        }
        .tap(|result| {
            let entry = self.evidence.entry(challenge.backend).or_insert_with(|| {
                CapabilityEvidenceRecord {
                    worker: challenge.worker,
                    backend: challenge.backend,
                    observed_at: now_secs,
                    successful_challenges: 0,
                    failed_challenges: 0,
                    successful_tasks: 0,
                    failed_tasks: 0,
                    last_throughput_micros: 1,
                    last_error: None,
                }
            });
            entry.observed_at = now_secs;
            entry.last_throughput_micros = result.elapsed_micros;
            if result.success {
                entry.successful_challenges = entry.successful_challenges.saturating_add(1);
                entry.last_error = None;
            } else {
                entry.failed_challenges = entry.failed_challenges.saturating_add(1);
                entry.last_error = result.error.clone();
            }
        })
    }

    pub fn evidence(&self) -> Vec<CapabilityEvidenceRecord> {
        self.evidence.values().cloned().collect()
    }
}

trait Tap: Sized {
    fn tap<F: FnOnce(&Self)>(self, function: F) -> Self {
        function(&self);
        self
    }
}

impl<T> Tap for T {}

fn host_available_memory() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let mut parts = line.split_whitespace();
                if parts.next() == Some("MemAvailable:") {
                    parts
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|value| value.saturating_mul(1024))
                } else {
                    None
                }
            })
        })
        .filter(|value| *value > 0)
        .unwrap_or(256 * 1024 * 1024)
        .min(1 << 40)
}

/// Construct a bounded capability record for a planner profile.  This is a
/// claim when it comes from a remote advertisement; only the local registry's
/// CPU record is physical evidence on a CPU-only machine.
pub fn reference_backend_capability(
    kind: BackendKind,
    memory_bytes: u64,
    physical_verified: bool,
) -> BackendCapabilities {
    let memory_bytes = memory_bytes.max(1);
    BackendCapabilities {
        kind,
        runtime_version: "reference-profile-1".to_string(),
        device_count: 1,
        device_memory_bytes: memory_bytes,
        available_memory_bytes: memory_bytes,
        formats: vec![NumericFormat::F32],
        max_tensor_elements: MAX_TASK_ELEMENTS,
        features: vec![
            ComputeFeature::MatrixMultiply,
            ComputeFeature::TensorParallel,
            ComputeFeature::PipelineStage,
            ComputeFeature::DeterministicKernel,
        ],
        device_architecture: None,
        driver_available: physical_verified,
        runtime_available: true,
        peer_to_peer: false,
        unified_memory: matches!(kind, BackendKind::Cpu | BackendKind::Metal),
        max_concurrent_tasks: 1,
        safety_margin_permille: 100,
        health: BackendHealth::Ready,
        physical_verified,
        observed_successes: 0,
        observed_failures: 0,
    }
}

pub fn backend_can_run(
    capabilities: &BackendCapabilities,
    requirements: &ComputeRequirements,
) -> bool {
    let task = ComputeTask {
        operation: ComputeOperation::Challenge,
        requirements: requirements.clone(),
        input_elements: 1,
        output_elements: 1,
        graph_generation: 1,
        model_generation: 1,
        shard_id: 0,
        deadline_ms: 1_000,
    };
    validate_requirements(capabilities, &task).is_ok()
}

pub fn backend_satisfies_requirements(
    capabilities: &BackendCapabilities,
    requirements: &ComputeRequirements,
) -> bool {
    if !capabilities.runtime_available || capabilities.health == BackendHealth::Unavailable {
        return false;
    }
    // A failed challenge with no successful observation is enough to remove a
    // self-claimed backend from this job's eligible set.  A later successful
    // challenge can restore eligibility; this is bounded local evidence, not a
    // global reputation claim.
    if capabilities.observed_failures > 0 && capabilities.observed_successes == 0 {
        return false;
    }
    if requirements.required_backend != Some(capabilities.kind)
        && !requirements.allowed_backends.contains(&capabilities.kind)
    {
        return false;
    }
    let safe_memory = capabilities
        .available_memory_bytes
        .saturating_mul(u64::from(
            1000_u16.saturating_sub(capabilities.safety_margin_permille),
        ))
        / 1000;
    requirements.required_memory_bytes <= safe_memory
        && requirements.max_tensor_elements <= capabilities.max_tensor_elements
        && requirements
            .required_formats
            .iter()
            .all(|format| capabilities.formats.contains(format))
        && requirements
            .required_features
            .iter()
            .all(|feature| capabilities.features.contains(feature))
}

pub fn default_requirements(
    task_kind: ComputeTaskKind,
    backend: BackendKind,
    elements: u64,
    kernel_id: &str,
) -> ComputeRequirements {
    ComputeRequirements {
        task_kind,
        required_backend: Some(backend),
        allowed_backends: vec![backend],
        required_formats: vec![NumericFormat::F32],
        required_memory_bytes: elements.saturating_mul(8).max(8),
        max_tensor_elements: elements.max(1),
        required_features: vec![ComputeFeature::DeterministicKernel],
        kernel_id: kernel_id.to_string(),
        kernel_version: 1,
        fallback_backends: Vec::new(),
        fallback_allowed: false,
    }
}

pub fn new_challenge(
    job_id: intelligence_protocol::JobId,
    worker: intelligence_protocol::NodeId,
    backend: BackendKind,
    format: NumericFormat,
    seed: u64,
) -> CapabilityChallenge {
    CapabilityChallenge {
        challenge_id: RequestId::from_bytes(
            seed.to_le_bytes().repeat(2).try_into().unwrap_or([0; 16]),
        ),
        job_id,
        worker,
        backend,
        task_kind: ComputeTaskKind::CapabilityChallenge,
        format,
        input_elements: 16,
        seed,
        deadline_ms: 5_000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use intelligence_protocol::{BackendHealth, ComputeFeature, JobId, NodeId};

    struct FalseCapabilityBackend {
        capabilities: BackendCapabilities,
    }

    impl ComputeBackend for FalseCapabilityBackend {
        fn kind(&self) -> BackendKind {
            self.capabilities.kind
        }

        fn capabilities(&self) -> BackendCapabilities {
            self.capabilities.clone()
        }

        fn validate(&self, task: &ComputeTask) -> Result<(), ComputeError> {
            validate_requirements(&self.capabilities, task)
        }

        fn prepare(&mut self, task: &ComputeTask) -> Result<PreparedTask, ComputeError> {
            self.validate(task)?;
            Ok(PreparedTask {
                task: task.clone(),
                backend: self.kind(),
                reserved_bytes: task.requirements.required_memory_bytes,
            })
        }

        fn execute(
            &mut self,
            prepared: &PreparedTask,
            _inputs: &[ComputeInput],
        ) -> Result<ComputeOutput, ComputeError> {
            Ok(ComputeOutput {
                values: vec![0; usize::try_from(prepared.task.output_elements).unwrap_or(0)],
                format: NumericFormat::F32,
                backend: self.kind(),
                elapsed_micros: 1,
                converted: false,
            })
        }

        fn synchronize(&mut self) -> Result<(), ComputeError> {
            Ok(())
        }

        fn health(&self) -> BackendHealth {
            self.capabilities.health
        }
    }

    fn task(kind: BackendKind) -> ComputeTask {
        ComputeTask {
            operation: ComputeOperation::MatrixVector { rows: 2, cols: 2 },
            requirements: ComputeRequirements {
                task_kind: ComputeTaskKind::TensorForward,
                required_backend: Some(kind),
                allowed_backends: vec![kind],
                required_formats: vec![NumericFormat::F32],
                required_memory_bytes: 64,
                max_tensor_elements: 8,
                required_features: vec![ComputeFeature::DeterministicKernel],
                kernel_id: "test".to_string(),
                kernel_version: 1,
                fallback_backends: Vec::new(),
                fallback_allowed: false,
            },
            input_elements: 2,
            output_elements: 2,
            graph_generation: 1,
            model_generation: 1,
            shard_id: 0,
            deadline_ms: 1_000,
        }
    }

    #[test]
    fn cpu_reference_runs_through_backend_boundary() {
        let mut registry = BackendRegistry::discover();
        let output = registry
            .execute(
                BackendKind::Cpu,
                &task(BackendKind::Cpu),
                &[
                    ComputeInput {
                        values: vec![1, 0, 0, 1],
                        format: NumericFormat::F32,
                    },
                    ComputeInput {
                        values: vec![2, 4],
                        format: NumericFormat::F32,
                    },
                ],
            )
            .unwrap();
        assert_eq!(output.values, vec![2, 4]);
        assert_eq!(output.backend, BackendKind::Cpu);
    }

    #[test]
    fn native_cuda_reference_ops_match_cpu_when_available() {
        let mut registry = BackendRegistry::discover();
        let cuda_ready = registry
            .capabilities()
            .into_iter()
            .find(|capability| capability.kind == BackendKind::Cuda)
            .is_some_and(|capability| {
                capability.runtime_available
                    && capability.health == BackendHealth::Ready
                    && capability.physical_verified
            });
        if !cuda_ready {
            return;
        }

        let matrix = registry
            .execute(
                BackendKind::Cuda,
                &task(BackendKind::Cuda),
                &[
                    ComputeInput {
                        values: vec![1, 0, 0, 1],
                        format: NumericFormat::F32,
                    },
                    ComputeInput {
                        values: vec![2, 4],
                        format: NumericFormat::F32,
                    },
                ],
            )
            .expect("native CUDA matrix-vector operation");
        assert_eq!(matrix.values, vec![2, 4]);

        let mut affine_task = task(BackendKind::Cuda);
        affine_task.operation = ComputeOperation::Affine {
            coefficient: 2,
            bias: 1,
        };
        let affine = registry
            .execute(
                BackendKind::Cuda,
                &affine_task,
                &[ComputeInput {
                    values: vec![3, 4],
                    format: NumericFormat::F32,
                }],
            )
            .expect("native CUDA affine operation");
        assert_eq!(affine.values, vec![7, 9]);
        eprintln!(
            "CUDA reference fixture: matrix_vector={}us affine={}us",
            matrix.elapsed_micros, affine.elapsed_micros
        );
    }

    #[test]
    fn native_cuda_training_fixture_decreases_loss_when_available() {
        let mut registry = BackendRegistry::discover();
        let cuda_ready = registry
            .capabilities()
            .into_iter()
            .find(|capability| capability.kind == BackendKind::Cuda)
            .is_some_and(|capability| {
                capability.runtime_available
                    && capability.health == BackendHealth::Ready
                    && capability.physical_verified
            });
        if !cuda_ready {
            return;
        }

        let training_task =
            |operation: ComputeOperation, input_count: u64, output_count: u64| -> ComputeTask {
                let mut training = task(BackendKind::Cuda);
                training.operation = operation;
                training.requirements.task_kind = ComputeTaskKind::TrainingStep;
                training.requirements.required_memory_bytes = 128;
                training.requirements.max_tensor_elements = input_count.max(output_count);
                training.input_elements = input_count;
                training.output_elements = output_count;
                training
            };

        let target = 6_i64;
        let input = 2_i64;
        let mut weight = 1_i64;
        let initial_prediction = registry
            .execute(
                BackendKind::Cuda,
                &training_task(
                    ComputeOperation::Affine {
                        coefficient: weight,
                        bias: 0,
                    },
                    1,
                    1,
                ),
                &[ComputeInput {
                    values: vec![input],
                    format: NumericFormat::F32,
                }],
            )
            .expect("native CUDA training forward pass")
            .values[0];
        let initial_loss = (initial_prediction - target).pow(2);

        let error = initial_prediction - target;
        let product = registry
            .execute(
                BackendKind::Cuda,
                &training_task(ComputeOperation::ElementwiseMultiply, 1, 1),
                &[
                    ComputeInput {
                        values: vec![error],
                        format: NumericFormat::F32,
                    },
                    ComputeInput {
                        values: vec![input],
                        format: NumericFormat::F32,
                    },
                ],
            )
            .expect("native CUDA training error-input product")
            .values[0];
        let gradient = registry
            .execute(
                BackendKind::Cuda,
                &training_task(ComputeOperation::Gradient { coefficient: 1 }, 1, 1),
                &[ComputeInput {
                    values: vec![product],
                    format: NumericFormat::F32,
                }],
            )
            .expect("native CUDA training gradient")
            .values[0];
        weight -= gradient / 4;

        let final_prediction = registry
            .execute(
                BackendKind::Cuda,
                &training_task(
                    ComputeOperation::Affine {
                        coefficient: weight,
                        bias: 0,
                    },
                    1,
                    1,
                ),
                &[ComputeInput {
                    values: vec![input],
                    format: NumericFormat::F32,
                }],
            )
            .expect("native CUDA training final forward pass")
            .values[0];
        let final_loss = (final_prediction - target).pow(2);
        eprintln!(
            "CUDA training fixture: initial_loss={} final_loss={} final_weight={}",
            initial_loss, final_loss, weight
        );
        assert!(final_loss < initial_loss);
        assert_eq!(final_prediction, target);
    }

    #[test]
    fn cpu_backend_boundary_overhead_is_bounded() {
        let mut direct = CpuBackend::new();
        let direct_task = task(BackendKind::Cpu);
        let direct_input = [
            ComputeInput {
                values: vec![1, 0, 0, 1],
                format: NumericFormat::F32,
            },
            ComputeInput {
                values: vec![2, 4],
                format: NumericFormat::F32,
            },
        ];
        let iterations = 2_000_u32;
        let direct_started = Instant::now();
        for _ in 0..iterations {
            let prepared = direct.prepare(&direct_task).expect("direct CPU prepare");
            let output = direct
                .execute(&prepared, &direct_input)
                .expect("direct CPU execute");
            std::hint::black_box(output);
        }
        let direct_elapsed = direct_started.elapsed();

        let mut registry = BackendRegistry::discover();
        let boundary_started = Instant::now();
        for _ in 0..iterations {
            let output = registry
                .execute(BackendKind::Cpu, &direct_task, &direct_input)
                .expect("CPU boundary execute");
            std::hint::black_box(output);
        }
        let boundary_elapsed = boundary_started.elapsed();
        let allowance = direct_elapsed
            .saturating_mul(100)
            .saturating_add(std::time::Duration::from_millis(5));
        eprintln!(
            "CPU backend microbenchmark: direct={:?} boundary={:?} ratio={:.2}x",
            direct_elapsed,
            boundary_elapsed,
            boundary_elapsed.as_secs_f64() / direct_elapsed.as_secs_f64().max(f64::EPSILON)
        );
        assert!(boundary_elapsed <= allowance);
    }

    #[test]
    fn unsupported_format_and_backend_are_rejected() {
        let mut registry = BackendRegistry::discover();
        let mut invalid = task(BackendKind::Cpu);
        invalid.requirements.required_formats = vec![NumericFormat::Bf16];
        assert!(matches!(
            registry.validate(BackendKind::Cpu, &invalid),
            Err(ComputeError::Unsupported)
        ));
        let cuda_ready = registry
            .capabilities()
            .into_iter()
            .find(|capability| capability.kind == BackendKind::Cuda)
            .is_some_and(|capability| {
                capability.runtime_available && capability.health == BackendHealth::Ready
            });
        let result = registry.execute(BackendKind::Cuda, &task(BackendKind::Cuda), &[]);
        if cuda_ready {
            assert!(matches!(result, Err(ComputeError::InvalidTask(_))));
        } else {
            assert!(matches!(result, Err(ComputeError::Unavailable)));
        }
    }

    #[test]
    fn capability_challenge_matches_runtime_evidence() {
        let mut registry = BackendRegistry::discover();
        let challenge = new_challenge(
            JobId::from_bytes([1; 16]),
            NodeId::from_bytes([2; 32]),
            BackendKind::Cuda,
            NumericFormat::F32,
            7,
        );
        let result = registry.challenge(&challenge, 1);
        let cuda_ready = registry
            .capabilities()
            .into_iter()
            .find(|capability| capability.kind == BackendKind::Cuda)
            .is_some_and(|capability| {
                capability.runtime_available && capability.health == BackendHealth::Ready
            });
        assert_eq!(result.success, cuda_ready);
        assert_eq!(
            result.health,
            if cuda_ready {
                BackendHealth::Ready
            } else {
                BackendHealth::Unavailable
            }
        );
    }

    #[test]
    fn false_capability_challenge_removes_backend_from_eligibility() {
        let mut registry = BackendRegistry::discover();
        registry.register(Box::new(FalseCapabilityBackend {
            capabilities: reference_backend_capability(BackendKind::Cuda, 1 << 30, false),
        }));
        let challenge = new_challenge(
            JobId::from_bytes([3; 16]),
            NodeId::from_bytes([4; 32]),
            BackendKind::Cuda,
            NumericFormat::F32,
            11,
        );
        let result = registry.challenge(&challenge, 2);
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("output mismatch"))
        );
        let requirements = default_requirements(
            ComputeTaskKind::TensorForward,
            BackendKind::Cuda,
            4,
            "v5.test",
        );
        let capabilities = registry
            .capabilities()
            .into_iter()
            .find(|capability| capability.kind == BackendKind::Cuda)
            .expect("false capability is registered");
        assert!(!backend_satisfies_requirements(
            &capabilities,
            &requirements
        ));
    }

    #[test]
    fn backend_health_override_removes_and_restores_local_eligibility() {
        let mut registry = BackendRegistry::discover();
        registry.set_health(BackendKind::Cpu, BackendHealth::Unavailable);
        assert_eq!(
            registry.health(BackendKind::Cpu),
            BackendHealth::Unavailable
        );
        assert!(
            registry
                .advertised_capabilities()
                .iter()
                .all(|capability| capability.kind != BackendKind::Cpu)
        );
        assert!(matches!(
            registry.execute(BackendKind::Cpu, &task(BackendKind::Cpu), &[]),
            Err(ComputeError::Unavailable)
        ));
        registry.set_health(BackendKind::Cpu, BackendHealth::Ready);
        assert!(
            registry
                .advertised_capabilities()
                .iter()
                .any(|capability| capability.kind == BackendKind::Cpu)
        );
    }
}
