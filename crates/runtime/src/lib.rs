//! Bounded local execution and resource accounting.

use blake3::hash;
use intelligence_protocol::{
    ArtifactId, JobId, JobRequest, MAX_ARTIFACT_CHUNK, MAX_JOB_INPUT, MAX_JOB_OUTPUT,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io,
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
    sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore, mpsc},
    time::sleep,
};

const MAX_PROCESS_ARGS: usize = 64;
const MAX_PROCESS_ENV: usize = 32;
const MAX_ERROR_BYTES: usize = 8192;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub max_queued_jobs: usize,
    pub max_concurrent_jobs: usize,
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub default_timeout_ms: u64,
    pub process_memory_bytes: u64,
    pub process_cpu_seconds: u64,
    pub work_dir: PathBuf,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_queued_jobs: 32,
            max_concurrent_jobs: 2,
            max_input_bytes: MAX_JOB_INPUT,
            max_output_bytes: MAX_JOB_OUTPUT,
            default_timeout_ms: 30_000,
            process_memory_bytes: 512 * 1024 * 1024,
            process_cpu_seconds: 30,
            work_dir: PathBuf::from("runtime-work"),
        }
    }
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.max_queued_jobs == 0
            || self.max_concurrent_jobs == 0
            || self.max_concurrent_jobs > self.max_queued_jobs
            || self.max_input_bytes == 0
            || self.max_input_bytes > MAX_JOB_INPUT
            || self.max_output_bytes == 0
            || self.max_output_bytes > MAX_JOB_OUTPUT
            || self.default_timeout_ms == 0
            || self.process_memory_bytes == 0
            || self.process_cpu_seconds == 0
        {
            return Err(RuntimeError::InvalidConfig(
                "runtime bounds are empty, oversized, or inconsistently ordered".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ProcessSandbox {
    TrustedLocal,
    Bubblewrap { executable: PathBuf },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ExecutorSpec {
    BuiltinText {
        model: String,
    },
    BuiltinTraining,
    ExternalProcess {
        program: PathBuf,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        sandbox: ProcessSandbox,
    },
    LlamaCpp {
        program: PathBuf,
        model_path: PathBuf,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        sandbox: ProcessSandbox,
    },
}

impl ExecutorSpec {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        match self {
            Self::BuiltinText { model } => {
                if model.is_empty() || model.len() > 128 {
                    return Err(RuntimeError::InvalidExecutor(
                        "built-in model identity is invalid".to_string(),
                    ));
                }
            }
            Self::BuiltinTraining => {}
            Self::ExternalProcess {
                program,
                args,
                env,
                sandbox,
            } => {
                if program.as_os_str().is_empty() {
                    return Err(RuntimeError::InvalidExecutor(
                        "external process path is empty".to_string(),
                    ));
                }
                if args.len() > MAX_PROCESS_ARGS || env.len() > MAX_PROCESS_ENV {
                    return Err(RuntimeError::InvalidExecutor(
                        "external process configuration exceeds bounds".to_string(),
                    ));
                }
                for (key, value) in env {
                    if key.is_empty() || key.len() > 128 || value.len() > 4096 {
                        return Err(RuntimeError::InvalidExecutor(
                            "external process environment is invalid".to_string(),
                        ));
                    }
                }
                if let ProcessSandbox::Bubblewrap { executable } = sandbox {
                    if executable.as_os_str().is_empty() {
                        return Err(RuntimeError::InvalidExecutor(
                            "bubblewrap executable path is empty".to_string(),
                        ));
                    }
                }
            }
            Self::LlamaCpp {
                program,
                model_path,
                args,
                env,
                sandbox,
            } => {
                if program.as_os_str().is_empty()
                    || model_path.as_os_str().is_empty()
                    || !model_path.is_file()
                {
                    return Err(RuntimeError::InvalidExecutor(
                        "llama_cpp program or model path is invalid".to_string(),
                    ));
                }
                if args.len() > MAX_PROCESS_ARGS || env.len() > MAX_PROCESS_ENV {
                    return Err(RuntimeError::InvalidExecutor(
                        "llama_cpp configuration exceeds bounds".to_string(),
                    ));
                }
                for (key, value) in env {
                    if key.is_empty() || key.len() > 128 || value.len() > 4096 {
                        return Err(RuntimeError::InvalidExecutor(
                            "llama_cpp environment is invalid".to_string(),
                        ));
                    }
                }
                if let ProcessSandbox::Bubblewrap { executable } = sandbox
                    && executable.as_os_str().is_empty()
                {
                    return Err(RuntimeError::InvalidExecutor(
                        "bubblewrap executable path is empty".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionResult {
    pub output: Vec<u8>,
    pub output_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionOutcome {
    Succeeded(ExecutionResult),
    Cancelled,
    TimedOut,
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("runtime configuration is invalid: {0}")]
    InvalidConfig(String),
    #[error("executor configuration is invalid: {0}")]
    InvalidExecutor(String),
    #[error("runtime queue is full")]
    QueueFull,
    #[error("job {0} is already admitted")]
    Duplicate(JobId),
    #[error("job input exceeds the local limit")]
    InputTooLarge,
    #[error("job output exceeds the local limit")]
    OutputTooLarge,
    #[error("job execution failed: {0}")]
    Process(String),
    #[error("job execution I/O failed: {0}")]
    Io(#[from] io::Error),
}

#[derive(Clone)]
pub struct Cancellation {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Cancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    async fn wait(&self) {
        if !self.is_cancelled() {
            self.notify.notified().await;
        }
    }
}

struct AdmissionInner {
    job_id: JobId,
    cancellation: Cancellation,
    _queue_permit: OwnedSemaphorePermit,
}

pub struct Admission(AdmissionInner);

impl Admission {
    pub fn job_id(&self) -> JobId {
        self.0.job_id
    }

    pub fn cancellation(&self) -> Cancellation {
        self.0.cancellation.clone()
    }
}

pub struct Runtime {
    config: RuntimeConfig,
    queue: Arc<Semaphore>,
    running: Arc<Semaphore>,
    active: Arc<Mutex<std::collections::HashMap<JobId, Cancellation>>>,
}

impl Runtime {
    pub fn new(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        config.validate()?;
        std::fs::create_dir_all(&config.work_dir)?;
        Ok(Self {
            queue: Arc::new(Semaphore::new(config.max_queued_jobs)),
            running: Arc::new(Semaphore::new(config.max_concurrent_jobs)),
            active: Arc::new(Mutex::new(std::collections::HashMap::new())),
            config,
        })
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub fn queue_depth(&self) -> usize {
        self.config
            .max_queued_jobs
            .saturating_sub(self.queue.available_permits())
    }

    pub fn running_jobs(&self) -> usize {
        self.config
            .max_concurrent_jobs
            .saturating_sub(self.running.available_permits())
    }

    pub async fn admit(&self, job: &JobRequest) -> Result<Admission, RuntimeError> {
        job.validate()
            .map_err(|error| RuntimeError::Process(error.to_string()))?;
        if job.input.len() > self.config.max_input_bytes {
            return Err(RuntimeError::InputTooLarge);
        }
        if job.max_output_bytes as usize > self.config.max_output_bytes {
            return Err(RuntimeError::OutputTooLarge);
        }
        let permit = self
            .queue
            .clone()
            .try_acquire_owned()
            .map_err(|_| RuntimeError::QueueFull)?;
        let cancellation = Cancellation {
            cancelled: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        };
        let mut active = self.active.lock().await;
        if active.contains_key(&job.job_id) {
            drop(permit);
            return Err(RuntimeError::Duplicate(job.job_id));
        }
        active.insert(job.job_id, cancellation.clone());
        drop(active);
        Ok(Admission(AdmissionInner {
            job_id: job.job_id,
            cancellation,
            _queue_permit: permit,
        }))
    }

    pub async fn cancel(&self, job_id: JobId) -> bool {
        if let Some(cancellation) = self.active.lock().await.get(&job_id) {
            cancellation.cancel();
            true
        } else {
            false
        }
    }

    pub async fn cancel_all(&self) {
        let active = self.active.lock().await;
        for cancellation in active.values() {
            cancellation.cancel();
        }
    }

    pub async fn abandon(&self, admission: Admission) {
        self.finish(admission.job_id()).await;
    }

    pub async fn execute(
        &self,
        admission: Admission,
        job: &JobRequest,
        executor: &ExecutorSpec,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        self.execute_inner(admission, job, executor, None).await
    }

    pub async fn execute_streaming(
        &self,
        admission: Admission,
        job: &JobRequest,
        executor: &ExecutorSpec,
        chunks: mpsc::Sender<Vec<u8>>,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        self.execute_inner(admission, job, executor, Some(chunks))
            .await
    }

    async fn execute_inner(
        &self,
        admission: Admission,
        job: &JobRequest,
        executor: &ExecutorSpec,
        chunks: Option<mpsc::Sender<Vec<u8>>>,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        if let Err(error) = executor.validate() {
            self.finish(admission.job_id()).await;
            return Err(error);
        }
        let cancellation = admission.cancellation();
        let running = tokio::select! {
            permit = self.running.clone().acquire_owned() => permit.map_err(|_| RuntimeError::Process("runtime closed".to_string()))?,
            _ = cancellation.wait() => {
                self.finish(admission.job_id()).await;
                return Ok(ExecutionOutcome::Cancelled);
            }
            _ = sleep(Duration::from_millis(job.deadline_ms.min(self.config.default_timeout_ms))) => {
                self.finish(admission.job_id()).await;
                return Ok(ExecutionOutcome::TimedOut);
            }
        };
        let timeout_ms = job.deadline_ms.min(self.config.default_timeout_ms);
        let max_output_bytes = self
            .config
            .max_output_bytes
            .min(job.max_output_bytes as usize);
        // Keep the executor future alive until it has cleaned up. In particular, dropping
        // `run_process` from an outer cancellation branch could otherwise leave a child
        // process alive after its parent job had already become terminal.
        let result = run_executor(
            executor,
            &job.input,
            &self.config,
            cancellation,
            timeout_ms,
            max_output_bytes,
            chunks,
        )
        .await;
        drop(running);
        self.finish(admission.job_id()).await;
        result
    }

    async fn finish(&self, job_id: JobId) {
        self.active.lock().await.remove(&job_id);
    }
}

async fn run_executor(
    executor: &ExecutorSpec,
    input: &[u8],
    config: &RuntimeConfig,
    cancellation: Cancellation,
    timeout_ms: u64,
    max_output_bytes: usize,
    chunks: Option<mpsc::Sender<Vec<u8>>>,
) -> Result<ExecutionOutcome, RuntimeError> {
    match executor {
        ExecutorSpec::BuiltinText { model } => {
            run_builtin(model, input, cancellation, max_output_bytes, chunks).await
        }
        ExecutorSpec::BuiltinTraining => {
            run_builtin_training(input, cancellation, max_output_bytes, chunks).await
        }
        ExecutorSpec::ExternalProcess { .. } | ExecutorSpec::LlamaCpp { .. } => {
            run_process(
                executor,
                input,
                config,
                cancellation,
                timeout_ms,
                max_output_bytes,
                chunks,
            )
            .await
        }
    }
}

async fn run_builtin(
    model: &str,
    input: &[u8],
    cancellation: Cancellation,
    max_output_bytes: usize,
    chunks: Option<mpsc::Sender<Vec<u8>>>,
) -> Result<ExecutionOutcome, RuntimeError> {
    if cancellation.is_cancelled() {
        return Ok(ExecutionOutcome::Cancelled);
    }
    let text = match serde_json::from_slice::<serde_json::Value>(input) {
        Ok(value) => value
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map_or_else(
                || String::from_utf8_lossy(input).into_owned(),
                ToString::to_string,
            ),
        Err(_) => String::from_utf8_lossy(input).into_owned(),
    };
    let mut score = 0i32;
    let mut tokens = 0u32;
    for token in text.split(|character: char| !character.is_ascii_alphanumeric()) {
        if token.is_empty() {
            continue;
        }
        tokens = tokens.saturating_add(1);
        match token.to_ascii_lowercase().as_str() {
            "good" | "great" | "excellent" | "love" | "like" | "useful" | "fast" => score += 2,
            "bad" | "terrible" | "awful" | "hate" | "slow" | "broken" | "fail" => score -= 2,
            _ => {}
        }
    }
    let normalized = (score as f32 / tokens.max(1) as f32).clamp(-1.0, 1.0);
    let label = if normalized > 0.15 {
        "positive"
    } else if normalized < -0.15 {
        "negative"
    } else {
        "neutral"
    };
    // This deterministic adversarial model is a lab fixture for the V6
    // evaluator-collusion experiment. It has the same bounded executor path
    // as a normal built-in model, but deliberately reports the opposite
    // classification so the production selector/majority path is exercised
    // by an actual node process. It is not a trust signal and is never
    // enabled implicitly.
    let reported_label = if model == "builtin.adversarial.sentiment.v1" {
        match label {
            "positive" => "negative",
            "negative" => "positive",
            _ => "neutral",
        }
    } else {
        label
    };
    let output = serde_json::json!({
        "model": model,
        "label": reported_label,
        "score": normalized,
        "tokens": tokens,
        "input_hash": hex::encode(hash(input).as_bytes()),
    });
    let bytes =
        serde_json::to_vec(&output).map_err(|error| RuntimeError::Process(error.to_string()))?;
    if bytes.len() > max_output_bytes {
        return Err(RuntimeError::OutputTooLarge);
    }
    emit_chunks(chunks, &bytes).await;
    Ok(ExecutionOutcome::Succeeded(ExecutionResult {
        output_hash: ArtifactId::from_bytes_hashed(&bytes),
        output: bytes,
    }))
}

#[derive(Deserialize)]
struct TrainingInput {
    model: TrainingModel,
    samples: Vec<TrainingSample>,
    learning_rate: f64,
    step: u64,
    job_id: String,
    model_artifact: String,
    dataset_artifact: String,
}

#[derive(Clone, Copy, Deserialize)]
struct TrainingModel {
    weight: f64,
    bias: f64,
}

#[derive(Clone, Copy, Deserialize)]
struct TrainingSample {
    x: f64,
    y: f64,
}

async fn run_builtin_training(
    input: &[u8],
    cancellation: Cancellation,
    max_output_bytes: usize,
    chunks: Option<mpsc::Sender<Vec<u8>>>,
) -> Result<ExecutionOutcome, RuntimeError> {
    if cancellation.is_cancelled() {
        return Ok(ExecutionOutcome::Cancelled);
    }
    let request: TrainingInput = serde_json::from_slice(input).map_err(|error| {
        RuntimeError::Process(format!("invalid reference training input: {error}"))
    })?;
    if request.samples.is_empty()
        || request.samples.len() > 4096
        || !request.learning_rate.is_finite()
        || request.learning_rate <= 0.0
        || request.learning_rate > 10.0
        || !request.model.weight.is_finite()
        || !request.model.bias.is_finite()
    {
        return Err(RuntimeError::Process(
            "reference training input is outside bounds".to_string(),
        ));
    }
    let mut weight_gradient = 0.0;
    let mut bias_gradient = 0.0;
    let mut loss = 0.0;
    for sample in &request.samples {
        if !sample.x.is_finite() || !sample.y.is_finite() {
            return Err(RuntimeError::Process(
                "reference training sample is not finite".to_string(),
            ));
        }
        let error = request.model.weight * sample.x + request.model.bias - sample.y;
        loss += error * error;
        weight_gradient += 2.0 * error * sample.x;
        bias_gradient += 2.0 * error;
    }
    let count = request.samples.len() as f64;
    let output = serde_json::json!({
        "kind": "reference_gradient",
        "job_id": request.job_id,
        "model_artifact": request.model_artifact,
        "dataset_artifact": request.dataset_artifact,
        "step": request.step,
        "samples": request.samples.len(),
        "weight_gradient": weight_gradient / count,
        "bias_gradient": bias_gradient / count,
        "loss": loss / count,
    });
    let bytes =
        serde_json::to_vec(&output).map_err(|error| RuntimeError::Process(error.to_string()))?;
    if bytes.len() > max_output_bytes {
        return Err(RuntimeError::OutputTooLarge);
    }
    emit_chunks(chunks, &bytes).await;
    Ok(ExecutionOutcome::Succeeded(ExecutionResult {
        output_hash: ArtifactId::from_bytes_hashed(&bytes),
        output: bytes,
    }))
}

async fn run_process(
    executor: &ExecutorSpec,
    input: &[u8],
    config: &RuntimeConfig,
    cancellation: Cancellation,
    timeout_ms: u64,
    max_output_bytes: usize,
    chunks: Option<mpsc::Sender<Vec<u8>>>,
) -> Result<ExecutionOutcome, RuntimeError> {
    let mut llama_args = Vec::new();
    let (program, args, env, sandbox, model_path) = match executor {
        ExecutorSpec::ExternalProcess {
            program,
            args,
            env,
            sandbox,
        } => (program, args.as_slice(), env, sandbox, None),
        ExecutorSpec::LlamaCpp {
            program,
            model_path,
            args,
            env,
            sandbox,
        } => {
            llama_args.push("--model".to_string());
            llama_args.push(model_path.to_string_lossy().into_owned());
            llama_args.extend(args.iter().cloned());
            (
                program,
                llama_args.as_slice(),
                env,
                sandbox,
                Some(model_path),
            )
        }
        ExecutorSpec::BuiltinText { .. } => {
            return Err(RuntimeError::InvalidExecutor(
                "run_process received a non-process executor".to_string(),
            ));
        }
        ExecutorSpec::BuiltinTraining => {
            return Err(RuntimeError::InvalidExecutor(
                "run_process received a non-process executor".to_string(),
            ));
        }
    };
    let mut command = match sandbox {
        ProcessSandbox::TrustedLocal => {
            let mut command = Command::new(program);
            command.args(args);
            command
        }
        ProcessSandbox::Bubblewrap { executable } => {
            let mut command = Command::new(executable);
            command
                .arg("--die-with-parent")
                .arg("--unshare-all")
                .arg("--new-session")
                .arg("--proc")
                .arg("/proc")
                .arg("--dev")
                .arg("/dev")
                .arg("--tmpfs")
                .arg("/tmp")
                .arg("--ro-bind")
                .arg("/usr")
                .arg("/usr")
                .arg("--ro-bind")
                .arg("/bin")
                .arg("/bin")
                .arg("--ro-bind")
                .arg("/usr/lib")
                .arg("/lib")
                .arg("--ro-bind")
                .arg("/usr/lib")
                .arg("/lib64");
            if let Some(model_path) = model_path.filter(|path| path.is_absolute()) {
                command.arg("--ro-bind").arg(model_path).arg(model_path);
            }
            command
                .arg("--ro-bind")
                .arg(&config.work_dir)
                .arg("/work")
                .arg("--chdir")
                .arg("/work")
                .arg("--")
                .arg(program)
                .args(args);
            command
        }
    };
    command
        .env_clear()
        .envs(env)
        .current_dir(&config.work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_unix_limits(&mut command, config);
    let mut child = command
        .spawn()
        .map_err(|error| RuntimeError::Process(error.to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| RuntimeError::Process("external process has no stdout".to_string()))?;
    let stderr = child.stderr.take();
    let max_output = max_output_bytes;
    let output_task = tokio::spawn(read_limited_with_chunks(stdout, max_output, chunks));
    let stderr_task = tokio::spawn(async move {
        match stderr {
            Some(stream) => read_limited(stream, MAX_ERROR_BYTES)
                .await
                .unwrap_or_else(|_| LimitedOutput::empty()),
            None => LimitedOutput::empty(),
        }
    });
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(error) = stdin.write_all(input).await {
            if error.kind() != io::ErrorKind::BrokenPipe {
                kill_child(&mut child).await;
                let _ = output_task.await;
                let _ = stderr_task.await;
                return Err(RuntimeError::Io(error));
            }
        }
        if let Err(error) = stdin.shutdown().await {
            if error.kind() != io::ErrorKind::BrokenPipe {
                kill_child(&mut child).await;
                let _ = output_task.await;
                let _ = stderr_task.await;
                return Err(RuntimeError::Io(error));
            }
        }
    }
    let status = tokio::select! {
        _ = sleep(Duration::from_millis(timeout_ms)) => {
            kill_child(&mut child).await;
            let _ = output_task.await;
            let _ = stderr_task.await;
            return Ok(ExecutionOutcome::TimedOut);
        }
        result = child.wait() => result?,
        _ = cancellation.wait() => {
            kill_child(&mut child).await;
            let _ = output_task.await;
            let _ = stderr_task.await;
            return Ok(ExecutionOutcome::Cancelled);
        }
    };
    let limited_output = output_task
        .await
        .map_err(|error| RuntimeError::Process(error.to_string()))??;
    let stderr = stderr_task.await.unwrap_or_else(|_| LimitedOutput::empty());
    if limited_output.truncated {
        return Err(RuntimeError::OutputTooLarge);
    }
    let output = limited_output.bytes;
    if !status.success() {
        let details = String::from_utf8_lossy(&stderr.bytes);
        return Err(RuntimeError::Process(format!(
            "process exited with {status}: {details}"
        )));
    }
    Ok(ExecutionOutcome::Succeeded(ExecutionResult {
        output_hash: ArtifactId::from_bytes_hashed(&output),
        output,
    }))
}

#[derive(Default)]
struct LimitedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

impl LimitedOutput {
    fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            truncated: false,
        }
    }
}

async fn read_limited<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
) -> io::Result<LimitedOutput> {
    read_limited_with_chunks(&mut reader, limit, None).await
}

async fn read_limited_with_chunks<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
    chunks: Option<mpsc::Sender<Vec<u8>>>,
) -> io::Result<LimitedOutput> {
    let mut output = Vec::with_capacity(limit.min(8192));
    let mut buffer = [0u8; 8192];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(LimitedOutput {
                bytes: output,
                truncated: false,
            });
        }
        if output.len().saturating_add(read) > limit {
            let allowed = (limit.saturating_sub(output.len())).min(read);
            let piece = &buffer[..allowed];
            output.extend_from_slice(piece);
            emit_chunks_ref(chunks.as_ref(), piece).await;
            return Ok(LimitedOutput {
                bytes: output,
                truncated: true,
            });
        }
        let piece = &buffer[..read];
        output.extend_from_slice(piece);
        emit_chunks_ref(chunks.as_ref(), piece).await;
    }
}

async fn emit_chunks(chunks: Option<mpsc::Sender<Vec<u8>>>, bytes: &[u8]) {
    emit_chunks_ref(chunks.as_ref(), bytes).await;
}

async fn emit_chunks_ref(chunks: Option<&mpsc::Sender<Vec<u8>>>, bytes: &[u8]) {
    let Some(chunks) = chunks else { return };
    for piece in bytes.chunks(MAX_ARTIFACT_CHUNK) {
        if chunks.send(piece.to_vec()).await.is_err() {
            return;
        }
    }
}

async fn kill_child(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id().filter(|pid| *pid > 0) {
        // External executors may create descendants. The child is placed in its
        // own process group before launch so timeout/cancellation can terminate
        // the whole group instead of leaving work behind on the host.
        unsafe {
            let _ = libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
        }
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

#[cfg(unix)]
fn apply_unix_limits(command: &mut Command, config: &RuntimeConfig) {
    let cpu_seconds = config.process_cpu_seconds;
    let memory_bytes = config.process_memory_bytes;
    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            let cpu = libc::rlimit {
                rlim_cur: cpu_seconds,
                rlim_max: cpu_seconds,
            };
            let memory = libc::rlimit {
                rlim_cur: memory_bytes,
                rlim_max: memory_bytes,
            };
            if libc::setrlimit(libc::RLIMIT_CPU, &cpu) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::setrlimit(libc::RLIMIT_AS, &memory) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn apply_unix_limits(_command: &mut Command, _config: &RuntimeConfig) {}

#[cfg(test)]
mod tests {
    use super::*;
    use intelligence_protocol::{JobKind, NodeId, PrivacyPolicy};

    fn job(input: &[u8]) -> JobRequest {
        JobRequest {
            job_id: JobId::from_bytes([1; 16]),
            origin: NodeId::from_bytes([2; 32]),
            kind: JobKind::Inference,
            capability: "inference.text".to_string(),
            model: None,
            input: input.to_vec(),
            deadline_ms: 1000,
            max_output_bytes: 4096,
            privacy: PrivacyPolicy::default(),
        }
    }

    #[tokio::test]
    async fn built_in_executor_returns_real_deterministic_output() {
        let config = RuntimeConfig {
            work_dir: std::env::temp_dir()
                .join(format!("intelligence-runtime-{}", std::process::id())),
            ..RuntimeConfig::default()
        };
        let runtime = Runtime::new(config).unwrap();
        let request = job(br#"{"text":"good and useful"}"#);
        let admission = runtime.admit(&request).await.unwrap();
        let outcome = runtime
            .execute(
                admission,
                &request,
                &ExecutorSpec::BuiltinText {
                    model: "builtin.tiny-sentiment.v1".to_string(),
                },
            )
            .await
            .unwrap();
        let ExecutionOutcome::Succeeded(result) = outcome else {
            panic!("expected success")
        };
        assert!(
            String::from_utf8(result.output)
                .unwrap()
                .contains("positive")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn operator_supplied_llama_adapter_passes_model_path_without_shell_interpolation() {
        let work_dir = std::env::temp_dir().join(format!(
            "intelligence-runtime-llama-{}-{}",
            std::process::id(),
            now_millis_for_test()
        ));
        let model_path = work_dir.join("operator-model.gguf");
        std::fs::create_dir_all(&work_dir).unwrap();
        std::fs::write(&model_path, b"tiny operator fixture").unwrap();
        let runtime = Runtime::new(RuntimeConfig {
            work_dir: work_dir.clone(),
            ..RuntimeConfig::default()
        })
        .unwrap();
        let request = job(b"operator prompt");
        let admission = runtime.admit(&request).await.unwrap();
        let outcome = runtime
            .execute(
                admission,
                &request,
                &ExecutorSpec::LlamaCpp {
                    program: PathBuf::from("/bin/echo"),
                    model_path: model_path.clone(),
                    args: vec!["--verbose".to_string()],
                    env: BTreeMap::new(),
                    sandbox: ProcessSandbox::TrustedLocal,
                },
            )
            .await
            .unwrap();
        let ExecutionOutcome::Succeeded(result) = outcome else {
            panic!("expected operator adapter success")
        };
        let output = String::from_utf8(result.output).unwrap();
        assert!(output.contains("--model"));
        assert!(output.contains(model_path.to_str().unwrap()));
        let _ = std::fs::remove_dir_all(work_dir);
    }

    #[tokio::test]
    async fn cancellation_is_terminal_before_execution() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        let request = job(b"good");
        let admission = runtime.admit(&request).await.unwrap();
        admission.cancellation().cancel();
        let outcome = runtime
            .execute(
                admission,
                &request,
                &ExecutorSpec::BuiltinText {
                    model: "builtin.tiny-sentiment.v1".to_string(),
                },
            )
            .await
            .unwrap();
        assert_eq!(outcome, ExecutionOutcome::Cancelled);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streaming_executor_emits_bounded_chunks_before_terminal_result() {
        let work_dir = std::env::temp_dir().join(format!(
            "intelligence-runtime-stream-{}-{}",
            std::process::id(),
            now_millis_for_test()
        ));
        let runtime = Arc::new(
            Runtime::new(RuntimeConfig {
                work_dir: work_dir.clone(),
                ..RuntimeConfig::default()
            })
            .unwrap(),
        );
        let input = vec![b'x'; 64 * 1024];
        let request = JobRequest {
            job_id: JobId::from_bytes([6; 16]),
            max_output_bytes: input.len() as u32,
            ..job(&input)
        };
        let executor = ExecutorSpec::ExternalProcess {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), "cat".to_string()],
            env: BTreeMap::new(),
            sandbox: ProcessSandbox::TrustedLocal,
        };
        let admission = runtime.admit(&request).await.unwrap();
        let (sender, mut receiver) = mpsc::channel(4);
        let task_runtime = runtime.clone();
        let task_request = request.clone();
        let task_executor = executor.clone();
        let task = tokio::spawn(async move {
            task_runtime
                .execute_streaming(admission, &task_request, &task_executor, sender)
                .await
        });
        let mut streamed = Vec::new();
        while let Some(chunk) = receiver.recv().await {
            assert!(!chunk.is_empty());
            assert!(chunk.len() <= MAX_ARTIFACT_CHUNK);
            streamed.extend_from_slice(&chunk);
        }
        let outcome = task.await.unwrap().unwrap();
        let ExecutionOutcome::Succeeded(result) = outcome else {
            panic!("expected success")
        };
        assert_eq!(streamed, result.output);
        assert!(streamed.len() > MAX_ARTIFACT_CHUNK.min(8192));
        let _ = std::fs::remove_dir_all(work_dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn external_process_timeout_cancellation_and_output_limits_are_enforced() {
        let work_dir = std::env::temp_dir().join(format!(
            "intelligence-runtime-process-{}-{}",
            std::process::id(),
            now_millis_for_test()
        ));
        let runtime = Arc::new(
            Runtime::new(RuntimeConfig {
                work_dir: work_dir.clone(),
                ..RuntimeConfig::default()
            })
            .unwrap(),
        );

        let executor = |script: &str| ExecutorSpec::ExternalProcess {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), script.to_string()],
            env: BTreeMap::new(),
            sandbox: ProcessSandbox::TrustedLocal,
        };

        let oversized = JobRequest {
            job_id: JobId::from_bytes([3; 16]),
            max_output_bytes: 8,
            ..job(b"ignored")
        };
        let admission = runtime.admit(&oversized).await.unwrap();
        let result = runtime
            .execute(admission, &oversized, &executor("printf 123456789"))
            .await;
        assert!(matches!(result, Err(RuntimeError::OutputTooLarge)));

        let timed_out = JobRequest {
            job_id: JobId::from_bytes([4; 16]),
            deadline_ms: 50,
            ..job(b"ignored")
        };
        let admission = runtime.admit(&timed_out).await.unwrap();
        let result = runtime
            .execute(admission, &timed_out, &executor("sleep 2"))
            .await
            .unwrap();
        assert_eq!(result, ExecutionOutcome::TimedOut);

        let cancellable = JobRequest {
            job_id: JobId::from_bytes([5; 16]),
            deadline_ms: 5000,
            ..job(b"ignored")
        };
        let admission = runtime.admit(&cancellable).await.unwrap();
        let task_runtime = runtime.clone();
        let cancellable_executor = executor("sleep 2");
        let cancellable_id = cancellable.job_id;
        let task = tokio::spawn(async move {
            task_runtime
                .execute(admission, &cancellable, &cancellable_executor)
                .await
        });
        sleep(Duration::from_millis(100)).await;
        assert!(runtime.cancel(cancellable_id).await);
        assert_eq!(task.await.unwrap().unwrap(), ExecutionOutcome::Cancelled);

        let _ = std::fs::remove_dir_all(work_dir);
    }

    fn now_millis_for_test() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis())
    }
}
