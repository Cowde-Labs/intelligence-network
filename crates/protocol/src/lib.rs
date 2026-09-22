//! Versioned, bounded wire and artifact contracts for the Intelligence Network.

mod codec;
mod types;
mod validation;

pub use codec::{
    CodecError, FRAME_HEADER_SIZE, FRAME_MAGIC, FrameHeader, FrameKind, decode_frame,
    encode_message, encode_message_at_version,
};
pub use types::*;
pub use validation::{
    MAX_ADDRESS_LEN, MAX_ADDRESSES, MAX_ARTIFACT_CHUNK, MAX_CAPABILITIES, MAX_DHT_CONTACTS,
    MAX_DHT_QUERY_MESSAGE, MAX_DHT_RECORD_VALUE, MAX_DHT_RECORDS, MAX_EVIDENCE_PAYLOAD,
    MAX_FRAME_SIZE, MAX_JOB_INPUT, MAX_JOB_OUTPUT, MAX_METADATA, MAX_PEERS, MAX_RELAY_PAYLOAD,
    MAX_STRING_LEN, MAX_V4_BYTES, MAX_V4_GROUPS, MAX_V4_PLAN_WORKERS, MAX_V4_STAGES, MAX_V4_VECTOR,
    MAX_V5_ASSIGNMENTS, MAX_V5_BACKENDS, MAX_V5_FEATURES, MAX_V5_FORMATS, MAX_V5_REQUIREMENTS,
    ValidationError,
};
