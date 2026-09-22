use crate::{Message, PROTOCOL_MAJOR, PROTOCOL_MINOR, ValidationError};
use thiserror::Error;

pub const FRAME_MAGIC: [u8; 4] = *b"INP1";
pub const FRAME_HEADER_SIZE: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameHeader {
    pub major: u16,
    pub minor: u16,
    pub kind: FrameKind,
    pub payload_len: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum FrameKind {
    Hello = 1,
    PeerExchange = 2,
    Announcement = 3,
    JobRequest = 4,
    JobUpdate = 5,
    CancelJob = 6,
    ArtifactRequest = 7,
    ArtifactChunk = 8,
    Ping = 9,
    Pong = 10,
    Goodbye = 11,
    Error = 12,
    AddressUpdate = 13,
    AddressObservation = 14,
    ArtifactTransferRequest = 15,
    ArtifactTransferChunk = 16,
    KeyRotation = 17,
    RelayEnvelope = 18,
    DhtRequest = 19,
    DhtResponse = 20,
    Training = 21,
    TrainingV4 = 22,
    TrainingV5 = 23,
}

impl TryFrom<u16> for FrameKind {
    type Error = CodecError;

    fn try_from(value: u16) -> Result<Self, CodecError> {
        match value {
            1 => Ok(Self::Hello),
            2 => Ok(Self::PeerExchange),
            3 => Ok(Self::Announcement),
            4 => Ok(Self::JobRequest),
            5 => Ok(Self::JobUpdate),
            6 => Ok(Self::CancelJob),
            7 => Ok(Self::ArtifactRequest),
            8 => Ok(Self::ArtifactChunk),
            9 => Ok(Self::Ping),
            10 => Ok(Self::Pong),
            11 => Ok(Self::Goodbye),
            12 => Ok(Self::Error),
            13 => Ok(Self::AddressUpdate),
            14 => Ok(Self::AddressObservation),
            15 => Ok(Self::ArtifactTransferRequest),
            16 => Ok(Self::ArtifactTransferChunk),
            17 => Ok(Self::KeyRotation),
            18 => Ok(Self::RelayEnvelope),
            19 => Ok(Self::DhtRequest),
            20 => Ok(Self::DhtResponse),
            21 => Ok(Self::Training),
            22 => Ok(Self::TrainingV4),
            23 => Ok(Self::TrainingV5),
            _ => Err(CodecError::UnknownKind(value)),
        }
    }
}

impl FrameHeader {
    pub fn encode(self) -> [u8; FRAME_HEADER_SIZE] {
        let mut bytes = [0u8; FRAME_HEADER_SIZE];
        bytes[..4].copy_from_slice(&FRAME_MAGIC);
        bytes[4..6].copy_from_slice(&self.major.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.minor.to_le_bytes());
        bytes[8..10].copy_from_slice(&(self.kind as u16).to_le_bytes());
        bytes[10..12].copy_from_slice(&0u16.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.payload_len.to_le_bytes());
        bytes
    }

    pub fn decode(
        bytes: &[u8; FRAME_HEADER_SIZE],
        max_frame_size: usize,
    ) -> Result<Self, CodecError> {
        if bytes[..4] != FRAME_MAGIC {
            return Err(CodecError::BadMagic);
        }
        let major = u16::from_le_bytes([bytes[4], bytes[5]]);
        let minor = u16::from_le_bytes([bytes[6], bytes[7]]);
        if major != PROTOCOL_MAJOR || minor > PROTOCOL_MINOR {
            return Err(CodecError::UnsupportedVersion { major, minor });
        }
        let kind = FrameKind::try_from(u16::from_le_bytes([bytes[8], bytes[9]]))?;
        let payload_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let frame_size = FRAME_HEADER_SIZE.saturating_add(payload_len as usize);
        if frame_size > max_frame_size {
            return Err(CodecError::FrameTooLarge {
                size: frame_size,
                max: max_frame_size,
            });
        }
        Ok(Self {
            major,
            minor,
            kind,
            payload_len,
        })
    }
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("bad frame magic")]
    BadMagic,
    #[error("unsupported protocol version {major}.{minor}")]
    UnsupportedVersion { major: u16, minor: u16 },
    #[error("unknown frame kind {0}")]
    UnknownKind(u16),
    #[error("frame payload is {size} bytes, maximum is {max}")]
    FrameTooLarge { size: usize, max: usize },
    #[error("payload length {declared} does not match {actual} bytes")]
    LengthMismatch { declared: usize, actual: usize },
    #[error("wire serialization failed: {0}")]
    Serialize(#[from] postcard::Error),
    #[error("message validation failed: {0}")]
    Validation(#[from] ValidationError),
    #[error("message kind {actual:?} does not match frame kind {expected:?}")]
    KindMismatch {
        expected: FrameKind,
        actual: FrameKind,
    },
}

pub fn encode_message(message: &Message, max_frame_size: usize) -> Result<Vec<u8>, CodecError> {
    encode_message_at_version(message, max_frame_size, PROTOCOL_MAJOR, PROTOCOL_MINOR)
}

pub fn encode_message_at_version(
    message: &Message,
    max_frame_size: usize,
    major: u16,
    minor: u16,
) -> Result<Vec<u8>, CodecError> {
    if major != PROTOCOL_MAJOR || minor > PROTOCOL_MINOR {
        return Err(CodecError::UnsupportedVersion { major, minor });
    }
    message.validate()?;
    if minor == 0 && message.kind() >= 13 {
        return Err(CodecError::UnsupportedVersion { major, minor });
    }
    if minor < 2 && message.kind() >= 19 {
        return Err(CodecError::UnsupportedVersion { major, minor });
    }
    if minor < 3 && message.kind() >= 21 {
        return Err(CodecError::UnsupportedVersion { major, minor });
    }
    if minor < 4 && message.kind() >= 22 {
        return Err(CodecError::UnsupportedVersion { major, minor });
    }
    if minor < 6 && message.kind() >= 23 {
        return Err(CodecError::UnsupportedVersion { major, minor });
    }
    if let Message::TrainingV4(training) = message
        && minor < training.minimum_minor()
    {
        return Err(CodecError::UnsupportedVersion { major, minor });
    }
    let payload = postcard::to_allocvec(message)?;
    let frame_size = FRAME_HEADER_SIZE.saturating_add(payload.len());
    if frame_size > max_frame_size {
        return Err(CodecError::FrameTooLarge {
            size: frame_size,
            max: max_frame_size,
        });
    }
    let kind =
        FrameKind::try_from(message.kind()).map_err(|_| CodecError::UnknownKind(message.kind()))?;
    let header = FrameHeader {
        major,
        minor,
        kind,
        payload_len: payload.len() as u32,
    };
    let mut encoded = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    encoded.extend_from_slice(&header.encode());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

pub fn decode_frame(frame: &[u8], max_frame_size: usize) -> Result<Message, CodecError> {
    if frame.len() < FRAME_HEADER_SIZE {
        return Err(CodecError::LengthMismatch {
            declared: FRAME_HEADER_SIZE,
            actual: frame.len(),
        });
    }
    let header_bytes: &[u8; FRAME_HEADER_SIZE] =
        frame[..FRAME_HEADER_SIZE]
            .try_into()
            .map_err(|_| CodecError::LengthMismatch {
                declared: FRAME_HEADER_SIZE,
                actual: frame.len(),
            })?;
    let header = FrameHeader::decode(header_bytes, max_frame_size)?;
    let payload = &frame[FRAME_HEADER_SIZE..];
    if payload.len() != header.payload_len as usize {
        return Err(CodecError::LengthMismatch {
            declared: header.payload_len as usize,
            actual: payload.len(),
        });
    }
    let message: Message = postcard::from_bytes(payload)?;
    let actual =
        FrameKind::try_from(message.kind()).map_err(|_| CodecError::UnknownKind(message.kind()))?;
    if actual != header.kind {
        return Err(CodecError::KindMismatch {
            expected: header.kind,
            actual,
        });
    }
    message.validate()?;
    Ok(message)
}
