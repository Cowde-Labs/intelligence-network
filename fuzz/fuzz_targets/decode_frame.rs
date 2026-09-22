#![no_main]

use intelligence_protocol::{MAX_FRAME_SIZE, decode_frame};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = decode_frame(data, MAX_FRAME_SIZE);
});
