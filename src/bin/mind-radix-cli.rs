//! Line-oriented CLI so the Python router can use the index without a binding layer.
//!
//! One JSON object per line on stdin, one JSON object per line on stdout. The index
//! lives in this process, so the router keeps a single long-running child rather than
//! replaying state per request.
//!
//!   {"op":"observe","backend":"studio","tokens":[...],"cached":1234,"prompt":1300}
//!   {"op":"estimate","tokens":[...]}          -> {"estimates":[...]}
//!   {"op":"forget_backend","backend":"amd"}
//!   {"op":"stats"}                            -> {"backends":{...},"recorded":N}
//!
//! Tokens, not text: the router must hash exactly what the engine was sent, and the
//! tokenizer already exists on the client side.

use std::io::{self, BufRead, Write};

use mind_radix::{Observation, PrefixIndex};

const BLOCK_SIZE: usize = 16;

fn main() {
    let block_size = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(BLOCK_SIZE);
    let mut index = PrefixIndex::new(block_size);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue,
            Err(_) => break,
        };
        let reply = handle(&mut index, &line);
        let _ = writeln!(stdout, "{reply}");
        let _ = stdout.flush();
    }
}

/// Minimal JSON reading: this is a private protocol between two of our own programs,
/// so a full parser would be more code than the whole index. Values are only ever
/// strings, numbers, and arrays of numbers.
fn handle(index: &mut PrefixIndex, line: &str) -> String {
    let op = field(line, "op").unwrap_or_default();
    match op.as_str() {
        "observe" => {
            let backend = field(line, "backend").unwrap_or_default();
            let cached = number(line, "cached").unwrap_or(0.0) as usize;
            let prompt = number(line, "prompt").unwrap_or(0.0) as usize;
            let tokens = tokens_of(line);
            index.observe(&backend, &tokens, Observation { cached_tokens: cached, prompt_tokens: prompt });
            format!("{{\"ok\":true,\"recorded\":{}}}", index.recorded())
        }
        "estimate" => {
            let tokens = tokens_of(line);
            let estimates = index.estimate(&tokens);
            let items: Vec<String> = estimates
                .iter()
                .map(|e| {
                    format!(
                        "{{\"backend\":\"{}\",\"cached_tokens\":{},\"shared_blocks\":{},\"measured\":{}}}",
                        e.backend, e.cached_tokens, e.shared_blocks, e.measured
                    )
                })
                .collect();
            format!("{{\"estimates\":[{}]}}", items.join(","))
        }
        "forget_backend" => {
            let backend = field(line, "backend").unwrap_or_default();
            index.forget_backend(&backend);
            "{\"ok\":true}".to_string()
        }
        "stats" => {
            let backends: Vec<String> = index
                .backends()
                .iter()
                .map(|b| format!("\"{b}\":{}", index.blocks(b)))
                .collect();
            format!(
                "{{\"block_size\":{},\"recorded\":{},\"backends\":{{{}}}}}",
                index.block_size(),
                index.recorded(),
                backends.join(",")
            )
        }
        other => format!("{{\"error\":\"unknown op {other:?}\"}}"),
    }
}

/// First `"name"` value in the line, as a string.
fn field(line: &str, name: &str) -> Option<String> {
    let key = format!("\"{name}\"");
    let start = line.find(&key)? + key.len();
    let rest = line[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// First `"name"` value in the line, as a number.
fn number(line: &str, name: &str) -> Option<f64> {
    let key = format!("\"{name}\"");
    let start = line.find(&key)? + key.len();
    let rest = line[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-')).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The `"tokens":[1,2,3]` array, in order.
fn tokens_of(line: &str) -> Vec<u32> {
    let key = "\"tokens\"";
    let Some(start) = line.find(key) else { return Vec::new() };
    let rest = &line[start + key.len()..];
    let Some(open) = rest.find('[') else { return Vec::new() };
    let Some(close) = rest[open..].find(']') else { return Vec::new() };
    rest[open + 1..open + close]
        .split(',')
        .filter_map(|piece| piece.trim().parse::<u32>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fields_and_tokens() {
        let line = r#"{"op":"observe","backend":"studio","tokens":[1,2,3],"cached":32,"prompt":64}"#;
        assert_eq!(field(line, "op").as_deref(), Some("observe"));
        assert_eq!(field(line, "backend").as_deref(), Some("studio"));
        assert_eq!(number(line, "cached"), Some(32.0));
        assert_eq!(tokens_of(line), vec![1, 2, 3]);
    }

    #[test]
    fn missing_values_are_none_not_panics() {
        assert_eq!(field("{}", "op"), None);
        assert_eq!(number("{}", "cached"), None);
        assert!(tokens_of("{}").is_empty());
    }

    /// `{:?}` on a Vec prints "[1, 2, 3]" with spaces, which is not JSON the parser
    /// accepts, so tests build the array the same way a client would.
    fn json_tokens(tokens: &[u32]) -> String {
        let parts: Vec<String> = tokens.iter().map(u32::to_string).collect();
        format!("[{}]", parts.join(","))
    }

    #[test]
    fn observe_then_estimate_round_trips() {
        let mut index = PrefixIndex::new(16);
        let tokens: Vec<u32> = (0..64).collect();
        let observe = format!(
            "{{\"op\":\"observe\",\"backend\":\"s\",\"tokens\":{},\"cached\":0,\"prompt\":64}}",
            json_tokens(&tokens)
        );
        assert!(handle(&mut index, &observe).contains("\"ok\":true"));
        let estimate = format!("{{\"op\":\"estimate\",\"tokens\":{}}}", json_tokens(&tokens));
        let reply = handle(&mut index, &estimate);
        assert!(reply.contains("\"backend\":\"s\""), "{reply}");
        assert!(reply.contains("\"cached_tokens\":64"), "{reply}");
    }

    #[test]
    fn a_measurement_lowers_the_estimate_through_the_cli() {
        let mut index = PrefixIndex::new(16);
        let tokens: Vec<u32> = (0..64).collect();
        handle(&mut index, &format!(
            "{{\"op\":\"observe\",\"backend\":\"s\",\"tokens\":{},\"cached\":0,\"prompt\":64}}",
            json_tokens(&tokens)));
        handle(&mut index, &format!(
            "{{\"op\":\"observe\",\"backend\":\"s\",\"tokens\":{},\"cached\":16,\"prompt\":64}}",
            json_tokens(&tokens)));
        let reply = handle(&mut index, &format!(
            "{{\"op\":\"estimate\",\"tokens\":{}}}", json_tokens(&tokens)));
        assert!(reply.contains("\"cached_tokens\":16"), "{reply}");
        assert!(reply.contains("\"measured\":true"), "{reply}");
    }
}
