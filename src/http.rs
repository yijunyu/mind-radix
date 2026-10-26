//! Just enough HTTP/1.1 to proxy chat completions, in the standard library.
//!
//! Both services are reverse proxies for JSON and server-sent events on a trusted
//! network, so the full surface of an HTTP crate would be dead weight (N-1). Two
//! decisions keep this small enough to read:
//!
//! * **`Connection: close` on everything.** Responses then end at EOF, so no chunked
//!   decoding is needed in either direction. The Python these services replace already
//!   sent `Connection: close`, so this is parity, not a regression.
//! * **`Content-Length` request bodies only.** OpenAI-compatible clients send one, and
//!   `BaseHTTPRequestHandler` never read anything else either. A chunked request body
//!   is answered 411 rather than mis-parsed.
//!
//! No TLS: a backend is loopback or LAN, and the router is what faces clients.

use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Headers are compared case-insensitively, so they are stored lowercased.
pub type Headers = BTreeMap<String, String>;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    /// Full target as sent, query string included.
    pub target: String,
    pub headers: Headers,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
    }

    /// Path with any query string removed.
    pub fn path(&self) -> &str {
        match self.target.find('?') {
            Some(at) => &self.target[..at],
            None => &self.target,
        }
    }

    /// One query parameter, undecoded beyond `+` and `%XX`.
    pub fn query(&self, key: &str) -> Option<String> {
        let query = self.target.split_once('?')?.1;
        for pair in query.split('&') {
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            if name == key {
                return Some(percent_decode(value));
            }
        }
        None
    }

    /// The bearer token or `x-api-key`, whichever is present (R-21).
    pub fn client_key(&self) -> String {
        if let Some(auth) = self.header("authorization")
            && auth.len() >= 7 && auth[..7].eq_ignore_ascii_case("bearer ") {
                return auth[7..].trim().to_string();
            }
        self.header("x-api-key").unwrap_or("").trim().to_string()
    }
}

#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        Response {
            status,
            headers: vec![("Content-Type".into(), content_type.into())],
            body,
        }
    }

    pub fn json(status: u16, value: &serde_json::Value) -> Self {
        Response::new(status, "application/json", value.to_string().into_bytes())
    }

    /// An error in the shape OpenAI clients expect, so a client's own error handling
    /// reports something useful instead of "unparseable".
    pub fn error(status: u16, kind: &str, message: &str) -> Self {
        Response::json(
            status,
            &serde_json::json!({"error": {"type": kind, "message": message}}),
        )
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

/// What a handler returns. `Stream` exists for `text/event-stream`: the body is copied
/// through as it arrives rather than buffered, so a client sees tokens as they are
/// generated (R-18).
pub enum Reply {
    Buffered(Response),
    Stream {
        status: u16,
        headers: Vec<(String, String)>,
        source: Box<dyn Read + Send>,
    },
}

impl From<Response> for Reply {
    fn from(response: Response) -> Self {
        Reply::Buffered(response)
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

pub fn parse_request<R: BufRead>(reader: &mut R) -> io::Result<Request> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "no request line"));
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    if method.is_empty() || target.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad request line"));
    }

    let mut headers = Headers::new();
    let mut read = line.len();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        read += header.len();
        if read > MAX_HEADER_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "headers too large"));
        }
        let header = header.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if length > MAX_BODY_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "body too large"));
    }
    let mut body = vec![0u8; length];
    if length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Request { method, target, headers, body })
}

fn write_head<W: Write>(out: &mut W, status: u16, headers: &[(String, String)]) -> io::Result<()> {
    write!(out, "HTTP/1.1 {status} {}\r\n", reason(status))?;
    for (name, value) in headers {
        write!(out, "{name}: {value}\r\n")?;
    }
    Ok(())
}

fn write_reply<W: Write>(out: &mut W, reply: Reply) -> io::Result<()> {
    match reply {
        Reply::Buffered(response) => {
            write_head(out, response.status, &response.headers)?;
            write!(out, "Content-Length: {}\r\nConnection: close\r\n\r\n", response.body.len())?;
            out.write_all(&response.body)?;
            out.flush()
        }
        Reply::Stream { status, headers, mut source } => {
            write_head(out, status, &headers)?;
            write!(out, "Connection: close\r\n\r\n")?;
            out.flush()?;
            // Copy in small chunks and flush each one: an SSE client that waits for a
            // large buffer to fill sees no tokens until the generation ends.
            let mut buffer = [0u8; 4096];
            loop {
                match source.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        out.write_all(&buffer[..n])?;
                        out.flush()?;
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            Ok(())
        }
    }
}

/// Serve until the process ends, one thread per connection.
///
/// Thread-per-connection matches what these services actually do: a handful of
/// long-lived streaming requests, not ten thousand idle sockets. It is also the model
/// the Python used (`ThreadingHTTPServer`), so concurrency behaviour is unchanged.
pub fn serve<A, F>(addr: A, handler: F) -> io::Result<()>
where
    A: ToSocketAddrs,
    F: Fn(Request) -> Reply + Send + Sync + 'static,
{
    serve_on(TcpListener::bind(addr)?, handler)
}

/// Serve on a listener that is already bound.
///
/// Separate from [`serve`] so a caller can learn the port before serving: a test binds
/// `127.0.0.1:0`, reads `local_addr`, and hands the listener over. Without this, tests
/// would have to guess free ports, which is how a suite starts failing only on a busy
/// machine.
pub fn serve_on<F>(listener: TcpListener, handler: F) -> io::Result<()>
where
    F: Fn(Request) -> Reply + Send + Sync + 'static,
{
    let handler = std::sync::Arc::new(handler);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let handler = std::sync::Arc::clone(&handler);
        std::thread::spawn(move || {
            let _ = handle_connection(stream, handler.as_ref());
        });
    }
    Ok(())
}

fn handle_connection<F>(stream: TcpStream, handler: &F) -> io::Result<()>
where
    F: Fn(Request) -> Reply,
{
    // A client that opens a socket and says nothing must not hold a thread forever;
    // a generation, however, can legitimately take many minutes to write.
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3600)))?;
    let mut out = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let reply = match parse_request(&mut reader) {
        Ok(request) => {
            if request.header("transfer-encoding").is_some_and(|v| v.contains("chunked")) {
                Reply::Buffered(Response::error(411, "invalid_request_error", "chunked request body"))
            } else {
                handler(request)
            }
        }
        Err(_) => return Ok(()), // malformed or closed; nothing useful to answer
    };
    write_reply(&mut out, reply)
}

/// `http://host:port/path` split into its parts. No scheme other than `http` is
/// accepted, which is also the audit answer for R-20: a target is built from our own
/// configuration plus the client's path, never from a client-supplied URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub host: String,
    pub port: u16,
    pub path: String,
}

impl Url {
    pub fn parse(url: &str) -> Option<Url> {
        let rest = url.strip_prefix("http://")?;
        let (authority, path) = match rest.find('/') {
            Some(at) => (&rest[..at], &rest[at..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().ok()?),
            None => (authority.to_string(), 80),
        };
        if host.is_empty() {
            return None;
        }
        Some(Url { host, port, path: path.to_string() })
    }

    pub fn authority(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// An upstream response whose body has not been read yet, so a caller can either
/// buffer it or stream it.
pub struct Upstream {
    pub status: u16,
    pub headers: Headers,
    reader: BufReader<TcpStream>,
}

impl Upstream {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
    }

    pub fn content_type(&self) -> &str {
        self.header("content-type").unwrap_or("application/json")
    }

    pub fn is_event_stream(&self) -> bool {
        self.content_type().starts_with("text/event-stream")
    }

    pub fn read_body(mut self) -> io::Result<Vec<u8>> {
        let mut body = Vec::new();
        self.reader.read_to_end(&mut body)?;
        Ok(body)
    }

    pub fn into_reader(self) -> Box<dyn Read + Send> {
        Box::new(self.reader)
    }
}

/// One request to a backend. `base` is configuration; `path` may carry a client's.
pub fn fetch(
    base: &str,
    path: &str,
    method: &str,
    headers: &[(String, String)],
    body: Option<&[u8]>,
    timeout: Duration,
) -> io::Result<Upstream> {
    let url = Url::parse(base)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("bad base url {base:?}")))?;
    let target = if url.path == "/" {
        path.to_string()
    } else {
        format!("{}{}", url.path.trim_end_matches('/'), path)
    };

    // Resolve and connect with a bounded timeout: an unreachable LAN host must fail a
    // probe quickly rather than stall every request behind it (R-5).
    let addr = url
        .authority()
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no address"))?;
    let stream = TcpStream::connect_timeout(&addr, timeout.min(Duration::from_secs(10)))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let mut out = stream.try_clone()?;
    write!(out, "{method} {target} HTTP/1.1\r\nHost: {}\r\n", url.authority())?;
    for (name, value) in headers {
        write!(out, "{name}: {value}\r\n")?;
    }
    write!(out, "Content-Length: {}\r\nConnection: close\r\n\r\n", body.map_or(0, <[u8]>::len))?;
    if let Some(body) = body {
        out.write_all(body)?;
    }
    out.flush()?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "no status line"));
    }
    let status: u16 = line
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad status line"))?;

    let mut response_headers = Headers::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            response_headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    Ok(Upstream { status, headers: response_headers, reader })
}

fn percent_decode(value: &str) -> String {
    let bytes = value.replace('+', " ").into_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &str) -> Request {
        parse_request(&mut BufReader::new(raw.as_bytes())).expect("parses")
    }

    #[test]
    fn parses_a_request_with_a_body() {
        let request = parse("POST /v1/chat HTTP/1.1\r\nContent-Length: 2\r\nX-A: b\r\n\r\nhi");
        assert_eq!(request.method, "POST");
        assert_eq!(request.path(), "/v1/chat");
        assert_eq!(request.header("x-a"), Some("b"));
        assert_eq!(request.body, b"hi");
    }

    #[test]
    fn header_names_are_case_insensitive() {
        let request = parse("GET / HTTP/1.1\r\nCoNtEnT-TyPe: application/json\r\n\r\n");
        assert_eq!(request.header("content-type"), Some("application/json"));
        assert_eq!(request.header("Content-Type"), Some("application/json"));
    }

    #[test]
    fn splits_path_from_query_and_decodes_values() {
        let request = parse("POST /slots/2?action=save&name=a%20b HTTP/1.1\r\n\r\n");
        assert_eq!(request.path(), "/slots/2");
        assert_eq!(request.query("action").as_deref(), Some("save"));
        assert_eq!(request.query("name").as_deref(), Some("a b"));
        assert_eq!(request.query("missing"), None);
    }

    /// R-21: both header spellings, and a missing one is empty rather than a panic.
    #[test]
    fn reads_either_key_header() {
        let bearer = parse("GET / HTTP/1.1\r\nAuthorization: Bearer abc\r\n\r\n");
        assert_eq!(bearer.client_key(), "abc");
        let x_api = parse("GET / HTTP/1.1\r\nx-api-key: def\r\n\r\n");
        assert_eq!(x_api.client_key(), "def");
        assert_eq!(parse("GET / HTTP/1.1\r\n\r\n").client_key(), "");
    }

    #[test]
    fn bearer_prefix_is_case_insensitive() {
        let request = parse("GET / HTTP/1.1\r\nAuthorization: BEARER tok\r\n\r\n");
        assert_eq!(request.client_key(), "tok");
    }

    #[test]
    fn parses_urls_with_and_without_a_port_or_path() {
        assert_eq!(
            Url::parse("http://127.0.0.1:8010"),
            Some(Url { host: "127.0.0.1".into(), port: 8010, path: "/".into() })
        );
        assert_eq!(
            Url::parse("http://example:9/base"),
            Some(Url { host: "example".into(), port: 9, path: "/base".into() })
        );
        assert_eq!(Url::parse("http://host").map(|u| u.port), Some(80));
        assert_eq!(Url::parse("https://host"), None, "only http is accepted");
        assert_eq!(Url::parse(""), None);
    }

    #[test]
    fn writes_a_buffered_response() {
        let mut out = Vec::new();
        write_reply(&mut out, Response::new(200, "application/json", b"{}".to_vec()).into()).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(text.contains("Content-Length: 2\r\n"), "{text}");
        assert!(text.ends_with("\r\n\r\n{}"), "{text}");
    }

    #[test]
    fn streams_a_body_through_without_a_content_length() {
        let mut out = Vec::new();
        let reply = Reply::Stream {
            status: 200,
            headers: vec![("Content-Type".into(), "text/event-stream".into())],
            source: Box::new(io::Cursor::new(b"data: a\n\ndata: b\n\n".to_vec())),
        };
        write_reply(&mut out, reply).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("Content-Length"), "{text}");
        assert!(text.ends_with("data: a\n\ndata: b\n\n"), "{text}");
    }
}
