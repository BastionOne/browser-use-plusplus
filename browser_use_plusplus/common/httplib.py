import base64
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
from playwright.sync_api import Request, Response

from browser_use_plusplus.src.llm import RequestPart
from browser_use_plusplus.logger import PROXY_LOGGER_NAME
from logging import getLogger

proxy_log = getLogger(PROXY_LOGGER_NAME)


DEFAULT_INCLUDE_MIME = ["html", "script", "xml", "flash", "other_text"]
DEFAULT_INCLUDE_STATUS = ["2xx", "3xx", "4xx", "5xx"]
MAX_PAYLOAD_SIZE = 4000

def _decompress(body: bytes, headers: Dict[str, str]) -> bytes:
    """Decompress body according to Content-Encoding in headers.

    Supports gzip, deflate, br (brotli), zstd, bz2, lzma/xz.
    Unknown or unavailable encodings are ignored gracefully.
    """
    if not body:
        return body
    if not headers:
        return body

    content_encoding = headers.get("content-encoding", "").lower()
    if not content_encoding:
        return body

    encodings = [e.strip() for e in content_encoding.split(",") if e.strip()]
    # Decoders should be applied in reverse order of encodings applied by server
    data = body
    for enc in reversed(encodings):
        try:
            if enc in ("gzip", "x-gzip"):
                import gzip
                data = gzip.decompress(data)
            elif enc == "deflate":
                import zlib
                try:
                    data = zlib.decompress(data)
                except Exception:
                    # Try raw DEFLATE stream
                    data = zlib.decompress(data, -zlib.MAX_WBITS)
            elif enc == "br":
                try:
                    import brotli  # type: ignore
                    data = brotli.decompress(data)
                except Exception:
                    try:
                        import brotlicffi  # type: ignore
                        data = brotlicffi.decompress(data)
                    except Exception:
                        # brotli not available; leave as-is
                        pass
            elif enc in ("zstd", "zstandard"):
                try:
                    import zstandard as zstd  # type: ignore
                    dctx = zstd.ZstdDecompressor()
                    data = dctx.decompress(data)
                except Exception:
                    # zstd not available; leave as-is
                    pass
            elif enc in ("bzip2", "bz2"):
                import bz2
                data = bz2.decompress(data)
            elif enc in ("xz", "lzma"):
                import lzma
                data = lzma.decompress(data)
            elif enc == "identity":
                # No-op
                data = data
            else:
                # Unknown encoding; stop trying further to avoid corruption
                data = data
        except Exception:
            # If any decoder fails, keep current data unchanged and continue
            data = data
    return data


def parse_res_data(body_obj: Any, headers: Dict[str, str]) -> Union[str, Dict[str, Any], bytes]:
    """Return body as JSON (dict) if possible, else string, else bytes.

    The function will attempt to:
    1) Parse JSON and return a dict when possible
    2) Fallback to a plain UTF-8 string
    3) Finally return raw bytes if it's binary
    """
    if body_obj is None:
        return ""

    def is_textual_content_type(hdrs: Dict[str, str]) -> bool:
        if not hdrs:
            return False
        ct = hdrs.get("content-type", "").lower()
        if not ct:
            return False
        textual_indicators = (
            "text/",
            "json",
            "+json",
            "xml",
            "javascript",
            "x-www-form-urlencoded",
            "html",
            "csv",
        )
        return any(ind in ct for ind in textual_indicators)

    # Already a dict-like payload
    if isinstance(body_obj, dict):
        return body_obj

    # Bytes path
    if isinstance(body_obj, (bytes, bytearray)):
        raw_bytes = bytes(body_obj)
        try:
            raw_bytes = _decompress(raw_bytes, headers)
        except Exception:
            pass

        # Try JSON first
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Fallback to string if textual
        if is_textual_content_type(headers):
            try:
                return raw_bytes.decode("utf-8", errors="replace")
            except Exception:
                return raw_bytes

        # Otherwise return binary
        return raw_bytes

    # String path
    if isinstance(body_obj, str):
        s = body_obj
        # Try JSON first when it looks like JSON
        stripped = s.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return s

    # Fallback for other types
    try:
        return str(body_obj)
    except Exception:
        return ""

def parse_post_data(post_data: str | None):
    """Convert post data to dictionary format.
    
    Args:
        post_data: Raw post data string
        
    Returns:
        Dictionary of post data parameters
    """
    if not post_data:
        return {}
    
    result = {}
    
    # Handle URL-encoded form data (param1=value1&param2=value2)
    if isinstance(post_data, str):
        if "&" in post_data:
            pairs = post_data.split("&")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    result[key] = value
        # Try to parse as JSON
        elif post_data.strip().startswith("{") and post_data.strip().endswith("}"):
            try:
                result = json.loads(post_data)
            except json.JSONDecodeError:
                # Not valid JSON, return as is
                pass
    
    # if not result and len(post_data) > 0:
        # proxy_log.warning(f"Failed to parse post data: {post_data}")
        # result = {"error": "Failed to parse post data"}

    return result

class ResourceLocator(BaseModel):
    """How to locate a particular resource id in a template request."""
    id: str
    request_part: RequestPart
    type_name: str

# TODO: support for CSRF tokens
class AuthSession(BaseModel):
    headers: Dict[str, str]
    body: Optional[Dict[str, str]] = None

class HTTPRequestData(BaseModel):
    """Internal representation of HTTP request data"""
    method: str
    url: str
    headers: Dict[str, str]
    post_data: Optional[Dict] = None
    redirected_from_url: Optional[str] = ""
    redirected_to_url: Optional[str] = ""
    is_iframe: bool = False

class HTTPRequest(BaseModel):
    """HTTP request class with unified implementation"""
    data: HTTPRequestData
    auth_session: Optional[Any] = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def init_auth_session(self):
        if self.auth_session is None:
            self.auth_session = AuthSession(
                headers=self.headers,
                body=None
            )
        return self

    @property
    def method(self) -> str:
        return self.data.method

    @property
    def url(self) -> str:
        return self.data.url

    @property
    def headers(self) -> Dict[str, str]:
        return self.data.headers

    @property
    def post_data(self) -> Optional[Dict]:
        return self.data.post_data

    def get_body(self) -> Union[str, Dict[str, Any], bytes]:
        """Return request body as dict/string/bytes.

        For requests we typically store structured dicts built from the
        Playwright post_data, so return that when available; otherwise empty string.
        """
        if self.data.post_data is None:
            return ""
        return self.data.post_data

    @property
    def redirected_from(self) -> Optional["HTTPRequest"]:
        if self.data.redirected_from_url:
            # Create minimal request object for redirect
            data = HTTPRequestData(
                method="",
                url=self.data.redirected_from_url,
                headers={},
                post_data=None,
                redirected_from_url=None,
                redirected_to_url=None,
                is_iframe=False
            )
            return HTTPRequest(data=data)
        return None

    @property
    def redirected_to(self) -> Optional["HTTPRequest"]:
        if self.data.redirected_to_url:
            # Create minimal request object for redirect
            data = HTTPRequestData(
                method="",
                url=self.data.redirected_to_url,
                headers={},
                post_data=None,
                redirected_from_url=None,
                redirected_to_url=None,
                is_iframe=False
            )
            return HTTPRequest(data=data)
        return None

    @property
    def is_iframe(self) -> bool:
        return self.data.is_iframe

    def to_json(self) -> Dict[str, Any]:
        # return {
        #     "method": self.method,
        #     "url": self.url,
        #     "headers": self.headers,
        #     "post_data": self.post_data,
        #     "redirected_from": self.data.redirected_from_url,
        #     "redirected_to": self.data.redirected_to_url,
        #     "is_iframe": self.is_iframe
        # }
        return self.model_dump()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "HTTPRequest":
        return cls(**data)

    @classmethod
    def from_pw(cls, request: Request) -> "HTTPRequest":
        request_data = HTTPRequestData(
            method=request.method,
            url=request.url,
            headers=dict(request.headers),
            post_data=parse_post_data(request.post_data),
            redirected_from_url=request.redirected_from.url if request.redirected_from else None,
            redirected_to_url=request.redirected_to.url if request.redirected_to else None,
            is_iframe=bool(request.frame.parent_frame)
        )
        return cls(data=request_data)

    def to_str(self) -> str:
        """String representation of HTTP request"""
        req_str = "[Request]: \n"
        req_str += str(self.method) + " " + str(self.url) + "\n"
        
        if self.redirected_from:
            req_str += "Redirected from: " + str(self.redirected_from.url) + "\n"
        if self.redirected_to:
            req_str += "Redirecting to: " + str(self.redirected_to.url) + "\n"
        if self.is_iframe:
            req_str += "From iframe\n"

        for k,v in self.headers.items():
            req_str += f"{k} : {v}\n"
            
        req_str += str(self.post_data)
        return req_str
    
    def __hash__(self):
        """Hash function for HTTPRequest"""
        return hash((self.method, self.url, str(self.post_data), str(self.headers)))

class HTTPResponseData(BaseModel):
    """Internal representation of HTTP response data"""
    url: str
    status: int
    headers: Dict[str, str]
    is_iframe: bool
    body: Optional[bytes] = None
    body_error: Optional[str] = None
    content_type: Optional[str] = None

    def get_body(self) -> Union[str, Dict[str, Any], bytes]:
        return parse_res_data(self.body, self.headers)

class HTTPResponse(BaseModel):
    """HTTP response class with unified implementation"""
    data: HTTPResponseData

    @property
    def url(self) -> str:
        return self.data.url

    @property
    def status(self) -> int:
        return self.data.status

    @property
    def headers(self) -> Dict[str, str]:
        return self.data.headers

    @property
    def is_iframe(self) -> bool:
        return self.data.is_iframe

    def get_body(self) -> Union[str, Dict[str, Any], bytes]:
        if self.data.body_error:
            raise Exception(self.data.body_error)
        if self.data.body is None:
            raise Exception("Response body not available")
        return self.data.get_body()

    def get_content_type(self) -> str:
        """Get content type from response headers"""
        if not self.headers:
            return ""
        content_type = self.headers.get("content-type", "")
        return content_type.lower()
    
    def get_status_code(self) -> int:
        """Get HTTP status code"""
        if not self.status:
            return 0
        return self.status
    
    def get_response_size(self) -> int:
        """Get response payload size in bytes"""
        if not self.headers:
            return 0
        content_length = self.headers.get("content-length")
        if content_length and content_length.isdigit():
            return int(content_length)
        return 0

    async def to_json(self) -> Dict[str, Any]:
        json_data = {
            "url": self.url,
            "status": self.status,
            "headers": self.headers,
            "content_type": self.get_content_type(),
            "content_length": self.get_response_size(),
            "is_iframe": self.is_iframe
        }

        if not (300 <= self.status < 400):
            if self.data.body_error:
                json_data["body_error"] = self.data.body_error
            elif self.data.body:
                json_data["body"] = str(self.data.body)

        return {
            "data": json_data,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "HTTPResponse":
        return cls(**data)

    @classmethod
    def from_pw(cls, response: Response) -> "HTTPResponse":
        response_data = HTTPResponseData(
            url=response.url,
            status=response.status,
            headers=dict(response.headers),
            is_iframe=bool(response.frame.parent_frame)
        )
        return cls(data=response_data)

    async def to_str(self) -> str:
        """String representation of HTTP response"""
        resp_str = "[Response]: " + str(self.url) + " " + str(self.status) + "\n"
        
        if self.is_iframe:
            resp_str += "From iframe\n"
        resp_str += str(self.headers) + "\n"
        
        if 300 <= self.status < 400:
            resp_str += "[Redirect response - no body]"
            return resp_str
            
        try:
            body_value = self.get_body()
            resp_str += str(body_value)
        except Exception as e:
            resp_str += f"[Error getting response body: {str(e)}]"
            
        return resp_str

    def __hash__(self):
        """Hash function for HTTPResponse"""
        return hash((self.url, self.status, str(self.headers), str(self.data.body)))

class HTTPMessage(BaseModel):
    """Encapsulates a request/response pair"""
    request: HTTPRequest 
    response: Optional[HTTPResponse] = None

    @property
    def url(self):
        return self.request.url
    
    @property
    def method(self):
        return self.request.method
    
    @property
    def body(self):
        return self.request.post_data
    
    @property
    def id(self):
        return f"{self.method} {self.url}\n{self.body}"
    
    async def to_str(self) -> str:
        req_str = str(self.request)
        resp_str = await self.response.to_str() if self.response else ""
        return f"{req_str}\n{resp_str}"

    async def to_json(self) -> Dict[str, Any]:
        json_data = {
            "request": self.request.to_json()
        }
        if self.response:
            json_data["response"] = await self.response.to_json()
        return json_data
    
    async def to_payload(self):
        payload = {
            "request": self.request.to_json()
        }
        if self.response:
            payload["response"] = await self.response.to_json()

        return payload  
          
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "HTTPMessage":
        request = HTTPRequest.from_json(data["request"])
        response = HTTPResponse.from_json(data["response"]) if data.get("response") else None
        return cls(request=request, response=response)

def parse_burp_headers(raw_headers: str) -> Dict[str, str]:
    """Parse HTTP headers from a raw string into a dictionary"""
    headers = {}
    lines = raw_headers.split('\n')
    
    # Skip the first line (HTTP method line) for request headers
    start_line = 1 if lines and (lines[0].startswith('GET') or lines[0].startswith('POST')) else 0
    
    for line in lines[start_line:]:
        if not line.strip():
            continue
        parts = line.split(':', 1)
        if len(parts) == 2:
            key, value = parts
            headers[key.strip().lower()] = value.strip()
    
    return headers

def parse_burp_request(request_text: str, is_base64: bool, url: str, method: str) -> HTTPRequest:
    """Parse raw HTTP request data into a HTTPRequest object"""
    if is_base64:
        try:
            request_text = base64.b64decode(request_text).decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error decoding base64 request: {e}")
            request_text = ""
    
    # Split headers and body
    parts = request_text.split('\n\n', 1)
    headers_text = parts[0]
    post_data = {}

    if method == "POST":
        post_payload = request_text.split("\r\n\r\n")[1]
        if "&" in post_payload:
            kv_pairs = post_payload.split("&")
            for k,v in [kv.split("=") for kv in kv_pairs]:
                post_data[k] = v       
            
    headers = parse_burp_headers(headers_text)
    # Create request data
    request_data = HTTPRequestData(
        method=method,
        url=url,
        headers=headers,
        post_data=post_data,
        redirected_from_url=None,  # No redirect info in Burp export
        redirected_to_url=None,    # No redirect info in Burp export
        is_iframe=False            # No iframe info in Burp export
    )
    return HTTPRequest(data=request_data)

def parse_burp_response(response_text: str, is_base64: bool, url: str, status: int) -> HTTPResponse:
    """Parse raw HTTP response data into a HTTPResponse object"""
    body = None
    body_error = None
    
    if is_base64:
        decoded_text = base64.b64decode(response_text)
        # Find the empty line that separates headers from body
        headers_end = decoded_text.find(b'\r\n\r\n')
        if headers_end != -1:
            headers_text = decoded_text[:headers_end].decode('utf-8', errors='replace')
            body = decoded_text[headers_end + 4:]  # Skip \r\n\r\n
        else:
            headers_text = decoded_text.decode('utf-8', errors='replace')
    else:
        parts = response_text.split('\n\n', 1)
        headers_text = parts[0]
        body = parts[1].encode('utf-8') if len(parts) > 1 else None
    
    headers = parse_burp_headers(headers_text)
    
    # Create response data
    response_data = HTTPResponseData(
        url=url,
        status=status,
        headers=headers,
        is_iframe=False,  # No iframe info in Burp export
        body=body,
        body_error=body_error
    )
    
    return HTTPResponse(data=response_data)

def parse_burp_xml(filepath: str) -> List[HTTPMessage]:
    """Parse a Burp Suite XML export file into an HTTPMessageList"""
    # Read the XML content
    with open(filepath, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Extract actual XML content if it's wrapped in document tags
    if "<document_content>" in xml_content:
        start = xml_content.find("<document_content>") + len("<document_content>")
        end = xml_content.find("</document_content>")
        xml_content = xml_content[start:end]

    root = ET.fromstring(xml_content)
    messages = []
    
    for item in root.findall(".//item"):
        # Extract basic information
        url_elem = item.find("url")
        method_elem = item.find("method")
        status_elem = item.find("status")
        url = url_elem.text if url_elem is not None and url_elem.text is not None else ""
        method = method_elem.text if method_elem is not None and method_elem.text is not None else ""
        status = int(status_elem.text) if status_elem is not None and status_elem.text is not None else 0
        
        # Parse request
        request_elem = item.find("request")
        is_request_base64 = (request_elem.get("base64") == "true") if request_elem is not None else False
        request_text = request_elem.text if request_elem is not None and request_elem.text is not None else ""
        request = parse_burp_request(request_text, is_request_base64, url, method)
        
        # Parse response
        response = None
        response_elem = item.find("response")
        if response_elem is not None and response_elem.text:
            is_response_base64 = (response_elem.get("base64") == "true") if response_elem is not None else False
            response_text = response_elem.text if response_elem.text is not None else ""
            response = parse_burp_response(response_text, is_response_base64, url, status)
        
        # Create HTTP message
        message = HTTPMessage(request=request, response=response)
        messages.append(message)
    
    
    return messages
