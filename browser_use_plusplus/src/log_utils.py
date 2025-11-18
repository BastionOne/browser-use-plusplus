import contextvars
from contextlib import contextmanager
from typing import Iterator, List
import difflib
import logging
import tiktoken

# The one and only ContextVar you export  –  anything else stays private
_log_context_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_context_id",
    default="-",
)

# Public helpers -------------------------------------------------------------
def get_ctxt_id() -> str:              # for the formatter
    return _log_context_id.get()

def set_ctxt_id(value: str) -> None:   # for your app code
    _log_context_id.set(value)

@contextmanager
def push_ctxt_id(value: str) -> Iterator[None]:
    """
    Temporary override of the current context ID.

    >>> with push("http-42"):
    ...     ...
    """
    token = _log_context_id.set(value)
    try:
        yield
    finally:
        _log_context_id.reset(token)

def get_token_count(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_state_from_history(history):
    """
    Extract all current_state keys from the history dictionary and return as a list.
    
    Args:
        history (dict): A dictionary containing history data with model_output entries
        
    Returns:
        list: A list of all current_state dictionaries found in the history
    """
    states = []
    
    if not history or "history" not in history:
        return states
    
    for entry in history["history"]:
        if "model_output" in entry and "current_state" in entry["model_output"]:
            states.append(entry["model_output"]["current_state"])
    
    return states

def diff_dom(a: str, b: str) -> str:
    diff = difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile="a.txt",
        tofile="b.txt",
        lineterm=""
    )
    return "".join(diff)


class LoggerProxy:
    def __init__(self, loggers):
        self.loggers: List[logging.Logger] = loggers
    
    def info(self, msg: str, *args, **kwargs):
        for logger in self.loggers:
            logger.info(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        for logger in self.loggers:
            logger.debug(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        for logger in self.loggers:
            logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        for logger in self.loggers:
            logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        for logger in self.loggers:
            logger.critical(msg, *args, **kwargs)


import mimetypes
import re
from urllib.parse import urlparse
from typing import Optional, Tuple

class ContentTypeDetector:
    """
    Enhanced content type detection for Playwright responses with multiple fallback strategies.
    """
    
    # Common file extension to MIME type mappings for web resources
    EXTENSION_MIME_MAP = {
        # Web documents
        '.html': 'text/html',
        '.htm': 'text/html',
        '.xhtml': 'application/xhtml+xml',
        '.xml': 'application/xml',
        
        # Scripts and data
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.json': 'application/json',
        '.jsonld': 'application/ld+json',
        '.yaml': 'application/x-yaml',
        '.yml': 'application/x-yaml',
        
        # Styles
        '.css': 'text/css',
        '.scss': 'text/x-scss',
        '.sass': 'text/x-sass',
        '.less': 'text/x-less',
        
        # Images
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.bmp': 'image/bmp',
        '.avif': 'image/avif',
        
        # Fonts
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.otf': 'font/otf',
        '.eot': 'application/vnd.ms-fontobject',
        
        # Documents
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        
        # Archives
        '.zip': 'application/zip',
        '.gz': 'application/gzip',
        '.tar': 'application/x-tar',
        '.rar': 'application/x-rar-compressed',
        '.7z': 'application/x-7z-compressed',
        
        # Media
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        
        # Other web formats
        '.wasm': 'application/wasm',
        '.map': 'application/json',  # Source maps
        '.webmanifest': 'application/manifest+json',
        '.rss': 'application/rss+xml',
        '.atom': 'application/atom+xml',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.tsv': 'text/tab-separated-values',
        '.md': 'text/markdown',
        '.sh': 'application/x-sh',
        '.sql': 'application/sql',
    }
    
    # Magic bytes for common file types (first few bytes of file)
    MAGIC_BYTES = {
        # Images
        b'\xff\xd8\xff': 'image/jpeg',
        b'\x89PNG\r\n\x1a\n': 'image/png',
        b'GIF87a': 'image/gif',
        b'GIF89a': 'image/gif',
        b'RIFF': 'image/webp',  # WebP starts with RIFF
        b'<svg': 'image/svg+xml',
        b'<?xml': 'application/xml',  # Could be SVG or other XML
        
        # Documents
        b'%PDF': 'application/pdf',
        b'PK\x03\x04': 'application/zip',  # Also covers DOCX, XLSX, etc.
        b'PK\x05\x06': 'application/zip',  # Empty archive
        b'PK\x07\x08': 'application/zip',  # Spanned archive
        
        # Web formats
        b'<!DOCTYPE html': 'text/html',
        b'<!doctype html': 'text/html',
        b'<html': 'text/html',
        b'<HTML': 'text/html',
        b'{"': 'application/json',
        b'[{': 'application/json',
        b'{"@': 'application/ld+json',  # JSON-LD often starts with @context
        
        # Scripts
        b'#!/usr/bin/env node': 'application/javascript',
        b'#!/usr/bin/env python': 'text/x-python',
        b'(function': 'application/javascript',
        b'var ': 'application/javascript',
        b'const ': 'application/javascript',
        b'let ': 'application/javascript',
        
        # Binary formats
        b'\x1f\x8b': 'application/gzip',
        b'BM': 'image/bmp',
        b'II\x2a\x00': 'image/tiff',
        b'MM\x00\x2a': 'image/tiff',
        b'wOFF': 'font/woff',
        b'wOF2': 'font/woff2',
        b'\x00\x01\x00\x00': 'font/ttf',
        b'OTTO': 'font/otf',
    }
    
    # URL patterns that suggest content type
    URL_PATTERNS = [
        (re.compile(r'/api/.*\.json$|/api/v\d+/|/graphql|/rest/'), 'application/json'),
        (re.compile(r'/ws$|/websocket|/socket\.io/'), 'websocket'),
        (re.compile(r'/stream$|/events$'), 'text/event-stream'),
        (re.compile(r'\.aspx?$|\.php$|\.jsp$'), 'text/html'),
        (re.compile(r'/feed$|/rss$'), 'application/rss+xml'),
        (re.compile(r'/manifest\.json$'), 'application/manifest+json'),
        (re.compile(r'\.map$'), 'application/json'),  # Source maps
    ]
    
    @classmethod
    def detect_content_type(cls, response, url: str = None) -> str:
        """
        Detect content type using multiple strategies in order of reliability.
        
        Args:
            response: Playwright response object
            url: Optional URL override (if None, uses response.url)
            
        Returns:
            Detected MIME type string, defaults to 'application/octet-stream' if unknown
        """
        url = url or response.url
        
        # Strategy 1: Check actual Content-Type header
        content_type = cls._get_content_type_header(response)
        if content_type and content_type != "":
            return content_type.lower()
        
        # Strategy 2: Check response body magic bytes (if accessible)
        body_type = cls._check_magic_bytes(response)
        if body_type:
            return body_type
        
        # Strategy 3: Check file extension from URL
        ext_type = cls._check_file_extension(url)
        if ext_type:
            return ext_type
        
        # Strategy 4: Check URL patterns
        pattern_type = cls._check_url_patterns(url)
        if pattern_type:
            return pattern_type
        
        # Strategy 5: Check if response looks like JSON/HTML based on first few chars
        content_hint = cls._analyze_response_start(response)
        if content_hint:
            return content_hint
        
        # Strategy 6: Use Python's mimetypes as fallback
        guessed_type, _ = mimetypes.guess_type(url)
        if guessed_type:
            return guessed_type
        
        # Default fallback
        return 'application/octet-stream'
    
    @classmethod
    def _get_content_type_header(cls, response) -> Optional[str]:
        """Extract Content-Type header from response."""
        try:
            headers = response.headers
            content_type = headers.get('content-type', '')
            if content_type:
                # Remove charset and other parameters
                return content_type.split(';')[0].strip()
        except:
            pass
        return None
    
    @classmethod
    def _check_magic_bytes(cls, response) -> Optional[str]:
        """Check magic bytes at the start of response body."""
        try:
            # Try to get response body (may not always be available)
            body = response.body()
            if body:
                # Check first 20 bytes against known signatures
                start = body[:20]
                for magic, mime_type in cls.MAGIC_BYTES.items():
                    if start.startswith(magic):
                        return mime_type
        except:
            # Body might not be accessible
            pass
        return None
    
    @classmethod
    def _check_file_extension(cls, url: str) -> Optional[str]:
        """Check file extension from URL."""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Remove query parameters and fragments
            path = path.split('?')[0].split('#')[0]
            
            for ext, mime_type in cls.EXTENSION_MIME_MAP.items():
                if path.endswith(ext):
                    return mime_type
        except:
            pass
        return None
    
    @classmethod
    def _check_url_patterns(cls, url: str) -> Optional[str]:
        """Check URL against known patterns."""
        try:
            for pattern, mime_type in cls.URL_PATTERNS:
                if pattern.search(url):
                    return mime_type
        except:
            pass
        return None
    
    @classmethod
    def _analyze_response_start(cls, response) -> Optional[str]:
        """Analyze first few characters of response for content hints."""
        try:
            body = response.body()
            if body:
                # Decode first 100 chars if possible
                try:
                    start = body[:100].decode('utf-8', errors='ignore').strip()
                    
                    # Check for common patterns
                    if start.startswith('{') or start.startswith('['):
                        return 'application/json'
                    elif start.startswith('<!DOCTYPE') or start.startswith('<html'):
                        return 'text/html'
                    elif start.startswith('<?xml'):
                        return 'application/xml'
                    elif start.startswith('<svg'):
                        return 'image/svg+xml'
                    elif re.match(r'^[\w\-]+\s*:\s*[\w\-]+', start):
                        # Looks like CSS
                        return 'text/css'
                except:
                    pass
        except:
            pass
        return None
    
    @classmethod
    def get_content_info(cls, response, url: str = "") -> Tuple[str, str]:
        """
        Get both the detected content type and a confidence level.
        
        Returns:
            Tuple of (content_type, detection_method)
        """
        url = url or response.url
        
        # Try each method and track which one succeeded
        content_type = cls._get_content_type_header(response)
        if content_type and content_type != "":
            return content_type.lower(), "header"
        
        body_type = cls._check_magic_bytes(response)
        if body_type:
            return body_type, "magic_bytes"
        
        ext_type = cls._check_file_extension(url)
        if ext_type:
            return ext_type, "file_extension"
        
        pattern_type = cls._check_url_patterns(url)
        if pattern_type:
            return pattern_type, "url_pattern"
        
        content_hint = cls._analyze_response_start(response)
        if content_hint:
            return content_hint, "content_analysis"
        
        guessed_type, _ = mimetypes.guess_type(url)
        if guessed_type:
            return guessed_type, "mimetypes_guess"
        
        return 'application/octet-stream', "default"