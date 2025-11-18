#!/usr/bin/env python
# Python 3
# LinkFinder
# By Gerben_Javado

# Fix webbrowser bug for MacOS
import os
from typing import Dict, Any, List

os.environ["BROWSER"] = "open"

# Import libraries
import re
import sys
import glob
import html
import argparse
import jsbeautifier
import webbrowser
import subprocess
import base64
import ssl
import xml.etree.ElementTree

from gzip import GzipFile
from string import Template
from logging import getLogger

# from browser_use_plusplus.common.httplib import HTTPMessage

try:
    from StringIO import StringIO

    readBytesCustom = StringIO
except ImportError:
    from io import BytesIO

    readBytesCustom = BytesIO

try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib2 import Request, urlopen


logger = getLogger(__name__)

# Regex used
regex_str = r"""

  (?:"|')                               # Start newline delimiter

  (
    (/                                  # Start with /
    [^"'><,;| *()(%%$^/\\\[\]]          # Next character can't be...
    [^"'><,;|()]{1,})                   # Rest of the characters can't be

    |

    (/[a-zA-Z0-9_\-/]{1,}/              # Relative endpoint with /
    [a-zA-Z0-9_\-/.]{1,}                # Resource name
    \.(?:[a-zA-Z]{1,4}|action)          # Rest + extension (length 1-4 or action)
    (?:[\?|#][^"|']{0,}|))              # ? or # mark with parameters

    |

    (/[a-zA-Z0-9_\-/]{1,}/              # REST API (no extension) with /
    [a-zA-Z0-9_\-/]{3,}                 # Proper REST endpoints usually have 3+ chars
    (?:[\?|#][^"|']{0,}|))              # ? or # mark with parameters
  )

  (?:"|')                               # End newline delimiter

"""

context_delimiter_str = "\n"


def parser_error(errmsg):
    """
    Error Messages
    """
    print("Usage: python %s [Options] use -h for help" % sys.argv[0])
    print("Error: %s" % errmsg)
    sys.exit()

    
def parser_input(input):
    """
    Parse Input
    """

    # Method 1 - URL
    if input.startswith(("http://", "https://", "file://", "ftp://", "ftps://")):
        return [input]

    # Method 2 - URL Inspector Firefox
    if input.startswith("view-source:"):
        return [input[12:]]

    # Method 3 - Burp file
    if args.burp:
        jsfiles = []
        items = xml.etree.ElementTree.fromstring(open(args.input, "r").read())

        for item in items:
            jsfiles.append(
                {
                    "js": base64.b64decode(item.find("response").text).decode(
                        "utf-8", "replace"
                    ),
                    "url": item.find("url").text,
                }
            )
        return jsfiles

    # Method 4 - Folder with a wildcard
    if "*" in input:
        paths = glob.glob(os.path.abspath(input))
        file_paths = [p for p in paths if os.path.isfile(p)]
        for index, path in enumerate(file_paths):
            file_paths[index] = "file://%s" % path
        return (
            file_paths
            if len(file_paths) > 0
            else parser_error(
                "Input with wildcard does \
        not match any files."
            )
        )

    # Method 5 - Local file
    path = "file://%s" % os.path.abspath(input)
    return [
        (
            path
            if os.path.exists(input)
            else parser_error(
                "file could not \
be found (maybe you forgot to add http/https)."
            )
        )
    ]


def send_request(url):
    """
    Send requests with Requests
    """
    q = Request(url)

    q.add_header(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    )
    q.add_header(
        "Accept",
        "text/html,\
        application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    )
    q.add_header("Accept-Language", "en-US,en;q=0.8")
    q.add_header("Accept-Encoding", "gzip")
    q.add_header("Cookie", args.cookies)

    try:
        sslcontext = ssl.create_default_context()
        response = urlopen(q, timeout=args.timeout, context=sslcontext)
    except:
        sslcontext = ssl.create_default_context()
        response = urlopen(q, timeout=args.timeout, context=sslcontext)

    if response.info().get("Content-Encoding") == "gzip":
        data = GzipFile(fileobj=readBytesCustom(response.read())).read()
    elif response.info().get("Content-Encoding") == "deflate":
        data = response.read().read()
    else:
        data = response.read()

    return data.decode("utf-8", "replace")


def getContext(list_matches, content, include_delimiter=0, context_delimiter_str="\n"):
    """
    Parse Input
    list_matches:       list of tuple (link, start_index, end_index)
    content:            content to search for the context
    include_delimiter   Set 1 to include delimiter in context
    """
    items = []
    for m in list_matches:
        match_str = m[0]
        match_start = m[1]
        match_end = m[2]
        context_start_index = match_start
        context_end_index = match_end
        delimiter_len = len(context_delimiter_str)
        content_max_index = len(content) - 1

        while (
            content[context_start_index] != context_delimiter_str
            and context_start_index > 0
        ):
            context_start_index = context_start_index - 1

        while (
            content[context_end_index] != context_delimiter_str
            and context_end_index < content_max_index
        ):
            context_end_index = context_end_index + 1

        if include_delimiter:
            context = content[context_start_index:context_end_index]
        else:
            context = content[context_start_index + delimiter_len : context_end_index]

        item = {"link": match_str, "context": context}
        items.append(item)

    return items


def parse_links(content, regex_str, mode=1, more_regex=None, no_dup=1):
    """
    Parse Input
    content:    string of content to be searched
    regex_str:  string of regex (The link should be in the group(1))
    mode:       mode of parsing. Set 1 to include surrounding contexts in the result
    more_regex: string of regex to filter the result
    no_dup:     remove duplicated link (context is NOT counted)

    Return the list of ["link": link, "context": context]
    The context is optional if mode=1 is provided.
    """
    global context_delimiter_str

    if mode == 1:
        # Beautify
        if len(content) > 1000000:
            content = content.replace(";", ";\r\n").replace(",", ",\r\n")
        else:
            content = jsbeautifier.beautify(content)

    regex = re.compile(regex_str, re.VERBOSE)

    if mode == 1:
        all_matches = [
            (m.group(1), m.start(0), m.end(0)) for m in re.finditer(regex, content)
        ]
        items = getContext(
            all_matches, content, context_delimiter_str=context_delimiter_str
        )
    else:
        items = [{"link": m.group(1)} for m in re.finditer(regex, content)]

    if no_dup:
        # Remove duplication
        all_links = set()
        no_dup_items = []
        for item in items:
            if item["link"] not in all_links:
                all_links.add(item["link"])
                no_dup_items.append(item)
        items = no_dup_items

    # Match Regex
    filtered_items = []
    for item in items:
        # Remove other capture groups from regex results
        if more_regex:
            if re.search(more_regex, item["link"]):
                filtered_items.append(item)
        else:
            filtered_items.append(item)

    return filtered_items

def parse_links_from_str(content: str) -> List[str]:
    return [
        item["link"] for item in parse_links(content, regex_str, mode=0, no_dup=1)
    ]

def cli_output(endpoints):
    """
    Output to CLI
    """
    for endpoint in endpoints:
        print(html.escape(endpoint["link"]).encode("ascii", "ignore").decode("utf8"))


def html_save(html):
    """Save *html* to ``args.output``.

    If a ``template.html`` file exists in the same directory as *linkFinder.py*
    it is used as a Jinja-style template with a ``$content`` placeholder.
    When the template is absent we fall back to a minimal built-in page.  This
    prevents FileNotFoundError crashes when the module is distributed without
    the optional template file.
    """

    from pathlib import Path

    output_path = Path(args.output)
    template_path = Path(__file__).with_name("template.html")

    # Build final HTML document
    if template_path.exists():
        try:
            tmpl = Template(template_path.read_text(encoding="utf8"))
            final_html = tmpl.substitute(content=html)
        except Exception as exc:  # pragma: no cover – guard against template bugs
            logger.warning("Template parsing failed – falling back to default. %s", exc)
            final_html = f"<html><body><pre>{html}</pre></body></html>"
    else:
        # Fallback minimal template
        final_html = (
            "<html><head><meta charset='utf-8'><title>LinkFinder Output"
            "</title></head><body>" + html + "</body></html>"
        )

    try:
        output_path.write_text(final_html, encoding="utf8")
        print(f"Output saved to: {output_path.resolve()}")
        # Optionally open in browser only if a GUI is present
        try:
            if sys.platform.startswith("linux"):
                subprocess.call(
                    ["xdg-open", str(output_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sys.platform.startswith("win"):
                os.startfile(output_path)  # type: ignore[attr-defined]
            else:
                webbrowser.open(output_path.as_uri())
        except Exception:
            pass  # Ignore browser launch issues
    except Exception as exc:
        logger.error("Failed to save HTML output to %s – %s", output_path, exc)


def check_url(url):
    nopelist = ["node_modules", "jquery.js"]
    if url[-3:] == ".js":
        words = url.split("/")
        for word in words:
            if word in nopelist:
                return False
        if url[:2] == "//":
            url = "https:" + url
        if url[:4] != "http":
            if url[:1] == "/":
                url = args.input + url
            else:
                url = args.input + "/" + url
        return url
    else:
        return False


# TODO: use the links with context to try to filter for a better api endpoint capturing regex
def detect_links(results: Dict[str, Any]) -> List[str]:
    """Extract potential endpoints from JS files referenced in *results*.

    The *results* dict is produced by ``detect_js`` and contains a list of
    ``HTTPMessage`` objects under the key ``http_messages``.  Each message
    exposes at minimum a ``url`` attribute.  For every JavaScript URL we
    download the file (with a small timeout) and run the same regex-based
    extraction that LinkFinder uses in CLI mode.  A **deduplicated** list of
    endpoint strings is returned.

    The implementation is self-contained – it does **not** depend on the
      command-line ``args`` object defined when the module is executed as a
    script.  That means it can be imported and called from other modules
    (e.g. *cli.py*) without any extra setup.
    """

    http_msgs: List[str] = results.get("http_messages", [])
    if not http_msgs:
        return results

    import httpx  # Local import to avoid mandatory dependency for CLI usage

    endpoints: set[str] = set()

    for url in http_msgs:
        # if not url or not url.endswith(".js"):
        #     continue

        logger.info("Running LinkFinder against: %s", url)

        # ------------------------------------------------------------------
        # 1. Fetch JS file (best-effort; skip on errors)
        # ------------------------------------------------------------------
        try:
            resp = httpx.get(url, timeout=10.0, follow_redirects=True)
            resp.raise_for_status()
            content = resp.text
        except Exception as exc:
            logger.warning("LinkFinder: failed to download %s – %s", url, exc)
            continue

        # ------------------------------------------------------------------
        # 2. Parse with existing LinkFinder helpers
        #    We use mode=0 to skip beautification & context collection – we only
        #    need the raw links.
        # ------------------------------------------------------------------
        try:
            parsed_items = parse_links(content, regex_str, mode=0, no_dup=1)
            for item in parsed_items:
                endpoints.add(item["link"])
        except Exception as exc:
            logger.exception("LinkFinder: parsing error for %s – %s", url, exc)
            continue

    return {**results, "links": list(endpoints)}


def detect_links_from_url(url: str) -> List[str]:
    """
    Convenience wrapper around `detect_links` that processes a single
    JavaScript URL.

    The function constructs the minimal *results* dict expected by
    `detect_links` and returns the list of extracted endpoint strings.
    """
    enriched = detect_links({"http_messages": [url]})
    return enriched.get("links", [])


if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        help="JavaScript URL to analyze for endpoints",
    )
    args = parser.parse_args()

    # Extract links from the URL
    with open(args.file, "r") as f:
        content = f.read()

    links = parse_links(content, regex_str, mode=0, no_dup=1)
    # Print results
    for link in links:
        print(link)
