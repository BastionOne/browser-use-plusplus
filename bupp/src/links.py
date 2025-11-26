# LinkFinder
# By Gerben_Javado

# Fix webbrowser bug for MacOS
import os
from typing import Dict, Any, List

os.environ["BROWSER"] = "open"

import re
import sys
import jsbeautifier
from logging import getLogger

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
# Extensions blacklist to ignore
blacklisted_extensions = ['.js', '.css']

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

    # Filter out blacklisted extensions
    items = [item for item in items if not any(item["link"].endswith(ext) for ext in blacklisted_extensions)]

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