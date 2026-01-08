# Re-export shared types from llm_lib
from llm_lib import (
    ResourceType,
    RequestPart,
    Resource,
    UserID,
    RequestResources,
    EXTRACT_REQUESTS_PROMPT,
    EXTRACT_REQUESTS_PROMPT_V2,
    extract_json,
    get_json_schema_prompt,
    parse_response,
)
