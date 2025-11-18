from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

# TODO: create an ExtractInjectionInformation request that takes into account the webpage context
class ResourceType(BaseModel):
    name: str
    # TODO: still need to decide on a unique way of identifying requests
    requests: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    
    def to_prompt_str(self):
        return f"Resource Type: {self.name}\nDescription: {self.description}\n"
    
class RequestPart(str, Enum):
    URL = "URL"
    BODY = "BODY"
    HEADERS = "HEADERS"
    
class Resource(BaseModel):
    id: str
    type: ResourceType
    request_part: RequestPart
    selected_slice: Dict[RequestPart, str] = Field(default_factory=dict)
    
class UserID(BaseModel):
    id: str
    request_part: RequestPart
    selected_slice: Dict[RequestPart, str] = Field(default_factory=dict)

# TODO: switch this output to ResourceLocator
# TODO: roll user_ids into resource_ids
# Improving parameter extraction
# 1. add description to request
# 2. explicitly ask it to identify resource vs. non-resource parameters
# 3. split up resource identification in URL vs. Body
# 4. provide additional context from web page to help with identification
class RequestResources(BaseModel):
    description: str = Field(default="")
    resources: Optional[List[Resource]] = Field(default_factory=list)

EXTRACT_REQUESTS_PROMPT = """
{{request}}

You are given an HTTP request.
Your task is to extract resource identifiers from the request:

These request parameters represent a resource used in the application, that is not just a configuration parameter for the API call. In thinking of a request as an action,
a request can be seen to perform some action on a resource. It is important to note that some requests will not have resource identifiers
For example:
- A request to "/api/posts/456/comments" likely represents a comment resource related to post "456"
- A request to "/api/update/tweet" with JSON payload {"tweet": 1234, "message": "hello world", "picture": "https://pic.png"} has ONLY the valid resource tweet:1234
- A request to "/api/list/tweets" HAS no resource identifiers, as it is just an action to list of tweets

If you did not identify any resources, make sure to return [] and not null for the resources field 

{% if resource_types %}
Does the resource match any of the existing resource types? If so then select that type
{% endif %}

If this is a new resource, then create it

When returning the identified resource_ids, please provide the selected_slice parameter, which identifies the specific position
of the resource in the request. The way you should give this data is to:
1. Take the surrounding of the resource, but make sure to not step across request element boundaries ie. if param in URL do not include the headers in surround
2. Next identify where in the request the resource is located by the following steps:
Identify whether its in the URL, BODY, or HEADERS:
NOTE: SEPARATOR = "$%&" ie. /post/$%&user1234$%&/comments
    if URL:
        - return selected_slice as the URL where selected resource value is enclosed within the SEPARATOR
        ie. /post/$%&user1234$%&/comments
        ie. /tweets?id=1234
    elif BODY:
        - return selected_slice as the name of body field
    elif HEADERS:
        - return selected_slice as {}

Here are some examples:
1. GET http://localhost:8000/tweets/list/
{
    "resources": []
}

2. GET http://localhost:8000/messages/
{
    "resources": []
}

3. GET http://twitter.com/api/tweets/2/replies/
{
    "resources": [
        {
            "id": "2",
            "type": {
                "name": "tweet_reply",
                "description": "This represents replies to a tweet, determined by the endpoint '/api/tweets/{tweet_id}/replies/'.",
                "requests": []
            },
            "request_part": "URL",
            "selected_slice": {
                "URL": "/api/tweets/$%&2$%&/replies/"
            }
        }
    ]
}

4. POST http://twitter.com/reply/
{'csrfmiddlewaretoken': 'h0EclP8OYE80II6kTnXGYgTqiyFPhTBxl3FPTOat7YxJuc9tnjV4qV2g37zmFle1', 'tweet': 'tw121345', 'hashtags': '#twitter', 'media_id': '', 'reply_to': '11', 'visibility': 'public', 'location': '1', 'notes': 'test tweet'}
{
    "resources": [
        {
            "id": "tw121345",
            "type": {
                "name": "tweet",
                "description": "Represents a tweet being created",
                "requests": []
            },
            "request_part": "BODY",
            "selected_slice": {
                "BODY": "tweet"
            }
        }
    ]
}

First give an overall description of the request and what it is trying to accomplish
Use this section to think aloud on the purpose of the request, what parameters are being used,
and most crucially, which ones are the resource identifiers and which ones are simply regular parameters

To reiterate:
Your task is to extract resource identifiers from the request:

These request parameters represent a resource used in the application, that is not just a configuration parameter for the API call. In thinking of a request as an action,
a request can be seen to perform some action on a resource

When returning the identified resource_ids, please provide the selected_slice parameter, which identifies the specific position
of the resource in the request. The way you should give this data is to:
1. Take the surrounding of the resource, but make sure to not step across request element boundaries ie. if param in URL do not include the headers in surround
2. Next identify where in the request the resource is located by the following steps:
Identify whether its in the URL, BODY, or HEADERS:
NOTE: SEPARATOR = "$%&" ie. /post/$%&user1234$%&/comments
    if URL:
        - return selected_slice as the URL where selected resource value is enclosed within the SEPARATOR
        ie. /post/$%&user1234$%&/comments
        ie. /tweets?id=1234
    elif BODY:
        - return selected_slice as the name of body field
    elif HEADERS:
        - return selected_slice as {}

If you did not identify any resources, make sure to return [] and not null for the resources field
Now give your response
"""

EXTRACT_REQUESTS_PROMPT_V2 = """
{{request}}

You are given an HTTP request.
Your task is to extract resource identifiers from the request:

These request parameters represent a resource used in the application, that is not just a configuration parameter for the API call. In thinking of a request as an action,
a request can be seen to perform some action on a resource. It is important to note that some requests will not have resource identifiers
For example:
- A request to "/api/posts/456/comments" likely represents a comment resource related to post "456"
- A request to "/api/update/tweet" with JSON payload {"tweet": 1234, "message": "hello world", "picture": "https://pic.png"} has ONLY the valid resource tweet:1234
- A request to "/api/list/tweets" HAS no resource identifiers, as it is just an action to list of tweets

If you did not identify any resources, make sure to return [] and not null for the resources field 

{% if resource_types %}
Does the resource match any of the existing resource types? If so then select that type
{% endif %}

If this is a new resource, then create it

When returning the identified resource_ids, please identify where in the request the resource is located by the following steps:
Identify whether its in the URL, BODY, or HEADERS.

Here are some examples:
1. GET http://localhost:8000/tweets/list/
{
    "resources": []
}

2. GET http://localhost:8000/messages/
{
    "resources": []
}

3. GET http://twitter.com/api/tweets/2/replies/
{
    "resources": [
        {
            "id": "2",
            "type": {
                "name": "tweet_reply",
                "description": "This represents replies to a tweet, determined by the endpoint '/api/tweets/{tweet_id}/replies/'.",
                "requests": []
            },
            "request_part": "URL"
        }
    ]
}

4. POST http://twitter.com/reply/
{'csrfmiddlewaretoken': 'h0EclP8OYE80II6kTnXGYgTqiyFPhTBxl3FPTOat7YxJuc9tnjV4qV2g37zmFle1', 'tweet': 'tw121345', 'hashtags': '#twitter', 'media_id': '', 'reply_to': '11', 'visibility': 'public', 'location': '1', 'notes': 'test tweet'}
{
    "resources": [
        {
            "id": "tw121345",
            "type": {
                "name": "tweet",
                "description": "Represents a tweet being created",
                "requests": []
            },
            "request_part": "BODY"
        }
    ]
}

First give an overall description of the request and what it is trying to accomplish
Use this section to think aloud on the purpose of the request, what parameters are being used,
and most crucially, which ones are the resource identifiers and which ones are simply regular parameters

To reiterate:
Your task is to extract resource identifiers from the request:

These request parameters represent a resource used in the application, that is not just a configuration parameter for the API call. In thinking of a request as an action,
a request can be seen to perform some action on a resource

When returning the identified resource_ids, please identify where in the request the resource is located (URL, BODY, or HEADERS).

If you did not identify any resources, make sure to return [] and not null for the resources field
Now give your response
"""
