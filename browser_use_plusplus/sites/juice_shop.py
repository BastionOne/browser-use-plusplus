import asyncio

from browser_use_plusplus.sites.base import start_discovery_agent

from browser_use_plusplus.eval.client import PagedDiscoveryEvalClient
from browser_use_plusplus.eval.datasets.discovery.juiceshop import JUICE_SHOP_ALL as JUICE_SHOP_ALL_DISCOVERY
from browser_use_plusplus.eval.datasets.discovery.juiceshop_exploit import JUICE_SHOP_VULNERABILITIES as JUICE_SHOP_VULNERABILITIES_EXPLOIT

async def main():
    START_URLS = [
        "http://147.79.78.153:3000/#/login",
        "http://147.79.78.153:3000/#/contact",
        "http://147.79.78.153:3000/#/search"
    ]
    SCOPES = [
        "http://147.79.78.153:3000/rest/",
        "http://147.79.78.153:3000/api/",
    ]
    TEST_PATHS = [
        "/login"
    ]
    JUICE_SHOP_BASE_URL = "http://147.79.78.153:3000"
    JUICE_SHOP_ALL = {**JUICE_SHOP_ALL_DISCOVERY, **JUICE_SHOP_VULNERABILITIES_EXPLOIT}
    JUICE_SHOP_SUBSET = {p: JUICE_SHOP_ALL.get(p, []) for p in TEST_PATHS if p}

    challenge_client=PagedDiscoveryEvalClient(
        challenges=JUICE_SHOP_SUBSET,
        base_url=JUICE_SHOP_BASE_URL,
    )

    await start_discovery_agent(START_URLS, SCOPES, challenge_client)

if __name__ == "__main__":
    asyncio.run(main())
