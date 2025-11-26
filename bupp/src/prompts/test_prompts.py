import concurrent.futures
import threading

from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.messages import UserMessage
from browser_use.tokens.service import TokenCost

from common.constants import BROWSER_USE_MODEL, DISCOVERY_MODEL_CONFIG
from bupp.src.llm_models import LLMHub

from pathlib import Path

import asyncio

# PROMPT_PATH = Path(r"C:\Users\jpeng\Documents\projects\code\web-ui3\.min_agent\2025-11-16\27\llm\update_plan\1.txt")
PROMPT_PATH = Path(r"C:\Users\jpeng\Documents\projects\code\web-ui3\.min_agent\2025-11-16\9\llm\browser_use\1.txt")

def get_lmp(path: Path) -> bool:
    func, _ = path.parts[-2:]
    return func

def sync_llm_hub(function: str, num: int = 1):
    llm_hub = LLMHub(DISCOVERY_MODEL_CONFIG["model_config"])
    model = llm_hub.get(function)

    prompt = open(PROMPT_PATH, "r").read()
    
    # Use ThreadPoolExecutor for concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=num) as executor:
        # Submit all tasks
        futures = [executor.submit(model.invoke, prompt) for _ in range(num)]
        
        # Collect results as they complete
        responses = []
        for future in concurrent.futures.as_completed(futures):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                print(f"Request failed: {e}")
    
    # Print responses
    for i, response in enumerate(responses):
        print(f"Response {i + 1}:")
        print(response.content)
        print("-" * 50)
    
    # Print total cost for this function
    total_cost = model.get_cost()
    print(f"Total cost for {function}: ${total_cost:.4f}")
    

async def async_openai(num: int = 1):
    model = ChatOpenAI(model=BROWSER_USE_MODEL)
    token_cost = TokenCost(include_cost=True)
    await token_cost.initialize()

    prompt = open(PROMPT_PATH, "r").read()
    
    # Create tasks for concurrent execution
    tasks = []
    for i in range(num):
        task = model.ainvoke([UserMessage(content=prompt)])
        tasks.append(task)
    
    # Gather all responses
    responses = await asyncio.gather(*tasks)
    
    total_cost = 0.0

    # Print responses one by one
    for i, response in enumerate(responses):
        print(f"Response {i + 1}:")
        print(response.completion)
        print("-" * 50)

        if response.usage:
            cost = await token_cost.calculate_cost(BROWSER_USE_MODEL, response.usage)
            if cost:
                total_cost += cost.total_cost

    print(f"Estimated total cost: ${total_cost:.4f}")

async def main(num: int = 1):
    func = get_lmp(PROMPT_PATH)
    if func == "browser_use":
        await async_openai(num)
    else:
        sync_llm_hub(func, num)

if __name__ == "__main__":
    import sys

    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    asyncio.run(main(num_requests))