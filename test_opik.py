import asyncio
from opik.integrations.langchain import opik_tracer

from browser_use.llm.openai.chat import ChatOpenAI

async def test_opik():
    client = ChatOpenAI(model="gpt-3.5-turbo")
    client = opik_tracer(client)
    await client.ainvoke("Hello, how are you?")

if __name__ == "__main__":
    asyncio.run(test_opik())

# @opik.track
# def retrieve_context(input_text):
#     # Your retrieval logic here, here we are just returning a
#     # hardcoded list of strings
#     context =[
#         "What specific information are you looking for?",
#         "How can I assist you with your interests today?",
#         "Are there any topics you'd like to explore?",
#     ]
#     return context

# @opik.track
# def generate_response(input_text, context):
#     full_prompt = (
#         f" If the user asks a non-specific question, use the context to provide a relevant response.\n"
#         f"Context: {', '.join(context)}\n"
#         f"User: {input_text}\n"
#         f"AI:"
#     )

#     response = client.ainvoke([UserMessage(content=full_prompt)])
#     return response.choices[0].message.content

# @opik.track(name="my_llm_application")
# def llm_chain(input_text):
#     context = retrieve_context(input_text)
#     response = generate_response(input_text, context)

#     return response

# # Use the LLM chain
# result = llm_chain("Hello, how are you?")
# print(result)