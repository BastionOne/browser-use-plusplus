from browser_use.tools.service import Tools

EXCLUDE_LIST = [
    "search",
    "extract",
    "screenshot",
    "write_file",
    "replace_file",
    "read_file",
    "evaluate"
]
def register_tools(tools: Tools):
    @tools.registry.action(
        'A fake tool for testing purposes',
        param_model=None,
    )
    async def fake_test_tool():
        return "This is a fake tool for testing"

if __name__ == "__main__":
    tools = Tools(exclude_actions=EXCLUDE_LIST)
    register_tools(tools)

    print(tools.registry.get_prompt_description()) 