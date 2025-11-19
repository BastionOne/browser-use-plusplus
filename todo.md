==================== TODOS =====================
Bugs:
- should do periodic performance testing to ensure that slowdown issue from e6f36ec doesnt show up

Internal:
- Think about setting up evals
- Test Exploit Agents
- Integrating Authz/AuthN attacker:
- construct dataset for web agents

BackBurner:
- consolidate cost logging for LLMs
- HTTPRequest parsing different payload data
> handling more exotic datatypes?

SiteMap:
- user_ids and other canonical IDs
- auth tokens 

Features:
> also integrate user_id into

Refactor:
- Only single init of MITMProxy should happen per server instance, but it is currently initialized per Discovery Agent in the Discovery Agent Pool. Should just attach single instance to app.state
- Unify proxy.log with agent_log -> tricky thing is we cant correlate by context_var since MITMProxy http handler forks new thread to handle each req/res flow

Design:
- Workflow for saving/running deterministic test cases
- Plugin system?
> where to expose the APIs

Non-Dev:
- look at offensive AICon

** IMPORTANT **
> POC for SSRF
> POC for HTTP-Smuggling attacks

==================== Goal for Thurs Demo [16/10/25] ====================
> Get aikido data loaded in + configure headers for exploit agent (ExploitAgent Auth Headers) to use 
> Workflow:
[Deprioritizations]
-> break down deprioritizations into groups with separate reason for each
-> issue these as the as separate accept / deny agents
-> if accepted, apply deprioritization to the UI first then server
-> ** if accepted we want to immediately execute an action **
--> means we have to cache some action beforehand
[Detection]
-> then we do detection on 

==================== Goal for Thurs Demo [23/10/25] ====================
#1 Workflow demonstrating working flow 
- deprio
- inference 
-> *Should have a manually triggered workflow that builds:
> initial set of agent questions

- Separating questions from agents
-> Questions -> have some immediate effect on DetectionStrategy
-> Agents -> multi-turn resolution

> Model
> View -> same, as we can see from 

-> Standardized View:
-> How to setup pre-compute workflow?


==================== DEV NOTES ====================
2025/10/11:
- added tests that confirmed MITMProxy handles edge cases without crashing
> probably will still need more robust HTTP handling a la httptoolkit


==================== BUGS ====================
- need we should consolidate the logic for all code paths that start the agent in different ways, and move their respective initializations in this method
> snapshot
> plan
> task

- potentially does not detect page change in _did_page_change in Agent
- problem with Browser Use dom diffing wrt to modal elements being hidden in the background
> .min_agent\2025-11-15\19\llm\update_plan\1.txt


==================== 2025/11/17 ====================
- UPDATE_PROMPT:
> test procedural v. functional framing
- IDEAs:
- [CREATEPLAN/UPDATEPLAN] ask the agent to explicitly identify actions that will trigger a backend API call and actions that prepare a request / parameters for the backend API call but does not trigger one
* think we should frame it still as UI actions tho
* although maybe we should think about integrating actions somehow?

UI tests strategy:
- component first
> check that we can recover from ephemeral modal

BrowserUse Challenges Eval:
6:
> fails to dropdown properly
> how often does batch actions