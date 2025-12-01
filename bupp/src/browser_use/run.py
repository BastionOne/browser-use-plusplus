from typing import Callable, Awaitable
import asyncio
import time
import json

from browser_use.agent.views import (
    ActionResult,
    AgentHistoryList,
    AgentHistory,
    AgentStructuredOutput,
    BrowserStateHistory,
)
from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList, AgentStructuredOutput

async def bu_run(agent: Agent, step_fn: Callable[[], Awaitable[bool]]) -> AgentHistoryList[AgentStructuredOutput]:        
    """Execute the task with maximum number of steps"""
    loop = asyncio.get_event_loop()
    agent_run_error: str | None = None  # Initialize error tracking variable
    agent._force_exit_telemetry_logged = False  # ADDED: Flag for custom telemetry on force exit

    # Set up the  signal handler with callbacks specific to this agent
    from browser_use.utils import SignalHandler

    # Define the custom exit callback function for second CTRL+C
    def on_force_exit_log_telemetry():
        agent._log_agent_event(max_steps=agent.max_steps, agent_run_error='SIGINT: Cancelled by user')
        # NEW: Call the flush method on the telemetry instance
        if hasattr(agent, 'telemetry') and agent.telemetry:
            agent.telemetry.flush()
        agent._force_exit_telemetry_logged = True  # Set the flag

    signal_handler = SignalHandler(
        loop=loop,
        pause_callback=agent.pause,
        resume_callback=agent.resume,
        custom_exit_callback=on_force_exit_log_telemetry,  # Pass the new telemetrycallback
        exit_on_second_int=True,
    )
    signal_handler.register()

    try:
        await agent._log_agent_run()

        agent.logger.debug(
            f'🔧 Agent setup: Agent Session ID {agent.session_id[-4:]}, Task ID {agent.task_id[-4:]}, Browser Session ID {agent.browser_session.id[-4:] if agent.browser_session else "None"} {"(connecting via CDP)" if (agent.browser_session and agent.browser_session.cdp_url) else "(launching local browser)"}'
        )

        # Initialize timing for session and task
        agent._session_start_time = time.time()
        agent._task_start_time = agent._session_start_time  # Initialize task start time

        # Only dispatch session events if this is the first run
        if not agent.state.session_initialized:
            agent.logger.debug('📡 Dispatching CreateAgentSessionEvent...')
            # Emit CreateAgentSessionEvent at the START of run()
            # agent.eventbus.dispatch(CreateAgentSessionEvent.from_agent(agent))

            agent.state.session_initialized = True

        agent.logger.debug('📡 Dispatching CreateAgentTaskEvent...')
        # Emit CreateAgentTaskEvent at the START of run()
        # agent.eventbus.dispatch(CreateAgentTaskEvent.from_agent(agent))

        # Log startup message on first step (only if we haven't already done steps)
        agent._log_first_step_startup()
        # Start browser session and attach watchdogs
        await agent.browser_session.start()

        # Normally there was no try catch here but the callback can raise an InterruptedError
        try:
            await agent._execute_initial_actions()
        except InterruptedError:
            pass
        except Exception as e:
            raise e

        agent.logger.info("Starting agent prerun ...")
        skip_rest = await agent.pre_run()
        if skip_rest:
            return

        agent.logger.debug(f'🔄 Starting main execution loop with max {agent.max_steps} steps...')
        while agent.curr_step <= agent.max_steps:
            agent._log(f"==================== Starting step {agent.curr_step} ====================")
            
            # Use the consolidated pause state management
            if agent.state.paused:
                agent.logger.debug(f'⏸️ Step {agent.curr_step}: Agent paused, waiting to resume...')
                await agent._external_pause_event.wait()
                signal_handler.reset()

            # Check if we should stop due to too many failures, if final_response_after_failure is True, we try one last time
            if (agent.state.consecutive_failures) >= agent.settings.max_failures + int(
                agent.settings.final_response_after_failure
            ):
                agent.logger.error(f'❌ Stopping due to {agent.settings.max_failures} consecutive failures')
                agent_run_error = f'Stopped due to {agent.settings.max_failures} consecutive failures'
                break

            # Check control flags before each step
            if agent.state.stopped:
                agent.logger.info('🛑 Agent stopped')
                agent_run_error = 'Agent stopped programmatically'
                break

            await step_fn()
        else:
            agent_run_error = 'Failed to complete task in maximum steps'

            agent.history.add_item(
                AgentHistory(
                    model_output=None,
                    result=[ActionResult(error=agent_run_error, include_in_memory=True)],
                    state=BrowserStateHistory(
                        url='',
                        title='',
                        tabs=[],
                        interacted_element=[],
                        screenshot_path=None,
                    ),
                    metadata=None,
                )
            )

            agent.logger.info(f'❌ {agent_run_error}')

        agent.history.usage = await agent.token_cost_service.get_usage_summary()

        # set the model output schema and call it on the fly
        if agent.history._output_model_schema is None and agent.output_model_schema is not None:
            agent.history._output_model_schema = agent.output_model_schema

        return agent.history

    except KeyboardInterrupt:
        # Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
        agent.logger.debug('Got KeyboardInterrupt during execution, returning current history')
        agent_run_error = 'KeyboardInterrupt'

        agent.history.usage = await agent.token_cost_service.get_usage_summary()
        return agent.history

    except Exception as e:
        agent.logger.error(f'Agent run failed with exception: {e}', exc_info=True)
        agent_run_error = str(e)
        raise e

    finally:
        # Log token usage summary
        await agent.token_cost_service.log_usage_summary()

        # Save snapshots
        if agent.agent_dir:
            if agent.save_snapshots:
                with open(agent.agent_dir / "snapshots.json", "w") as f:
                    serialized_snapshots = agent.agent_snapshots.model_dump()
                    json.dump(serialized_snapshots, f)
            
            if agent.pages.get_req_count() > 0:
                with open(agent.agent_dir / "pages.json", "w") as f:
                    serialized_pages = await agent.pages.to_json()
                    json.dump(serialized_pages, f)

        # Unregister signal handlers before cleanup
        signal_handler.unregister()

        if not agent._force_exit_telemetry_logged:  # MODIFIED: Check the flag
            try:
                agent._log_agent_event(max_steps=agent.max_steps, agent_run_error=agent_run_error)
            except Exception as log_e:  # Catch potential errors during logging itagent
                agent.logger.error(f'Failed to log telemetry event: {log_e}', exc_info=True)
        else:
            # ADDED: Info message when custom telemetry for SIGINT was already logged
            agent.logger.debug('Telemetry for force exit (SIGINT) was logged by custom exit callback.')

        # Log final messages to user based on outcome
        agent._log_final_outcome_messages()

        agent.logger.info("==================== [VIEW LOGS] ============================")
        agent.logger.info(f"View logs: python view_snapshot.py {agent.agent_dir}")

        # Stop the event bus gracefully, waiting for all events to be processed
        # Use longer timeout to avoid deadlocks in tests with multiple agents
        await agent.eventbus.stop(timeout=3.0)

        await agent.close()

