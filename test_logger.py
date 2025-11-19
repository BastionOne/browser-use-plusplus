from nt import get_inheritable
from browser_use_plusplus.logger import get_or_init_log_factory


if __name__ == "__main__":
    logger = get_or_init_log_factory(".min_agent")
    agent_log, full_log = logger.get_discovery_agent_loggers(streaming=False)

    agent_log.info("WTF!")