import asyncio
import signal
import sys
from browser_use_plusplus.sites.base import BrowserContextManager

async def main():
    """Open a browser and keep it open until the user hits Ctrl+C"""
    print("Starting browser... Press Ctrl+C to exit.")
    
    # Set up signal handler for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\nShutdown signal received. Closing browser...")
        shutdown_event.set()
    
    # Register signal handlers
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    else:
        # Windows doesn't support add_signal_handler, use different approach
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    
    try:
        async with BrowserContextManager(
            headless=False,
            use_proxy=False,
            n=1
        ) as browserdata_list:
            print("Browser opened successfully!")
            print("Browser is running... Press Ctrl+C to close.")
            
            # Wait for shutdown signal
            try:
                await shutdown_event.wait()
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Closing browser...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
