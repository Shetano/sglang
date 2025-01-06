import asyncio
import threading

import uvloop

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
