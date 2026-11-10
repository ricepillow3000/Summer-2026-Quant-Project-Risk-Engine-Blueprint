"""
flood a running app's websocket and make sure it hangs up on us.

    python -m tools.flood_check http://127.0.0.1:8501

both ways, each from its own made-up address (the X-Forwarded-For hop serve.py
reads): a busy visitor sends 100 tiny valid messages (heartbeats) and must stay
connected; a flood sends 4x MESSAGES_PER_MINUTE as fast as it can and must be
closed with 1013 ("try again later"). run it next to tools/load_check.mjs: the
flood must not stop a visitor from another address getting the headline.
exit code 1 if the visitor is cut off or the flood is not.
"""

import asyncio
import sys

import websockets
from streamlit.proto.BackMsg_pb2 import BackMsg

from serve import MESSAGES_PER_MINUTE

SENT = 4 * MESSAGES_PER_MINUTE         # far past the cap, still a second or two of traffic


async def flood(url: str, address: str = "198.51.100.23", count: int = SENT) -> tuple[int, int | None]:
    """-> (messages sent, the close code; None = still open after `count`)."""
    ws_url = url.replace("http", "ws", 1).rstrip("/") + "/_stcore/stream"
    beat = BackMsg(app_heartbeat=True).SerializeToString()
    sent = 0
    async with websockets.connect(ws_url, subprotocols=["streamlit"], origin=url.rstrip("/"),
                                  additional_headers={"X-Forwarded-For": address},
                                  max_size=None) as ws:
        try:
            for _ in range(count):
                await ws.send(beat)
                sent += 1
                await asyncio.sleep(0)
            await asyncio.wait_for(ws.wait_closed(), timeout=10 if count == SENT else 3)
        except websockets.ConnectionClosed:
            pass
        except TimeoutError:
            return sent, None
        return sent, ws.close_code


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m tools.flood_check <app url>")
        return 2
    sent, code = asyncio.run(flood(argv[0], "198.51.100.24", 100))
    print(f"visitor: {sent} messages sent, close code {code} (None = still connected)")
    if code is not None:
        print("FAIL: a busy visitor was cut off")
        return 1
    sent, code = asyncio.run(flood(argv[0]))
    print(f"flood: {sent} messages sent, closed with {code} (cap {MESSAGES_PER_MINUTE} a minute)")
    if code != 1013:
        print("FAIL: the flood was never cut off")
        return 1
    print("PASS: a busy visitor stays connected; a flood is hung up on at the cap")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
