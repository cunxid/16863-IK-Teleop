"""Integration test for the Pico bridge: a fake headset talks to PicoSource.

Runs entirely on localhost (no VR hardware): we open a real WSS connection,
send controller messages in the WebXR JSON format, and check that targets
move the way lesson §8 promises (XR "push forward" = robot +x, clutch
semantics, watchdog).
"""

import asyncio
import json
import ssl
import time

import numpy as np
import pytest

from teleop.mapping import Pose
from teleop.sources import TargetFrame
from teleop.sources.pico import PicoSource

PORT = 8766
IDENTITY_XYZW = [0.0, 0.0, 0.0, 1.0]


def hand(pos, grip=True):
    return {"pos": list(pos), "quat_xyzw": IDENTITY_XYZW, "grip": grip, "trigger": 0.0}


@pytest.fixture()
def source():
    initial = TargetFrame(
        left=Pose(np.array([0.3, 0.2, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        right=Pose(np.array([0.3, -0.2, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    )
    src = PicoSource(initial, scale=1.0, port=PORT)
    src.start()
    yield src
    src.stop()


async def wait_for(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not met in time")


def test_fake_headset_moves_left_target(source: PicoSource):
    async def scenario():
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        tls.check_hostname = False
        tls.verify_mode = ssl.CERT_NONE  # self-signed server, test only

        from websockets.asyncio.client import connect

        await wait_for(lambda: True, timeout=0.1)  # give the server thread a tick
        async with connect(f"wss://127.0.0.1:{PORT}/ws", ssl=tls, open_timeout=5) as ws:
            # 1. Grip pressed at the XR origin: clutch engages, nothing moves yet.
            await ws.send(json.dumps({"t": 0.0, "left": hand([0, 0, 0]), "right": None}))
            await wait_for(lambda: source.get_targets(0.0) is not None)
            frame = source.get_targets(0.0)
            assert np.allclose(frame.left.pos, [0.3, 0.2, 1.0], atol=1e-9)

            # 2. Push the controller "forward" 10 cm (XR forward = -z)
            #    ⇒ the robot-frame target moves +10 cm in x.
            await ws.send(json.dumps({"t": 0.1, "left": hand([0, 0, -0.1]), "right": None}))
            await wait_for(
                lambda: abs(source.get_targets(0.1).left.pos[0] - 0.4) < 1e-9, timeout=5.0
            )

            # 3. Right hand untracked the whole time ⇒ its target never moved.
            assert np.allclose(source.get_targets(0.2).right.pos, [0.3, -0.2, 1.0])

            # 4. Release the grip and move far away ⇒ target freezes (deadman).
            await ws.send(json.dumps({"t": 0.2, "left": hand([9, 9, 9], grip=False), "right": None}))
            await wait_for(
                lambda: abs(source.get_targets(0.3).left.pos[0] - 0.4) < 1e-9, timeout=5.0
            )

    asyncio.run(scenario())


def test_watchdog_goes_stale_without_messages(source: PicoSource):
    assert source.get_targets(0.0) is None  # nothing ever received ⇒ hold
