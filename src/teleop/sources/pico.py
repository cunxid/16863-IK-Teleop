"""PICO VR source: controller poses in, target poses out (lesson §8).

How the bridge works, end to end:

    Pico headset browser ──(1) GET /──▶  this server  (serves webxr/index.html)
    Pico headset browser ──(2) wss /ws─▶ this server  (~72 pose messages/s)
    main loop ──(3) get_targets()──▶ frame conversion + clutch ──▶ TargetFrame

The server runs in a background thread with its own asyncio event loop; the
sim loop just reads the *latest* message under a lock (old messages are
worthless — we always want the newest controller pose, so no queue).

WebXR only runs over HTTPS, so on first start we generate a self-signed TLS
certificate into ``.certs/`` — the headset browser shows a warning once and
you tap "continue". Same certificate also secures the WebSocket (wss://).

Safety behaviors (relied on by run_real.py — keep them):
* watchdog — if no message arrived for STALE_AFTER seconds, get_targets()
  returns None (targets freeze) and both clutches disengage;
* deadman  — the clutch only moves a target while that grip is held.

Recentering (the A/X button): WebXR's "forward" is wherever you looked when
the VR session began, so if you are not facing the robot's way, your motions
come out rotated. Pressing A (or X) while facing the robot's forward
direction recomputes a yaw correction from the headset pose
(``mapping.recenter_correction``); it is applied to every controller sample
before the XR→MJ conversion. Recentering also disengages both clutches, so
nothing jumps — re-grab and continue.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

import numpy as np

from teleop import REPO_ROOT, quat
from teleop.mapping import Clutch, Pose, recenter_correction, xr_to_mj_pose
from teleop.sources import TargetFrame, TargetSource

STALE_AFTER = 0.5  # [s] without data ⇒ hold position (watchdog)
PAGE = Path(__file__).parent / "webxr" / "index.html"
CERT_DIR = REPO_ROOT / ".certs"


@dataclass
class HandSample:
    """One controller, one instant — already converted to the robot frame."""

    pose: Pose
    grip: bool
    trigger: float


@dataclass
class Snapshot:
    """The newest message from the headset (either hand may be untracked)."""

    received_at: float  # time.monotonic() on arrival
    left: HandSample | None
    right: HandSample | None


class PicoSource(TargetSource):
    def __init__(self, initial: TargetFrame, scale: float = 1.0, port: int = 8443):
        assert initial.left is not None and initial.right is not None
        self.port = port
        self.clutches = {"left": Clutch(initial.left, scale), "right": Clutch(initial.right, scale)}
        self._lock = threading.Lock()
        self._latest: Snapshot | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._ready = threading.Event()  # set once the server is listening (or failed)
        self._startup_error: BaseException | None = None
        self._active_client = None  # the one headset allowed to drive (first wins)
        self._correction = np.array([1.0, 0.0, 0.0, 0.0])  # XR yaw fix (identity = none)
        self._recenter_held = False  # for edge-detecting the A/X button

    # ------------------------------------------------------- main-loop side --

    def start(self) -> None:
        """Start the server thread; returns only once it is actually listening."""
        self._thread = threading.Thread(target=self._run_server, name="pico-server", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=15.0)
        if self._startup_error is not None or not self._ready.is_set():
            raise RuntimeError(
                f"Pico bridge failed to start on port {self.port} — is another "
                f"test-in-sim/test-in-real still running? Try --port <number>."
            ) from self._startup_error
        print(
            f"\nPico bridge ready. ON THE HEADSET (same Wi-Fi):\n"
            f"  1. open the browser at  https://{lan_ip()}:{self.port}\n"
            f"  2. accept the certificate warning (self-signed, see lesson §8)\n"
            f"  3. tap 'Enter VR', then HOLD a grip button to move that hand\n"
            f"  4. feels rotated? face the robot's forward direction and press\n"
            f"     A (or X) to re-align 'forward' with your gaze\n"
        )

    def get_targets(self, t: float) -> TargetFrame | None:
        with self._lock:
            snap = self._latest
        if snap is None or time.monotonic() - snap.received_at > STALE_AFTER:
            for clutch in self.clutches.values():  # force re-anchor on next grab
                clutch.update(grip_held=False, controller=None)
            return None  # watchdog: no fresh data ⇒ leave targets untouched

        targets = {}
        for name, hand in (("left", snap.left), ("right", snap.right)):
            clutch = self.clutches[name]
            if hand is None:  # controller not tracked this instant
                targets[name] = clutch.update(grip_held=False, controller=None)
            else:
                targets[name] = clutch.update(grip_held=hand.grip, controller=hand.pose)
        return TargetFrame(left=targets["left"], right=targets["right"])

    def stop(self) -> None:
        if self._thread is None:
            return
        self._ready.wait(timeout=3.0)  # don't lose a stop() that races startup
        if self._loop is not None and self._shutdown is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._shutdown.set)
            except RuntimeError:
                pass  # the loop died in the meantime; nothing left to stop
        self._thread.join(timeout=3.0)

    # ---------------------------------------------------------- server side --

    def _run_server(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as err:
            self._startup_error = err
            if self._ready.is_set():  # crashed after a good start: make it visible
                print(f"pico: server crashed: {err!r}")
        finally:
            self._ready.set()  # unblock start()/stop() even when startup failed

    async def _serve(self) -> None:
        # Import here so `uv run test-in-sim` (trajectory/mouse) never needs these.
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        cert, key = ensure_certificate(CERT_DIR)
        tls.load_cert_chain(cert, key)
        async with serve(self._on_websocket, "0.0.0.0", self.port, ssl=tls, process_request=self._on_http):
            self._ready.set()  # the port is bound; start() may now print the banner
            await self._shutdown.wait()

    def _on_http(self, connection, request):
        """Plain HTTPS requests get the WebXR page; /ws continues as WebSocket."""
        if request.path == "/ws":
            return None  # None ⇒ proceed with the WebSocket handshake
        response = connection.respond(HTTPStatus.OK, PAGE.read_text(encoding="utf-8"))
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response

    async def _on_websocket(self, websocket) -> None:
        if self._active_client is not None:  # one headset drives; reject the rest
            print("pico: rejected a second client (one headset at a time)")
            await websocket.close(1008, "another headset is already connected")
            return
        self._active_client = websocket
        print("pico: headset connected")
        try:
            async for message in websocket:
                try:
                    self.ingest(json.loads(message))
                except (ValueError, KeyError, TypeError, AttributeError) as err:
                    print(f"pico: ignoring malformed message ({err!r})")
        finally:
            self._active_client = None
            print("pico: headset disconnected")

    def ingest(self, msg: dict) -> None:
        """Accept one message (see webxr/index.html for the JSON format).

        Called by the WebSocket handler above — and directly by synthetic
        drivers (tests and the video renderer script a fake headset through
        here).
        """
        self._maybe_recenter(msg)
        snap = Snapshot(
            received_at=time.monotonic(),
            left=parse_hand(msg.get("left"), self._correction),
            right=parse_hand(msg.get("right"), self._correction),
        )
        with self._lock:
            self._latest = snap

    def _maybe_recenter(self, msg: dict) -> None:
        """A/X pressed (rising edge) ⇒ re-zero "forward" to the current gaze."""
        pressed = bool(msg.get("recenter")) and msg.get("head") is not None
        if pressed and not self._recenter_held:
            x, y, z, w = msg["head"]["quat_xyzw"]  # WebXR order → wxyz
            correction = recenter_correction(np.array([w, x, y, z]))
            if correction is None:
                print("pico: recenter ignored — look level (not up/down) and retry")
            else:
                self._correction = correction
                for clutch in self.clutches.values():  # re-anchor on next grab
                    clutch.update(grip_held=False, controller=None)
                print("pico: recentered — 'forward' is now the way you are facing")
        self._recenter_held = pressed


def parse_hand(hand: dict | None, correction: np.ndarray) -> HandSample | None:
    if hand is None:
        return None
    x, y, z, w = hand["quat_xyzw"]  # WebXR order; MuJoCo wants w first
    # Recenter first (a yaw rotation in XR space), then convert to robot axes.
    pos_xr = quat.rotate(correction, np.array(hand["pos"]))
    quat_xr = quat.mul(correction, np.array([w, x, y, z]))
    pose = xr_to_mj_pose(pos_xr, quat_xr)
    return HandSample(pose=pose, grip=bool(hand["grip"]), trigger=float(hand["trigger"]))


def lan_ip() -> str:
    """This machine's LAN address (the one the headset should dial)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; just picks a route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def covers_current_ip(cert_path: Path) -> bool:
    """True if the cached certificate still lists this machine's LAN IP.

    The IP changes when the laptop moves to another network; the certificate
    is throwaway, so on a mismatch we simply regenerate (the headset shows the
    accept-the-warning screen once more, and the URL keeps matching the cert).
    """
    import ipaddress

    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        return ipaddress.ip_address(lan_ip()) in sans.get_values_for_type(x509.IPAddress)
    except (ValueError, x509.ExtensionNotFound):
        return False  # unreadable or SAN-less cert: regenerate


def ensure_certificate(cert_dir: Path) -> tuple[Path, Path]:
    """Create (once) and return a self-signed TLS certificate + key.

    Self-signed = encrypted but not identity-verified; browsers warn, which is
    fine on a classroom LAN. WebXR simply refuses to run without HTTPS.
    """
    cert_path, key_path = cert_dir / "cert.pem", cert_dir / "key.pem"
    if cert_path.exists() and key_path.exists() and covers_current_ip(cert_path):
        return cert_path, key_path

    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ik-teleop")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)  # self-signed: we are our own authority
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365 * 5))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(lan_ip()))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_dir.mkdir(exist_ok=True)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"pico: generated self-signed certificate in {cert_dir}")
    return cert_path, key_path
