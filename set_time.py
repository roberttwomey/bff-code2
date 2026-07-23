#!/usr/bin/env python3
"""Set this machine's clock from a peer on the LAN - in practice, the Mac.

The Jetsons have no RTC battery and no reachable NTP server on the robot's
network, so they boot without a clock. chat-manager already corrects that from
the Go2's own DDS stamp, which needs no laptop and no internet and is the right
default. But the Go2 is only as right as the last time *it* was set: snapper and
its robot have both been observed a little over two days behind, which is late
enough to look plausible and so passes every "is the clock broken" check while
still misdating every session folder. Worse, it moves - a session recorded at
20:34 was followed by one named 19:47, so directories no longer sort in the
order they happened, and `ls -t` silently points at the wrong session.

This is the other end of that problem: a machine that does know the time,
because a human looks at it. Run the server on the Mac, point the Jetson at it
at startup, and the Jetson's clock becomes as good as the Mac's.

    # on the Mac (leave it running, or wrap it in a LaunchAgent)
    python3 set_time.py --serve

    # on the Jetson, at startup
    python3 set_time.py --host Roberts-MacBook-Air.local

Also accepts a time pushed in from elsewhere, for the case where the Mac can
reach the Jetson but not the other way round:

    ssh cohab@snapper.local "python3 ~/code/bff-code2/set_time.py --epoch $(date -u +%s)"

Falls back the same way chat-manager does: it prefers to set the system clock
via passwordless sudo, because that fixes file mtimes and every other program on
the box, and otherwise reports an offset the caller can apply in-process.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

DEFAULT_PORT = 37020

# Below this, stepping the clock costs more than the error does - and on a
# machine whose sessions are named to the second, a sub-second correction only
# risks two folders colliding.
DEFAULT_THRESHOLD = 2.0


def serve(port: int = DEFAULT_PORT, bind: str = "0.0.0.0") -> None:
    """Answer 'what time is it' on a TCP port, forever.

    One line of text per connection, then close. Deliberately not NTP: the whole
    point is that this needs no daemon, no config and no privileges on the
    machine that happens to know the time."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind, port))
    server.listen(8)
    print(f"[set_time] Serving this machine's clock on {bind}:{port} "
          f"(now {datetime.now():%Y-%m-%d %H:%M:%S}). Ctrl-C to stop.", file=sys.stderr)
    try:
        while True:
            conn, addr = server.accept()
            try:
                # Read the clock as late as possible - after the connection is
                # established, so the handshake is not counted as drift.
                conn.sendall(f"{time.time():.6f}\n".encode())
                print(f"[set_time] Told {addr[0]} the time.", file=sys.stderr)
            except Exception as e:
                print(f"[set_time] Failed to answer {addr[0]}: {e}", file=sys.stderr)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[set_time] Stopped.", file=sys.stderr)
    finally:
        server.close()


def query_peer(host: str, port: int = DEFAULT_PORT, timeout: float = 3.0) -> float | None:
    """Ask a peer for its clock. Returns epoch seconds corrected for half the
    round trip, or None if the peer is unreachable or talking nonsense.

    Half the round trip is the honest correction on a LAN: the reply spent
    roughly the same time in flight as the request, and both are well under the
    second that anything here is named by."""
    try:
        started = time.time()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            chunks = []
            while b"\n" not in b"".join(chunks):
                chunk = sock.recv(64)
                if not chunk:
                    break
                chunks.append(chunk)
        elapsed = time.time() - started
        peer_time = float(b"".join(chunks).decode().strip())
    except (OSError, ValueError) as e:
        print(f"[set_time] No time from {host}:{port} ({e}).", file=sys.stderr)
        return None

    # A peer that also booted without a clock is no help. 2025-01-01.
    if peer_time < 1_735_689_600:
        print(f"[set_time] {host} reports {datetime.fromtimestamp(peer_time):%Y-%m-%d %H:%M}, "
              f"which is not a plausible now - ignoring it.", file=sys.stderr)
        return None

    return peer_time + elapsed / 2.0


def apply_epoch(epoch: float) -> tuple[bool, float]:
    """Set the system clock. Returns (set_it, offset_seconds).

    offset_seconds is what the caller should add to its own clock when the
    system clock could not be set, so session names come out right even though
    file mtimes will not."""
    offset = epoch - time.time()

    # -n means never prompt: with no sudoers drop-in this fails at once rather
    # than blocking startup on a password nobody is there to type.
    result = subprocess.run(
        ["sudo", "-n", "date", "-u", "-s", f"@{epoch:.0f}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        return True, 0.0
    print(f"[set_time] Cannot set the system clock ({result.stderr.strip() or 'sudo denied'}); "
          f"reporting a {offset:+.1f}s offset instead.", file=sys.stderr)
    return False, offset


def sync_clock_from_peer(
    host: str | None = None,
    port: int = DEFAULT_PORT,
    threshold: float = DEFAULT_THRESHOLD,
    timeout: float = 3.0,
    dry_run: bool = False,
) -> float:
    """Correct this machine's clock from a peer. Returns the offset still to be
    applied in-process: 0.0 when the clock was set (or was already right), and
    non-zero when the peer was reachable but sudo was not.

    Unlike the robot-clock path this corrects in *both* directions. That one
    only ever moves forwards, because a machine that booted without a clock is
    behind and a robot claiming otherwise is wrong. Here the peer is a laptop a
    human reads the time on, so it is simply the better clock, and a Jetson that
    has run ahead should come back."""
    host = host or os.getenv("BFF_TIME_HOST")
    if not host:
        return 0.0

    peer_time = query_peer(host, port, timeout)
    if peer_time is None:
        return 0.0

    delta = peer_time - time.time()
    if abs(delta) < threshold:
        print(f"[set_time] Clock already agrees with {host} to within "
              f"{abs(delta):.2f}s - leaving it alone.", file=sys.stderr)
        return 0.0

    print(f"[set_time] Local clock reads {datetime.now():%Y-%m-%d %H:%M:%S}, "
          f"{host} says {datetime.fromtimestamp(peer_time):%Y-%m-%d %H:%M:%S} "
          f"({delta / 3600:+.2f}h).", file=sys.stderr)

    if dry_run:
        print("[set_time] --check only, not touching the clock.", file=sys.stderr)
        return 0.0

    was_set, offset = apply_epoch(peer_time)
    if was_set:
        print(f"[set_time] System clock set from {host}: "
              f"{datetime.now():%Y-%m-%d %H:%M:%S}", file=sys.stderr)
    return offset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronise this machine's clock with a peer on the LAN.")
    parser.add_argument("--serve", action="store_true",
                        help="run the time server (do this on the machine that knows the time)")
    parser.add_argument("--host", default=os.getenv("BFF_TIME_HOST"),
                        help="peer to ask for the time (default: $BFF_TIME_HOST)")
    parser.add_argument("--port", type=int, default=int(os.getenv("BFF_TIME_PORT", DEFAULT_PORT)),
                        help=f"port to serve or query (default: {DEFAULT_PORT})")
    parser.add_argument("--bind", default="0.0.0.0", help="address to serve on (default: all)")
    parser.add_argument("--epoch", type=float,
                        help="set the clock to this epoch instead of asking a peer")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"only step the clock past this many seconds of error "
                             f"(default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="seconds to wait for the peer (default: 3.0)")
    parser.add_argument("--check", action="store_true",
                        help="report the difference without changing anything")
    args = parser.parse_args()

    if args.serve:
        serve(args.port, args.bind)
        return 0

    if args.epoch is not None:
        delta = args.epoch - time.time()
        print(f"[set_time] Local clock reads {datetime.now():%Y-%m-%d %H:%M:%S}, "
              f"pushed time is {datetime.fromtimestamp(args.epoch):%Y-%m-%d %H:%M:%S} "
              f"({delta / 3600:+.2f}h).", file=sys.stderr)
        if args.check:
            return 0
        if abs(delta) < args.threshold:
            print("[set_time] Close enough - leaving the clock alone.", file=sys.stderr)
            return 0
        was_set, _ = apply_epoch(args.epoch)
        if was_set:
            print(f"[set_time] System clock set: {datetime.now():%Y-%m-%d %H:%M:%S}", file=sys.stderr)
        return 0 if was_set else 1

    if not args.host:
        parser.error("need --serve, --host/BFF_TIME_HOST, or --epoch")

    offset = sync_clock_from_peer(
        args.host, args.port, args.threshold, args.timeout, dry_run=args.check)
    # Non-zero offset means the peer answered but the clock could not be set,
    # which is a partial success worth signalling to a shell caller.
    return 0 if offset == 0.0 else 2


if __name__ == "__main__":
    sys.exit(main())
