"""Make stdout able to print whatever the model generates.

A Windows console defaults to cp1252, which covers a few hundred characters. GPT-2's vocabulary
is byte-level and covers everything, so the moment sampling reaches into the tail -- a Cyrillic
letter, a CJK character, an emoji -- printing raises UnicodeEncodeError and the script dies
partway through its output.

This is a real failure mode for this project specifically, because printing generated text is
most of what the demos do.
"""

from __future__ import annotations

import re
import sys
from typing import TextIO


def use_utf8() -> None:
    """Switch stdin, stdout and stderr to UTF-8, replacing anything that cannot be decoded.

    `errors="replace"` rather than "strict": a character the font lacks should show as a
    replacement mark, not end the program.

    stdin matters for the chat scripts. Typed into a console, Windows hands Python Unicode
    directly; *piped* in, the bytes are decoded with the locale's codepage -- cp1252 here -- so
    a Burmese greeting arrived at the model as `\u00e1\u20ac\u2122...` and it answered the mojibake.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Redirected to something that cannot be reconfigured; printing still works.
                pass


#: Terminal colour codes, which make a live run readable and a saved log unreadable.
ANSI = re.compile(r"\033\[[0-9;]*m")


class Tee:
    """Write to the screen in colour and to a log without it, so a live, coloured run -- the
    knowledge check, the model's own answers -- leaves a plain record behind."""

    def __init__(self, log: TextIO) -> None:
        self.log = log

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        self.log.write(ANSI.sub("", text))

    def flush(self) -> None:
        sys.stdout.flush()
        self.log.flush()


if __name__ == "__main__":
    import io

    use_utf8()
    log = io.StringIO()
    out = Tee(log)
    out.write("\033[94mLLM ›\033[0m မင်္ဂလာပါ "
              "-- coloured here, and Burmese prints\n")
    print(f"the log holds: {log.getvalue()!r}")
