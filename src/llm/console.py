"""Make stdout able to print whatever the model generates.

A Windows console defaults to cp1252, which covers a few hundred characters. GPT-2's vocabulary
is byte-level and covers everything, so the moment sampling reaches into the tail -- a Cyrillic
letter, a CJK character, an emoji -- printing raises UnicodeEncodeError and the script dies
partway through its output.

This is a real failure mode for this project specifically, because printing generated text is
most of what the demos do.
"""

from __future__ import annotations

import sys


def use_utf8() -> None:
    """Switch stdout and stderr to UTF-8, replacing anything the terminal cannot render.

    `errors="replace"` rather than "strict": a character the font lacks should show as a
    replacement mark, not end the program.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Redirected to something that cannot be reconfigured; printing still works.
                pass
