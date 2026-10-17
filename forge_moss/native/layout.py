# MOSS-Image support for Forge Neo
# The layout prompt MOSS-Image reads (OpenMOSS's user guide, "Layout Prompts"): the whole prompt is one JSON object,
# {"summary": {"meta", "canvas", "caption"}, "items": [{"label", "caption", "text", "bbox"}, ...],
#  "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}.
#
# sd-dynamic-prompts reads {x} as a one-option variant and strips every brace, which leaves
# "summary":"meta":[...],"caption":"...","items":["label":...,"bbox":[...],"label":...],"bbox_format":"yxyx".
# The braces are put back where the layout's shape says they go: around the whole prompt, around the value of
# "summary" (it ends where the next top-level key starts), and around each element of a list of key-value pairs (a new
# element starts when a key repeats). Only the braces are added, the rest of the text is kept as it is, and the
# result is used only when it parses as a layout.

import json
import re

TOP_KEYS = frozenset(("summary", "items", "bbox_coord_space", "bbox_format"))
_STRIPPED_START = re.compile(r'\s*"(summary|items)"\s*:')
_SCALAR_END = re.compile(r'[\s:,\[\]"{}]')


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """(kind, start, end) for each JSON token: "s" a string, ":" "," "[" "]" punctuation, "v" any other scalar."""
    tokens, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                raise ValueError("unterminated string")
            tokens.append(("s", i, j + 1))
            i = j + 1
        elif c in ":,[]":
            tokens.append((c, i, i + 1))
            i += 1
        elif c in "{}":
            raise ValueError("braces left: not a stripped layout")
        else:
            m = _SCALAR_END.search(text, i)
            j = m.start() if m else n
            tokens.append(("v", i, j))
            i = j
    return tokens


class _Rebuilder:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokens(text)
        self.inserts: list[tuple[int, str]] = []

    def kind(self, i: int) -> str | None:
        return self.tokens[i][0] if i < len(self.tokens) else None

    def is_key(self, i: int) -> bool:
        return self.kind(i) == "s" and self.kind(i + 1) == ":"

    def key(self, i: int) -> str:
        _, a, b = self.tokens[i]
        return json.loads(self.text[a:b])

    def value(self, i: int) -> int:
        """Parses the value at token i, adding the braces it lost; returns the index of the token after it."""
        if self.is_key(i):
            # a stripped object as a value (the summary's): it ends where a top-level key starts
            return self.members(i, stop=TOP_KEYS, split_on_repeat=False)
        kind = self.kind(i)
        if kind in ("s", "v"):
            return i + 1
        if kind == "[":
            return self.array(i)
        raise ValueError(f"unexpected token at {i}")

    def array(self, i: int) -> int:
        i += 1
        while self.kind(i) != "]":
            # a list of key-value pairs is a list of stripped objects, one per repeated key
            i = self.members(i, stop=frozenset(), split_on_repeat=True) if self.is_key(i) else self.value(i)
            if self.kind(i) == ",":
                i += 1
            elif self.kind(i) != "]":
                raise ValueError("unterminated list")
        return i + 1

    def members(self, i: int, stop: frozenset, split_on_repeat: bool) -> int:
        """The key-value pairs of one stripped object from token i, wrapped in braces; returns the index after it."""
        self.inserts.append((self.tokens[i][1], "{"))
        seen = set()
        while True:
            seen.add(self.key(i))
            i = self.value(i + 2)
            if self.kind(i) != "," or not self.is_key(i + 1):
                break
            nxt = self.key(i + 1)
            if nxt in stop or (split_on_repeat and nxt in seen):
                break
            i += 1
        self.inserts.append((self.tokens[i - 1][2], "}"))
        return i

    def rebuild(self) -> str:
        end = self.members(0, stop=frozenset(), split_on_repeat=False)
        if end != len(self.tokens):
            raise ValueError("text after the layout")
        out, last = [], 0
        # stable by position: at one spot an inner brace was recorded before the outer one
        for pos, brace in sorted(self.inserts, key=lambda item: item[0]):
            out.append(self.text[last:pos])
            out.append(brace)
            last = pos
        out.append(self.text[last:])
        return "".join(out)


def is_layout(value) -> bool:
    return isinstance(value, dict) and isinstance(value.get("summary"), dict) and isinstance(value.get("items"), list)


def restore_layout_json(text: str) -> str:
    """The prompt with a brace-stripped layout rebuilt; unchanged when it is not one or the result does not parse."""
    if not _STRIPPED_START.match(text):
        return text
    try:
        rebuilt = _Rebuilder(text).rebuild()
        if is_layout(json.loads(rebuilt)):
            return rebuilt
    except (ValueError, IndexError):
        pass
    return text
