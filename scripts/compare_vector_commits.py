#!/usr/bin/env python3
import re
import sys


SPIKE_RE = re.compile(
    r"^.* (?P<pc>0x[0-9a-fA-F]+) "
    r"\((?P<inst>0x[0-9a-fA-F]+)\).* "
    r"(?P<reg>v\d+)\s+(?P<value>0x[0-9a-fA-F]+)"
)
VERILATOR_RE = re.compile(r"^pc 0x[0-9a-fA-F]+ inst 0x[0-9a-fA-F]+ (?P<reg>v\d+) 0x(?P<value>[0-9a-fA-FXx]+)$")


def split_bytes(value):
    value = value.removeprefix("0x")
    return [value[i : i + 2].lower() for i in range(0, len(value), 2)]


def parse_spike(path):
    commits = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            match = SPIKE_RE.match(raw.rstrip())
            if not match:
                continue
            commits.append(
                {
                    "line": line_no,
                    "pc": match.group("pc"),
                    "inst": match.group("inst"),
                    "reg": match.group("reg"),
                    "value": split_bytes(match.group("value")),
                    "raw": raw.rstrip(),
                }
            )
    return commits


def parse_verilator(path):
    writes = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            match = VERILATOR_RE.match(raw.rstrip())
            if not match:
                continue
            writes.append(
                {
                    "line": line_no,
                    "reg": match.group("reg"),
                    "value": split_bytes(match.group("value")),
                    "raw": raw.rstrip(),
                }
            )
    return writes


def apply_verilator_write(state, write):
    cur = state.setdefault(write["reg"], ["??"] * len(write["value"]))
    for i, byte in enumerate(write["value"]):
        if "x" not in byte:
            cur[i] = byte


def compatible(lhs, rhs):
    return len(lhs) == len(rhs) and all(a == b or a == "??" or b == "??" for a, b in zip(lhs, rhs))


def fmt(value):
    return "0x" + "".join(value)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: compare_vector_commits.py SPIKE_LOG VERILATOR_VREG_LOG")

    spike = parse_spike(sys.argv[1])
    verilator = parse_verilator(sys.argv[2])
    vstate = {}
    vi = 0

    print(f"spike vector commits: {len(spike)}")
    print(f"verilator vreg writes: {len(verilator)}")

    for si, scommit in enumerate(spike):
        matched = False
        while vi < len(verilator):
            write = verilator[vi]
            vi += 1
            apply_verilator_write(vstate, write)
            if write["reg"] != scommit["reg"]:
                continue
            if compatible(vstate[write["reg"]], scommit["value"]):
                matched = True
                break

            print(f"\nfirst vector divergence around spike vector commit {si}")
            print(f"spike log line : {scommit['line']}")
            print(f"spike pc/inst  : {scommit['pc']} {scommit['inst']}")
            print(f"register       : {scommit['reg']}")
            print(f"spike value    : {fmt(scommit['value'])}")
            print(f"verilator line : {write['line']}")
            print(f"verilator write: {write['raw']}")
            print(f"verilator arch : {fmt(vstate[write['reg']])}")
            return 1

        if not matched:
            print(f"\nmissing verilator write for spike vector commit {si}")
            print(scommit["raw"])
            return 1

    print(f"matched all spike vector commits; consumed {vi} verilator writes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
