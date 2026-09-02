"""Command dispatcher for ``python -m eval.analysis``."""

import argparse
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m eval.analysis",
        description="Dynamic-k artifact analysis commands.",
    )
    parser.add_argument("command", nargs="?", choices=("population", "radius"))
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = arguments.pop(0)
    if command == "population":
        from .population import main as population_main

        return population_main(arguments)
    if command == "radius":
        from .radius import main as radius_main

        return radius_main(arguments)
    parser.error("unknown command {!r}".format(command))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
