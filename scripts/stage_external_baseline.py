"""Stage verified official MR.Q/MRS.Q sources without importing or running them."""
import argparse
import json
from pathlib import Path

from cars.research.external_sources import stage_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("MRQ", "MRSQ"), required=True)
    parser.add_argument("--cache", type=Path, default=Path("runs/references/mrq_mrsq_audit_20261030"))
    parser.add_argument("--audit", type=Path, default=Path("reports/references/mrq_mrsq_source_audit_2026-10-30.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = stage_sources(args.algorithm, args.cache, args.audit, args.output)
    print(json.dumps({"algorithm": manifest["algorithm"], "revision": manifest["revision"],
        "output": str(args.output), "files": len(manifest["source_files"]),
        "runtime_status": manifest["runtime_status"]}, indent=2))


if __name__ == "__main__":
    main()
