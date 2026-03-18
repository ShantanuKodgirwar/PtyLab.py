import argparse
import sys


def cmd_check_gpu():
    try:
        import cupy as cp

        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            print("cupy is installed but no CUDA devices found.")
            sys.exit(1)
        print(f"GPU available: {n} device(s)")
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode()
            mem_gb = props["totalGlobalMem"] / 1024**3
            print(f"  [{i}] {name}  ({mem_gb:.1f} GB)")
    except ImportError:
        print(
            "cupy is not installed. Install with:\n"
            "  pip install ptylab[cuda12]   # for CUDA 12.x\n"
            "  pip install ptylab[cuda13]   # for CUDA 13.x"
        )
        sys.exit(1)
    except Exception as e:
        print(f"GPU check failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="ptylab")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="System checks")
    check_sub = check.add_subparsers(dest="target")
    check_sub.add_parser("gpu", help="Check GPU / cupy availability")

    args = parser.parse_args()

    if args.command == "check" and args.target == "gpu":
        cmd_check_gpu()
    else:
        parser.print_help()
        sys.exit(1)
