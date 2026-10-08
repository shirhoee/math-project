import argparse
import sys

def run_demo(args: argparse.Namespace) -> None:
    # Need to be able to import make_demo or just run it via subprocess
    import subprocess
    cmd = [sys.executable, "v1/tools/make_demo.py"]
    if args.image:
        cmd.extend(["--image", args.image])
    subprocess.run(cmd)

def run_benchmark(args: argparse.Namespace) -> None:
    import subprocess
    cmd = [sys.executable, "v1/tools/benchmark.py"]
    subprocess.run(cmd)

def run_quality(args: argparse.Namespace) -> None:
    import subprocess
    cmd = [sys.executable, "v1/tools/quality_check.py"]
    subprocess.run(cmd)

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CLI for 3D View Synthesis tools")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Generate a demo GIF from an image")
    demo_parser.add_argument("--image", type=str, help="Path to input image")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run the performance benchmark suite")

    # Quality Check
    quality_parser = subparsers.add_parser("quality", help="Run the rendering quality checks")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "quality":
        run_quality(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
