import subprocess
import sys


def test_basic_usage_example_runs() -> None:
    result = subprocess.run(
        [sys.executable, "examples/basic_usage.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
