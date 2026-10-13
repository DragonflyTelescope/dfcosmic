"""Tests that the examples in the README, the docs and the docstring run as written."""

import doctest
import os
import re
import textwrap
from pathlib import Path

import pytest

import dfcosmic.core
from dfcosmic.utils import cpp_median_available

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs" / "source"

# Set in the CI job that builds the extension, so a silently missing build fails there.
_REQUIRE_CPP = os.environ.get("DFCOSMIC_REQUIRE_CPP", "").lower() in {
    "1",
    "true",
    "yes",
}


def _python_examples(path: Path) -> list[str]:
    """The Python code blocks of a Markdown or reStructuredText file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return re.findall(r"^```python\n(.*?)^```", text, flags=re.S | re.M)
    blocks = re.findall(
        r"^\.\. code(?:-block)?:: python\n\n((?:[ \t]+.*\n|\n)+)", text, flags=re.M
    )
    return [textwrap.dedent(block) for block in blocks]


# One test per code block. The docs are not part of the source distribution, so
# their examples are only collected in a checkout of the repository.
EXAMPLES = [
    pytest.param(code, id=f"{path.name}-{number}")
    for path in [README, *sorted(DOCS.glob("*.rst"))]
    for number, code in enumerate(_python_examples(path), start=1)
]


@pytest.mark.parametrize("code", EXAMPLES)
def test_example_runs_as_written(code):
    """Every example is self-contained and runs on a CPU."""
    # The example that calls the C++ median filter directly needs the extension
    if "median_filter_cpp_torch" in code and not (
        cpp_median_available() or _REQUIRE_CPP
    ):
        pytest.skip("C++ median filter extension is not built")
    exec(compile(code, "<example>", "exec"), {"__name__": "__main__"})


@pytest.mark.parametrize("path", [README, DOCS / "index.rst"], ids=lambda p: p.name)
def test_quick_start_is_among_the_examples(path):
    """Guards against the quick start dropping out of the test above unnoticed."""
    if not path.exists():
        pytest.skip(f"{path.name} is not part of this distribution")
    examples = _python_examples(path)
    assert examples, f"no Python example found in {path.name}"
    assert any("lacosmic(" in code for code in examples)


def test_docstring_example():
    """The Examples section of the lacosmic docstring."""
    results = doctest.testmod(dfcosmic.core)
    assert results.attempted > 0
    assert results.failed == 0
