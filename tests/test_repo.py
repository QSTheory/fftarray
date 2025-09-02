import pytest
import subprocess

@pytest.mark.pandoc
def test_readme_consistency():
    """Ensure README.rst matches converted README.md content"""
    # Convert Markdown to RST using pandoc
    result = subprocess.run(
        ["pandoc", "-f", "markdown+lists_without_preceding_blankline", "-t", "rst", "README.md", "-o", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    # Read both versions
    with open("README.rst", "r") as f:
        assert result.stdout == f.read(), "README.rst is out of sync with README.md"
