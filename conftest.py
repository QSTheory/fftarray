# from https://docs.pytest.org/en/6.2.x/example/simple.html
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--include-slow", action="store_true", default=False, help="run slow tests",
    )
    parser.addoption(
        "--skip-pandoc", action="store_true", default=False, help="skip tests which require pandoc",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "pandoc: mark test which requires pandoc")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--include-slow"):
        skip_slow = pytest.mark.skip(reason="need --include-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if config.getoption("--skip-pandoc"):
        skip_pandoc: pytest.MarkDecorator = pytest.mark.skip(reason="need to omit --skip-pandoc option to run")
        for item in items:
            if "pandoc" in item.keywords:
                item.add_marker(skip_pandoc)