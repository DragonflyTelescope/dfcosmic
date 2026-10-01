from importlib.metadata import PackageNotFoundError, version

from .core import lacosmic as lacosmic

try:
    __version__ = version("dfcosmic")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["lacosmic", "__version__"]
