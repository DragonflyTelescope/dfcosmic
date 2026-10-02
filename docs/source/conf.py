# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import shutil
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path

project = "dfcosmic"
author = (
    "Carter Rhea, Pieter van Dokkum, Steven Janssens, Imad Pasha, Roberto Abraham, "
    "William P. Bowman, Deborah Lokhorst, Seery Chen"
)
copyright = f"2025, {author}"
# The docs are built with dfcosmic installed, so take the version from the package
try:
    release = _version("dfcosmic")
except PackageNotFoundError:
    release = "unknown"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx_copybutton",
    "nbsphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_nefertiti"
# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# AutoAPI settings
autoapi_dirs = ["../../src/dfcosmic"]  # Path to your Python package
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autodoc_mock_imports = ["torch"]
# Copy button settings (optional)
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# Exclude build directory and Jupyter backup files
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The example notebooks and the timing figures live in demos/ at the top of the
# repository. They are copied here at build time, so that there is only one copy to
# keep up to date, and are shown with their stored outputs rather than re-executed.
_demos = Path(__file__).resolve().parents[2] / "demos"
_docs_demos = Path(__file__).resolve().parent / "demos"
_docs_demos.mkdir(exist_ok=True)
for _pattern in ("*.ipynb", "comparison*.png"):
    for _file in _demos.glob(_pattern):
        shutil.copy2(_file, _docs_demos / _file.name)

nbsphinx_execute = "never"
