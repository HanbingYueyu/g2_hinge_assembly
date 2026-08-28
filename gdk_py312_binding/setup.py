"""Safe CPython 3.12 build recipe for the official GDK Python binding.

The official GDK source tree supplies the C++ binding sources, headers and
shared libraries. This recipe intentionally installs only into the active
Python environment; unlike the upstream setup.py, it never removes anything
from the official Python 3.10 installation.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.command.build_py import build_py


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ["G2_GDK_RUNTIME"]).expanduser().resolve()
GDK_ROOT = RUNTIME / "gdk"
BINDING_SOURCE = GDK_ROOT / "build_dep" / "python" / "pybind"

architecture = platform.machine()
if architecture not in {"x86_64", "aarch64"}:
    raise RuntimeError(f"unsupported architecture: {architecture}")
BUILD_DEP_ROOT = GDK_ROOT / "build_dep" / "cpp" / architecture

for required in (
    BINDING_SOURCE / "cpp" / "bind_gdk.cpp",
    BUILD_DEP_ROOT / "include" / "gdk" / "gdk.h",
    BUILD_DEP_ROOT / "lib" / "libgdk_adapter.so",
):
    if not required.is_file():
        raise RuntimeError(f"GDK build input is missing: {required}")


class CopyOfficialDdsHelper(build_py):
    """Include the official pure-Python DDS helper in this venv-only package."""

    def run(self) -> None:
        super().run()
        destination = Path(self.build_lib) / "agibot_gdk" / "dds.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BINDING_SOURCE / "python" / "dds.py", destination)


sources = sorted(str(path) for path in (BINDING_SOURCE / "cpp").glob("*.cpp"))
extension = Pybind11Extension(
    "agibot_gdk.agibot_gdk",
    sources,
    include_dirs=[
        str(BUILD_DEP_ROOT / "include"),
        str(BUILD_DEP_ROOT / "include" / "common"),
        str(BUILD_DEP_ROOT / "include" / "gdk"),
        str(BINDING_SOURCE),
        pybind11.get_include(),
    ],
    libraries=["gdk_adapter"],
    library_dirs=[
        str(GDK_ROOT / "lib"),
        str(BUILD_DEP_ROOT / "lib"),
        str(BUILD_DEP_ROOT / "lib" / "dds"),
    ],
    runtime_library_dirs=[
        "$ORIGIN",
        str(GDK_ROOT / "lib"),
        str(BUILD_DEP_ROOT / "lib"),
        str(BUILD_DEP_ROOT / "lib" / "dds"),
    ],
    extra_link_args=["-Wl,-rpath,$ORIGIN", "-Wl,--disable-new-dtags"],
    cxx_std=20,
)

setup(
    name="agibot_gdk_py312_local",
    version="4.1.5",
    description="Project-local CPython 3.12 binding for an installed AgiBot GDK runtime",
    packages=["agibot_gdk"],
    ext_modules=[extension],
    cmdclass={"build_ext": build_ext, "build_py": CopyOfficialDdsHelper},
    python_requires=">=3.12,<3.13",
    zip_safe=False,
)
