import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension

class CMakeBuild(build_ext):
    """
    A custom build extension for running CMake in a subdirectory.
    """
    def run(self):
        # Make sure CMake is available
        try:
            subprocess.check_call(["cmake", "--version"])
        except Exception:
            raise RuntimeError("CMake must be installed to build the extensions")

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYBIND11_INCLUDE_DIR={os.path.abspath('pybind11/include')}"
        ]

        subprocess.check_call(["cmake", ext.cmake_source_dir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] , cwd=build_temp)

class CMakeExtension(Extension):
    def __init__(self, name, cmake_source_dir=""):
        super().__init__(name, sources=[])
        self.cmake_source_dir = cmake_source_dir

# Define the extension module
ext_modules = [
    CMakeExtension(
        "pydyn.neighbors",  # Python module name
        cmake_source_dir=os.path.abspath("pydyn/neighbors")  # CMakeLists.txt所在目录
    )
]

setup(
    name="pydyn",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
