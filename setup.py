from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class CUDAExtension(Extension):
    def __init__(self, name: str, sources: list[str], cuda_sources: list[str], **kwargs):
        super().__init__(name=name, sources=sources, **kwargs)
        self.cuda_sources = cuda_sources


def _find_nvcc() -> Path | None:
    if os.getenv("CUDA_HOME"):
        candidate = Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"
        if candidate.exists():
            return candidate
    nvcc = shutil.which("nvcc")
    return Path(nvcc) if nvcc else None


def _build_cuda_ext() -> list[CUDAExtension]:
    if os.getenv("CACHEPILOT_BUILD_CUDA", "0") != "1":
        return []

    try:
        import pybind11
    except ImportError:
        print("Skipping CachePilot CUDA extension build because pybind11 is unavailable.")
        return []

    nvcc = _find_nvcc()
    if nvcc is None:
        print("Skipping CachePilot CUDA extension build because nvcc was not found.")
        return []

    cuda_home = nvcc.parent.parent
    library_dir = cuda_home / "lib64"
    if not library_dir.exists():
        library_dir = cuda_home / "lib"

    return [
        CUDAExtension(
            name="cachepilot._cuda_kernels",
            sources=["src/cuda/pybind_module.cpp"],
            cuda_sources=[
                "src/cuda/native_info.cu",
                "src/cuda/kv_block_copy.cu",
                "src/cuda/kv_quant.cu",
            ],
            include_dirs=[
                pybind11.get_include(),
                sysconfig.get_path("include"),
                str(ROOT / "src" / "cuda"),
            ],
            libraries=["cudart"],
            library_dirs=[str(library_dir)],
            language="c++",
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--compiler-options", "-fPIC"],
            },
        )
    ]


class BuildCUDAExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CUDAExtension):
            super().build_extension(ext)
            return

        extra_compile_args = ext.extra_compile_args if isinstance(ext.extra_compile_args, dict) else {}
        cxx_args = extra_compile_args.get("cxx", [])
        nvcc_args = extra_compile_args.get("nvcc", [])

        objects = self.compiler.compile(
            ext.sources,
            output_dir=self.build_temp,
            include_dirs=ext.include_dirs,
            extra_postargs=cxx_args,
        )

        nvcc = _find_nvcc()
        if nvcc is None:
            raise RuntimeError("nvcc is required when CACHEPILOT_BUILD_CUDA=1.")

        for source in ext.cuda_sources:
            output = Path(self.build_temp) / f"{Path(source).stem}.cu.o"
            cmd = [str(nvcc), "-c", source, "-o", str(output), *nvcc_args]
            for include_dir in ext.include_dirs or []:
                cmd.extend(["-I", str(include_dir)])
            subprocess.check_call(cmd)
            objects.append(str(output))

        language = self.compiler.detect_language(ext.sources)
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=language,
        )


ext_modules = _build_cuda_ext()

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCUDAExt} if ext_modules else {},
)
