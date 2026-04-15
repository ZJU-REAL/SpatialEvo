# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import ast
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def get_version() -> str:
    with open(os.path.join("verl", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"__version__\W*=\W*\"([^\"]+)\""
        (version,) = re.findall(pattern, file_content)
        return version


def _get_section_text(pyproject_text: str, section_name: str) -> str:
    match = re.search(rf"^\[{re.escape(section_name)}\]\s*$", pyproject_text, re.MULTILINE)
    if match is None:
        return ""
    start = match.end()
    next_section = re.search(r"^\[.*\]\s*$", pyproject_text[start:], re.MULTILINE)
    end = start + next_section.start() if next_section else len(pyproject_text)
    return pyproject_text[start:end]


def _extract_array_literal(section_text: str, key: str) -> list[str]:
    match = re.search(rf"^{re.escape(key)}\s*=\s*\[", section_text, re.MULTILINE)
    if match is None:
        return []

    array_start = match.end() - 1
    depth = 0
    array_end = None
    for idx, ch in enumerate(section_text[array_start:], start=array_start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                array_end = idx + 1
                break

    if array_end is None:
        raise ValueError(f"Failed to parse `{key}` from pyproject.toml")

    return list(ast.literal_eval(section_text[array_start:array_end]))


def get_requires() -> list[str]:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_section = _get_section_text(pyproject_text, "project")
    return _extract_array_literal(project_section, "dependencies")


def get_optional_requires() -> dict[str, list[str]]:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extras_section = _get_section_text(pyproject_text, "project.optional-dependencies")
    return {"dev": _extract_array_literal(extras_section, "dev")}


def main():
    setup(
        name="verl",
        version=get_version(),
        description="An Efficient, Scalable, Multi-Modality RL Training Framework based on veRL",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author="verl",
        author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk, hiyouga@buaa.edu.cn",
        license="Apache 2.0 License",
        url="https://github.com/volcengine/verl",
        package_dir={"": "."},
        packages=find_packages(where="."),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        extras_require=get_optional_requires(),
    )


if __name__ == "__main__":
    main()
