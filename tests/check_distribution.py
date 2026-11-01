"""Verify the licenses and S2MPJ attribution in a checkout and built ZIP, wheel or sdist."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LICENSE_SHA256 = "a8636fc42ac474fc85fbf451c6a0316f6cbd9efa9031d549797dec6b43e9e5b4"
ADAPTER_LICENSE_SHA256 = "f74fec06aef600da2eb63c20f028f198d5e2882405665e2ae1f0f81400666a7c"
REQUIRED = ("LICENCE.txt", "THIRD_PARTY_NOTICES.md", "README.md")
ADAPTER_LICENSE = "LICENSE"


def check_archive(path):
    names = (*REQUIRED, ADAPTER_LICENSE)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()
                        if PurePosixPath(name).name in names}
    else:
        with tarfile.open(path) as archive:
            contents = {member.name: archive.extractfile(member).read()
                        for member in archive.getmembers()
                        if member.isfile() and PurePosixPath(member.name).name in names}
    candidates = [name for name in contents if PurePosixPath(name).name == "LICENCE.txt"
                  and contents[name] == (ROOT / "LICENCE.txt").read_bytes()]
    assert candidates, f"{path}: missing exact upstream S2MPJ license"
    for candidate in candidates:
        parent = PurePosixPath(candidate).parent
        for name in REQUIRED:
            member = str(parent / name)
            assert contents.get(member) == (ROOT / name).read_bytes(), (
                f"{path}: missing or changed {member}"
            )
        # The core selects bundled files by extension or by name, so a core that
        # does not name this extensionless file yet leaves it out of its artifacts.
        member = str(parent / ADAPTER_LICENSE)
        if member in contents or parent.parts[-2:] != ("problem_libs", "s2mpj"):
            assert contents.get(member) == (ROOT / ADAPTER_LICENSE).read_bytes(), (
                f"{path}: missing or changed {member}"
            )
    print(f"{path}: exact S2MPJ license, notice and README verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256((ROOT / "LICENCE.txt").read_bytes()).hexdigest()
    assert digest == LICENSE_SHA256, "Upstream license bytes changed"
    adapter = hashlib.sha256((ROOT / ADAPTER_LICENSE).read_bytes()).hexdigest()
    assert adapter == ADAPTER_LICENSE_SHA256, "Adapter license bytes changed"
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for marker in ("GrattonToint/S2MPJ", "BSD-3-Clause", "fea6a70048eaad28b13a08703ddbfdbf65cd9c30",
                   "10.1080/10556788.2025.2490640", LICENSE_SHA256, "`LICENSE`"):
        assert marker in notice, f"Missing attribution: {marker}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[BSD 3-Clause License](LICENSE)" in readme, "README does not name the adapter license"
    print(f"Upstream license SHA-256: {digest}")
    for archive in args.archives:
        check_archive(archive)


if __name__ == "__main__":
    main()
