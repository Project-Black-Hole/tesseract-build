#!/usr/bin/env python3
"""Builds one small Tesseract library from pinned sources.

    python build.py pin --tesseract <tag> --leptonica <tag> --tessdata <commit>
        Downloads the three sources, hashes them and writes sources.pin.json.
    python build.py build
        Checks every source against its SHA-256, builds Leptonica (static,
        with no picture-format library) and Tesseract (one shared library:
        the LSTM engine only, no training tools, no libcurl, no libarchive,
        no OpenMP), checks what the library depends on, reads a test picture
        with it and writes dist/tesseract-<platform>.tgz.
    python build.py smoke --lib <library> --tessdata <folder>
        Reads the test picture with a library that is already built.

Standard library only. `--cache <folder>` (pin and build) keeps the
downloaded sources there and uses them when their hashes match.
"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIN = ROOT / "sources.pin.json"
WORK = ROOT / "work"
DIST = ROOT / "dist"
SMOKE_PICTURE = ROOT / "smoke" / "text.pgm"
SMOKE_WORDS = ["Pinned", "build", "42"]

# A library's name must not hold any of these: the build is meant to
# depend on the operating system alone.
FORBIDDEN = re.compile(
    r"vcruntime|msvcp|ucrtbase|api-ms-win-crt|vcomp|gomp|libomp|stdc\+\+|gcc_s|"
    r"curl|archive|lept|png|jpeg|jpg|tiff|gif|webp|openjp|zlib|libz\.",
    re.IGNORECASE,
)


def say(text=""):
    print(text, flush=True)


def fail(text):
    say(f"FAIL  {text}")
    sys.exit(1)


def platform_name():
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        fail(f"only x64 is built here, this is {machine}")
    return "win-x64" if os.name == "nt" else "linux-x64"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "tesseract-build"})
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(request, timeout=120) as reply, open(part, "wb") as out:
                shutil.copyfileobj(reply, out)
            part.replace(dest)
            return
        except OSError as error:
            say(f"   download try {attempt} of {url}: {error}")
            time.sleep(5 * attempt)
    fail(f"could not download {url}")


def exists(url):
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tesseract-build"})
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            return reply.status == 200
    except OSError:
        return False


# ---------------------------------------------------------------- pin

def cmd_pin(args):
    cache = Path(args.cache).resolve()
    tess, lept, data = args.tesseract, args.leptonica, args.tessdata
    lept_asset = f"https://github.com/DanBloomberg/leptonica/releases/download/{lept}/leptonica-{lept}.tar.gz"
    lept_url = lept_asset if exists(lept_asset) else f"https://github.com/DanBloomberg/leptonica/archive/refs/tags/{lept}.tar.gz"
    raw = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{data}/"
    sources = {
        "tesseract": {
            "version": tess,
            "file": f"tesseract-{tess}.tar.gz",
            "url": f"https://github.com/tesseract-ocr/tesseract/archive/refs/tags/{tess}.tar.gz",
        },
        "leptonica": {"version": lept, "file": f"leptonica-{lept}.tar.gz", "url": lept_url},
        "tessdata": {"version": data, "file": f"eng-{data[:12]}.traineddata", "url": raw + "eng.traineddata"},
        "tessdataLicense": {"version": data, "file": f"tessdata_fast-{data[:12]}-LICENSE", "url": raw + "LICENSE"},
    }
    for name, entry in sources.items():
        dest = cache / entry["file"]
        download(entry["url"], dest)
        entry["sha256"] = sha256(dest)
        entry["bytes"] = dest.stat().st_size
        say(f"PIN  {name}  {entry['version']}  {entry['sha256']}  {entry['bytes']} bytes")
    PIN.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8", newline="\n")
    say(f"wrote {PIN.name}")
    # The licence texts of the exact versions, beside the sources, to be read.
    out = cache / "licences"
    out.mkdir(parents=True, exist_ok=True)
    for name, wanted in (("tesseract", "LICENSE"), ("leptonica", "leptonica-license.txt")):
        text = read_member(cache / sources[name]["file"], wanted)
        (out / f"{name}-{sources[name]['version']}-{wanted}").write_bytes(text)
        say(f"LICENCE  {name}  {wanted}  {hashlib.sha256(text).hexdigest()}  {len(text)} bytes")
    shutil.copyfile(cache / sources["tessdataLicense"]["file"], out / sources["tessdataLicense"]["file"])


def read_member(archive, wanted):
    """A file at the top of a source archive (below its one top folder)."""
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = member.name.split("/")
            if member.isfile() and len(parts) == 2 and parts[1] == wanted:
                return tar.extractfile(member).read()
    fail(f"{wanted} is not at the top of {archive.name}")


# -------------------------------------------------------------- build

def obtain(entry, cache):
    """The pinned file, from the cache when its hash matches, else downloaded."""
    dest = cache / entry["file"]
    if not (dest.exists() and sha256(dest) == entry["sha256"]):
        download(entry["url"], dest)
    found = sha256(dest)
    if found != entry["sha256"]:
        fail(f"{entry['file']}: SHA-256 {found}, pinned {entry['sha256']}")
    say(f"OK    {entry['file']}  {found}")
    return dest


def extract(archive, dest):
    """Unpacks a source archive without its one top folder."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        members = []
        for member in tar:
            parts = member.name.split("/", 1)
            if len(parts) < 2 or not parts[1] or member.name.startswith(("/", "..")) or "/../" in member.name:
                continue
            if not (member.isfile() or member.isdir()):
                continue  # no links, no devices
            member.name = parts[1]
            members.append(member)
        if sys.version_info >= (3, 12):
            tar.extractall(dest, members=members, filter="data")
        else:
            tar.extractall(dest, members=members)


def run(command, env=None):
    say("   $ " + " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], check=True, env=env)


def cmake(source, build, options, env=None):
    run(["cmake", "-S", source, "-B", build, *options], env)
    run(["cmake", "--build", build, "--config", "Release", "--parallel", str(os.cpu_count() or 2)], env)
    run(["cmake", "--install", build, "--config", "Release"], env)


def cmd_build(args):
    plat = platform_name()
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    cache = Path(args.cache).resolve() if args.cache else WORK / "sources"
    files = {name: obtain(entry, cache) for name, entry in pin.items()}

    lept_src, tess_src, prefix = WORK / "l", WORK / "t", WORK / "prefix"
    for folder in (WORK / "lb", WORK / "tb", prefix):
        if folder.exists():
            shutil.rmtree(folder)
    extract(files["leptonica"], lept_src)
    extract(files["tesseract"], tess_src)

    common = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW",
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",  # the C runtime inside the library
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DSW_BUILD=OFF",
    ]

    say("\n== Leptonica: static, no picture-format library")
    formats = ["ZLIB", "PNG", "GIF", "JPEG", "TIFF", "WEBP", "OPENJPEG"]
    packages = ["ZLIB", "PNG", "GIF", "JPEG", "TIFF", "WebP", "OpenJPEG"]
    cmake(
        lept_src,
        WORK / "lb",
        common
        + ["-DBUILD_SHARED_LIBS=OFF", "-DBUILD_PROG=OFF", "-DSTRICT_CONF=OFF"]
        + [f"-DENABLE_{name}=OFF" for name in formats]
        + [f"-DCMAKE_DISABLE_FIND_PACKAGE_{name}=TRUE" for name in packages],
    )
    check_leptonica_config(WORK / "lb")

    say("\n== Tesseract: one shared library, the LSTM engine only")
    env = dict(os.environ)
    env["PKG_CONFIG_PATH"] = str(prefix / "lib" / "pkgconfig")
    options = common + [
        "-DBUILD_SHARED_LIBS=ON",
        f"-DCMAKE_PREFIX_PATH={prefix.as_posix()}",
        f"-DLeptonica_DIR={(prefix / 'lib' / 'cmake' / 'leptonica').as_posix()}",
        "-DWIN32_MT_BUILD=ON",
        "-DBUILD_TRAINING_TOOLS=OFF",
        "-DBUILD_TESTS=OFF",
        "-DDISABLED_LEGACY_ENGINE=ON",
        "-DGRAPHICS_DISABLED=ON",
        "-DDISABLE_CURL=ON",
        "-DDISABLE_ARCHIVE=ON",
        "-DDISABLE_TIFF=ON",
        "-DOPENMP_BUILD=OFF",
        "-DENABLE_NATIVE=OFF",
        "-DENABLE_LTO=OFF",
        "-DUSE_SYSTEM_ICU=OFF",
        "-DINSTALL_CONFIGS=OFF",
        "-DENABLE_CCACHE=OFF",
    ]
    if plat == "linux-x64":
        options.append("-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc")
    cmake(tess_src, WORK / "tb", options, env)

    library = find_library(prefix, plat)
    say(f"\nbuilt {library}  {library.stat().st_size} bytes")

    say("\n== What the library depends on")
    needs = imports_pe(library) if plat == "win-x64" else needs_elf(library)
    for name in needs:
        say(f"NEEDS  {name}")
    bad = [name for name in needs if FORBIDDEN.search(name)]
    if bad:
        fail("the library depends on " + ", ".join(bad))
    say("PASS  it depends on the operating system's own libraries alone")

    say("\n== Package")
    name = f"tesseract-{plat}"
    stage = DIST / name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "tessdata").mkdir(parents=True)
    (stage / "licenses").mkdir()
    inside = Path("bin/tesseract.dll") if plat == "win-x64" else Path("lib/libtesseract.so")
    (stage / inside).parent.mkdir()
    shutil.copyfile(library, stage / inside)
    shutil.copyfile(files["tessdata"], stage / "tessdata" / "eng.traineddata")
    shutil.copyfile(tess_src / "LICENSE", stage / "licenses" / "tesseract-LICENSE.txt")
    shutil.copyfile(lept_src / "leptonica-license.txt", stage / "licenses" / "leptonica-license.txt")
    shutil.copyfile(files["tessdataLicense"], stage / "licenses" / "tessdata_fast-LICENSE.txt")

    version = cmd_smoke_with(stage / inside, stage / "tessdata")

    build_info = {
        "platform": plat,
        "tesseract": pin["tesseract"]["version"],
        "tesseractReports": version,
        "leptonica": pin["leptonica"]["version"],
        "tessdata_fast": pin["tessdata"]["version"],
        "library": inside.as_posix(),
        "librarySha256": sha256(stage / inside),
        "languageSha256": sha256(stage / "tessdata" / "eng.traineddata"),
        "needs": needs,
        "tesseractOptions": [option for option in options if not option.startswith("-DCMAKE_INSTALL") and "PREFIX" not in option and "Leptonica_DIR" not in option],
    }
    (stage / "BUILD.json").write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8", newline="\n")

    archive = DIST / f"{name}.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    say(f"LIBRARY  {plat}  {inside.as_posix()}  {build_info['librarySha256']}  {(stage / inside).stat().st_size} bytes")
    say(f"ARCHIVE  {archive.name}  {sha256(archive)}  {archive.stat().st_size} bytes")
    say("BUILD: PASS")


def check_leptonica_config(build):
    """Leptonica must have found no picture-format library."""
    found = sorted(build.rglob("config_auto.h"))
    if not found:
        fail("Leptonica's config_auto.h is not in its build folder")
    text = found[0].read_text(encoding="utf-8", errors="replace")
    seen = 0
    for line in text.splitlines():
        match = re.match(r"\s*(/\*\s*#\s*undef|#\s*define)\s+(HAVE_LIB\w+)\s*(\S*)", line)
        if match:
            seen += 1
            off = match.group(1).startswith("/*") or match.group(3) == "0"
            say(f"   {match.group(2)} {'off' if off else 'ON'}")
            if not off:
                fail(f"Leptonica was built with {match.group(2)}")
    if seen < 5:
        fail(f"only {seen} HAVE_LIB lines in {found[0].name}: its shape changed, read it")
    say("PASS  Leptonica has no picture-format library")


def find_library(prefix, plat):
    if plat == "win-x64":
        found = [path for path in (prefix / "bin").glob("*.dll") if "tesseract" in path.name.lower()]
    else:
        found = [path.resolve() for path in (prefix / "lib").glob("libtesseract.so*")]
        found = sorted(set(found))
    if len(found) != 1:
        fail(f"expected one Tesseract library under {prefix}, found {[path.name for path in found]}")
    return found[0]


def needs_elf(library):
    out = subprocess.run(["readelf", "-d", str(library)], check=True, capture_output=True, text=True).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


def imports_pe(library):
    """The DLL names in a PE32+ file's import and delay-import tables."""
    data = Path(library).read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        fail(f"{library} is not a PE file")
    sections_count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    if struct.unpack_from("<H", data, optional)[0] != 0x20B:
        fail(f"{library} is not a 64-bit PE file")
    directories = optional + 112
    table = optional + optional_size
    sections = []
    for index in range(sections_count):
        at = table + 40 * index
        virtual_size, virtual_address, raw_size, raw_at = struct.unpack_from("<IIII", data, at + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_at))

    def offset(rva):
        for start, size, raw_at in sections:
            if start <= rva < start + size:
                return raw_at + rva - start
        fail(f"{library}: address {rva:#x} is in no section")

    def text(rva):
        at = offset(rva)
        return data[at:data.index(b"\0", at)].decode("ascii", "replace")

    names = []
    for directory, size, name_at in ((1, 20, 12), (13, 32, 4)):
        rva = struct.unpack_from("<I", data, directories + 8 * directory)[0]
        if not rva:
            continue
        at = offset(rva)
        while any(data[at:at + size]):
            names.append(text(struct.unpack_from("<I", data, at + name_at)[0]))
            at += size
    return sorted(set(names), key=str.lower)


# -------------------------------------------------------------- smoke

def read_pgm(path):
    data = Path(path).read_bytes()
    match = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", data)
    if not match or match.group(3) != b"255":
        fail(f"{path} is not an 8-bit binary PGM")
    width, height = int(match.group(1)), int(match.group(2))
    pixels = data[match.end():]
    if len(pixels) != width * height:
        fail(f"{path}: {len(pixels)} bytes of pixels for {width} x {height}")
    return width, height, pixels


def cmd_smoke_with(library, tessdata):
    say("\n== Reading the test picture with the library")
    width, height, pixels = read_pgm(SMOKE_PICTURE)
    lib = ctypes.CDLL(str(Path(library).resolve()))
    lib.TessVersion.restype = ctypes.c_char_p
    lib.TessBaseAPICreate.restype = ctypes.c_void_p
    lib.TessBaseAPIInit3.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.TessBaseAPIInit3.restype = ctypes.c_int
    lib.TessBaseAPISetImage.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.TessBaseAPISetImage.restype = None
    lib.TessBaseAPISetSourceResolution.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.TessBaseAPISetSourceResolution.restype = None
    lib.TessBaseAPIGetUTF8Text.argtypes = [ctypes.c_void_p]
    lib.TessBaseAPIGetUTF8Text.restype = ctypes.c_void_p
    lib.TessBaseAPIMeanTextConf.argtypes = [ctypes.c_void_p]
    lib.TessBaseAPIMeanTextConf.restype = ctypes.c_int
    lib.TessDeleteText.argtypes = [ctypes.c_void_p]
    lib.TessDeleteText.restype = None
    lib.TessBaseAPIEnd.argtypes = [ctypes.c_void_p]
    lib.TessBaseAPIEnd.restype = None
    lib.TessBaseAPIDelete.argtypes = [ctypes.c_void_p]
    lib.TessBaseAPIDelete.restype = None

    version = lib.TessVersion().decode("ascii", "replace")
    say(f"TessVersion: {version}")
    api = lib.TessBaseAPICreate()
    if not api:
        fail("TessBaseAPICreate returned nothing")
    started = time.time()
    code = lib.TessBaseAPIInit3(api, os.fsencode(str(Path(tessdata).resolve())), b"eng")
    if code != 0:
        fail(f"TessBaseAPIInit3 returned {code}")
    lib.TessBaseAPISetImage(api, pixels, width, height, 1, width)
    lib.TessBaseAPISetSourceResolution(api, 300)
    pointer = lib.TessBaseAPIGetUTF8Text(api)
    if not pointer:
        fail("TessBaseAPIGetUTF8Text returned nothing")
    words = ctypes.string_at(pointer).decode("utf-8", "replace").strip()
    confidence = lib.TessBaseAPIMeanTextConf(api)
    lib.TessDeleteText(pointer)
    lib.TessBaseAPIEnd(api)
    lib.TessBaseAPIDelete(api)
    say(f"read in {time.time() - started:.2f} s, confidence {confidence}: {words!r}")
    missing = [word for word in SMOKE_WORDS if word not in words]
    if missing:
        fail(f"the picture's words were not read: {missing}")
    say("PASS  the library reads the test picture")
    return version


def cmd_smoke(args):
    cmd_smoke_with(args.lib, args.tessdata)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin")
    pin.add_argument("--tesseract", required=True)
    pin.add_argument("--leptonica", required=True)
    pin.add_argument("--tessdata", required=True, help="a commit of tesseract-ocr/tessdata_fast")
    pin.add_argument("--cache", default=str(WORK / "sources"))
    pin.set_defaults(run=cmd_pin)
    build = commands.add_parser("build")
    build.add_argument("--cache")
    build.set_defaults(run=cmd_build)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--lib", required=True)
    smoke.add_argument("--tessdata", required=True)
    smoke.set_defaults(run=cmd_smoke)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
