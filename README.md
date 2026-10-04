# tesseract-build

Builds one small [Tesseract](https://github.com/tesseract-ocr/tesseract) OCR
library for Windows (x64) and Linux (x64), from sources pinned by SHA-256.

- `sources.pin.json` names the exact sources: Tesseract, Leptonica and the
  English language data of `tessdata_fast`, each with its SHA-256.
- `build.py build` checks every source against its hash, builds Leptonica as
  a static library with **no picture-format library** (the caller hands
  Tesseract pixels), and Tesseract as **one shared library**: the LSTM engine
  only, no training tools, no libcurl, no libarchive, no OpenMP, the C and
  C++ runtimes inside the library. It then checks what the library depends
  on (the operating system's own libraries alone), reads a test picture with
  it, and writes `dist/tesseract-<platform>.tgz`.
- The workflow runs that on GitHub's runners. A pushed tag publishes a
  release: the two archives, `SHA256SUMS` and `sources.pin.json`.

An archive holds the library (`bin/tesseract.dll` or `lib/libtesseract.so`),
`tessdata/eng.traineddata`, `licenses/` (Tesseract: Apache 2.0; Leptonica:
BSD 2-clause; the language data: Apache 2.0) and `BUILD.json` (versions,
options, hashes).

## Moving a pin

```
python build.py pin --tesseract <tag> --leptonica <tag> --tessdata <commit of tessdata_fast>
```

writes `sources.pin.json` from fresh downloads. Commit it and push a tag
named `<tesseract version>-<build number>`.
