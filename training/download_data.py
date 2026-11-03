"""Fetch and verify the official release datasets; parallelize the large file."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = "https://github.com/GDG-UAM/ml-contest-materials/releases/download/v1.0/"
FILES = {
    "train_100k.npz": "1c231e9a807c348b34290415f56e7c7b99543a81673a9d3cef1002b1a42e5668",
    "test_demo.npz": "03fce51b3dacfaaa04418c4907def59874c4b6963c0561c9ccac83bbca90aac0",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(name, expected):
    target = ROOT / "data" / name
    target.parent.mkdir(exist_ok=True)
    if target.exists() and digest(target) == expected:
        print(f"Verified {name}", flush=True)
        return
    with urlopen(Request(BASE + name, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
        url = response.url
    partial = target.with_suffix(".download")
    partdir = ROOT / "data" / (name + ".parts")
    partdir.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    chunks = [(i, start, min(start + chunk_size, size) - 1)
              for i, start in enumerate(range(0, size, chunk_size))]

    def fetch(chunk):
        i, start, end = chunk
        path = partdir / f"{i:03d}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(4):
            try:
                request = Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=120) as response:
                    if size > chunk_size and (response.status != 206 or
                            not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/")):
                        raise ValueError("Server did not honor the requested byte range")
                    body = response.read()
                if len(body) != end - start + 1:
                    raise ValueError("Incomplete download chunk")
                path.write_bytes(body)
                return path
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    start_time = time.monotonic()
    print(f"Downloading {name} ({size / 1e6:.1f} MB), reusing completed chunks", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, chunk) for chunk in chunks]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"{name}: {n}/{len(chunks)} chunks ({time.monotonic()-start_time:.0f}s)", flush=True)
    with partial.open("wb") as stream:
        for i, _, _ in chunks:
            stream.write((partdir / f"{i:03d}.part").read_bytes())
    actual = digest(partial)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {name}: {actual}")
    partial.replace(target)
    # Delete only the verified download chunks created by this script.
    for i, _, _ in chunks:
        (partdir / f"{i:03d}.part").unlink()
    partdir.rmdir()
    print(f"Verified {name}: {actual}", flush=True)


if __name__ == "__main__":
    for name, expected in FILES.items():
        download(name, expected)
