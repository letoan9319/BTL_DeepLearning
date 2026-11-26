"""Download and verify the six Stanford IWSLT15 English-Vietnamese files."""

import hashlib
import io
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "iwslt15"
BASE = "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/93193ddf269449c98495e3455b2bb261b4851234/data"
ARCHIVES = {"train": "train-en-vi", "tst2012": "dev-2012-en-vi", "tst2013": "test-2013-en-vi"}
CHECKSUMS = {
    "train.en": "c26dfeed74b6bf3752f5ca552f2412456f0de153f7c804df8717931fb3a5c78a",
    "train.vi": "707206edf2dc0280273952c7b70544ea8a1363aa69aaeb9d70514b888dc3067d",
    "tst2012.en": "b8fbcd0d4199d41276421d1f7228298df67dd9e2c5c134c2ee8cdf3ec27801e2",
    "tst2012.vi": "01004078b3f36b0c46e81b65fc6851230b3c823405c76d1d82ef3b786b512ff0",
    "tst2013.en": "3860ce60bc5b85d96cc2d3d580cf1bc9d62440e939c1b964676d3607c4e8f1df",
    "tst2013.vi": "29af16842808da2812b8fd4d47fc05d082ba870e60ab7755090ad04b380994cd",
}


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    for split, archive in ARCHIVES.items():
        paths = [DATA / f"{split}.{lang}" for lang in ("en", "vi")]
        missing = [p for p in paths if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != CHECKSUMS[p.name]]
        if missing:
            with urlopen(f"{BASE}/{archive}.tgz", timeout=90) as response:
                compressed = io.BytesIO(response.read())
            with tarfile.open(fileobj=compressed, mode="r:gz") as bundle:
                for path in missing:
                    content = bundle.extractfile(path.name).read()
                    if hashlib.sha256(content).hexdigest() != CHECKSUMS[path.name]:
                        raise ValueError(f"Invalid SHA-256 for {path.name}")
                    with NamedTemporaryFile(dir=DATA, delete=False) as tmp:
                        tmp.write(content)
                    Path(tmp.name).replace(path)
        counts = [len(p.read_text(encoding="utf-8").splitlines()) for p in paths]
        if counts[0] != counts[1]:
            raise ValueError(f"Unaligned files: {split}")
        print(f"{split}: {counts[0]:,} pairs (SHA-256 OK)")


if __name__ == "__main__":
    download()
