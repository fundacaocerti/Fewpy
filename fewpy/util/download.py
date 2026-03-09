import hashlib
import warnings
import sys

from pathlib import Path
from urllib.request import urlopen

from typing import List, Union

from tqdm import tqdm


model2url = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
}

def download(
    model: str,
    cache_dir: Union[str, None] = None,
):
    
    url = model2url[model]

    if not cache_dir:
        current_dir = Path(sys.path[0])
        cache_dir = current_dir / "weights"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    elif "mlfoundations" in url:
        expected_sha256 = Path(filename).stem.split("-")[-1]
    else:
        expected_sha256 = ""

    download_target = cache_dir / filename

    if download_target.exists() and not download_target.is_file():
        raise RuntimeError(
            f"{download_target} exists and is not a regular file"
        )

    if download_target.is_file():
        if expected_sha256:
            if (
                hashlib.sha256(download_target.read_bytes())
                .hexdigest()
                .startswith(expected_sha256)
            ):
                return str(download_target)
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
                )
        else:
            return str(download_target)

    with (
        urlopen(url) as source,
        download_target.open("wb") as output,
    ):
        with tqdm(
            total=int(source.headers.get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(
        download_target.read_bytes()
    ).hexdigest().startswith(expected_sha256):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return str(download_target)
