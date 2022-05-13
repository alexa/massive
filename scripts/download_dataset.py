"""
"""

import argparse
import tarfile
from pathlib import Path

import tqdm
import requests


def http_get(url, out_dir):
    """Get contents of a URL and save to a file.

    https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    print(f"Downloading {url}.")
    out_dir = Path(out_dir)
    out_file = out_dir / Path(url).name
    if out_file.exists():
        raise FileExistsError(f"File {out_file} already exists. Delete it or skipping downloading.")
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403 or req.status_code == 404:
        raise Exception(f"Could not reach {url}.")
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    with open(out_file, "wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()
    return out_file


def untar_file(path_to_tar_file, dst_dir):
    """Decompress a .tar.xx file to folder path"""
    print(f"Extracting file {path_to_tar_file} to {dst_dir}.")
    mode = Path(path_to_tar_file).suffix.replace(".", "r:").replace("tar", "")
    with tarfile.open(str(path_to_tar_file), mode) as tar_ref:
        tar_ref.extractall(str(dst_dir))


def main():
    parser = argparse.ArgumentParser(description="Download the MASSIVE dataset")
    parser.add_argument("-d", "--massive-data-path", help="path to store the MASSIVE dataset")
    args = parser.parse_args()

    out_dir = Path(args.massive_data_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_file = http_get(
        "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz",
        out_dir
    )
    untar_file(tar_file, out_dir)


if __name__ == "__main__":
    main()
