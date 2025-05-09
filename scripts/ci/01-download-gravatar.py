#!/usr/bin/env python
import argparse
from pathlib import Path
from tomllib import load as load_toml

import requests
from PIL import Image


def download_and_save(*, avatar_path, avatar_url: str):
    Image.open(requests.get(avatar_url, stream=True).raw).convert("RGB").save(avatar_path)


def main(*, static: str, hugo: str):
    static_dir = Path(static).resolve(strict=True)

    with open(hugo, 'rb') as f:
        site_config = load_toml(f)

    params = site_config['params']
    avatar_url = params['urls']['gravatar']
    avatar_path = static_dir / params['staticPaths']['avatar']
    download_and_save(avatar_path=avatar_path, avatar_url=avatar_url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downloads and saves a gravatar to static/")
    parser.add_argument('--static', required=True, type=str, help="Path to /static directory")
    parser.add_argument('--hugo', required=True, type=str, help="Path to hugo.toml")

    args = parser.parse_args()
    main(static=args.static, hugo=args.hugo)
