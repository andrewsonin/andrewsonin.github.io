#!/usr/bin/env python
import argparse
import re
from os import makedirs
from os.path import join as join_path
from pathlib import Path
from tomllib import load as load_toml
from urllib.parse import urljoin as join_url

import yaml
from og_preview import generate_og_images, ArticleInfo


def parse_front_matter(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not match:
        return {}

    front_matter_yaml = match.group(1)
    metadata = yaml.safe_load(front_matter_yaml)

    return metadata


def main(*, content: str, static: str, logo: str, hugo: str):
    content_dir = Path(content).resolve(strict=True)
    static_dir = Path(static).resolve(strict=True)

    with open(hugo, 'rb') as f:
        site_config = load_toml(f)

    base_url = site_config['baseurl']
    avatar_path = static_dir / site_config['params']['staticPaths']['avatar']

    previews_dir = static_dir / 'previews'

    articles = []
    for md_file in content_dir.rglob('*.md'):
        front_matter = parse_front_matter(md_file)
        if front_matter.get('type') != 'article':
            continue

        rel_path = md_file.relative_to(content_dir)
        name = md_file.name.removesuffix('.md')
        parent = rel_path.parent

        output_dir = join_path(previews_dir, parent)
        makedirs(output_dir, exist_ok=True)
        articles.append(
            ArticleInfo(
                title=front_matter['title'],
                description=front_matter['description'],
                author=front_matter.get('author', 'Andrew Sonin'),
                url=join_url(base_url, front_matter['alias']).removeprefix('https://'),
                output_path=join_path(output_dir, f'{name}.png')
            )
        )
        print(f'Article path found: {md_file}')

    generate_og_images(*articles, avatar_path=avatar_path, logo_path=logo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates OG Images for Articles")
    parser.add_argument('--content', required=True, type=str, help="Path to /content directory")
    parser.add_argument('--static', required=True, type=str, help="Path to /static directory")
    parser.add_argument('--logo', required=True, type=str, help="Path to logo")
    parser.add_argument('--hugo', required=True, type=str, help="Path to hugo.toml")

    args = parser.parse_args()

    main(content=args.content, static=args.static, logo=args.logo, hugo=args.hugo)
