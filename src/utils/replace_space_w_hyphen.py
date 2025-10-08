from pathlib import Path


def hyphenate_filenames(root: Path) -> int:
    count = 0
    for p in root.iterdir():
        if p.is_file():
            new_name = p.name.replace(" ", "-")
            if new_name != p.name:
                target = p.with_name(new_name)

                p.replace(target)
                count += 1
    return count


if __name__ == "__main__":
    root_dir = Path("../../data/processed/markdowns")
    if not root_dir.exists():
        raise SystemExit(f"Directory not found: {root_dir}")
    renamed = hyphenate_filenames(root_dir)
    print(f"Renamed {renamed} files in {root_dir}")
