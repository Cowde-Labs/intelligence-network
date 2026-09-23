#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

target=${1:-x86_64-unknown-linux-gnu}
out_root=${2:-dist}
case "$target" in
  x86_64-unknown-linux-gnu) artifact_arch=linux-x86_64; os=linux ;;
  aarch64-unknown-linux-gnu) artifact_arch=linux-aarch64; os=linux ;;
  x86_64-apple-darwin) artifact_arch=macos-x86_64; os=macos ;;
  aarch64-apple-darwin) artifact_arch=macos-aarch64; os=macos ;;
  x86_64-pc-windows-msvc) artifact_arch=windows-x86_64; os=windows ;;
  *) echo "unsupported release target: $target" >&2; exit 2 ;;
esac

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1"; }
else
  sha256() { shasum -a 256 "$1"; }
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-.cache/rust-target}"
export CARGO_INCREMENTAL=0
export INTELLIGENCE_COMMIT="${INTELLIGENCE_COMMIT:-${SOURCE_VERSION:-unknown}}"
cargo build -p intelligence-cli --release --target "$target"

bin_name=intelligence
[[ "$os" == windows ]] && bin_name=intelligence.exe
binary="$CARGO_TARGET_DIR/$target/release/$bin_name"
[[ -f "$binary" ]] || { echo "release binary missing: $binary" >&2; exit 1; }

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
package_dir="$out_root/intelligence-network-v${version}-${artifact_arch}"
rm -rf "$package_dir"
mkdir -p "$package_dir"
install -m 0755 "$binary" "$package_dir/$bin_name"
install -m 0644 config/node.toml.example "$package_dir/node.toml.example"
if [[ "$os" == linux ]]; then
  install -m 0644 infra/systemd/intelligence-node.service "$package_dir/intelligence-node.service"
fi
INTELLIGENCE_COMMIT="$INTELLIGENCE_COMMIT" "$package_dir/$bin_name" version > "$package_dir/version.json"
sha256 "$package_dir/$bin_name" > "$package_dir/SHA256SUMS"

if [[ "$os" == windows ]]; then
  archive="$package_dir.zip"
  rm -f "$archive"
  if command -v 7z >/dev/null 2>&1; then
    (cd "$out_root" && 7z a -tzip "$(basename "$archive")" "$(basename "$package_dir")" >/dev/null)
  else
    powershell -NoProfile -Command "Compress-Archive -Path '$package_dir' -DestinationPath '$archive'"
  fi
else
  archive="$package_dir.tar.gz"
  if tar --version 2>/dev/null | grep -q GNU; then
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -czf "$archive" -C "$out_root" "$(basename "$package_dir")"
  else
    tar -czf "$archive" -C "$out_root" "$(basename "$package_dir")"
  fi
fi
sha256 "$archive" >> "$package_dir/SHA256SUMS"
archive_checksum=$(sha256 "$archive" | awk '{print $1}')
printf '%s  %s\n' "$archive_checksum" "$(basename "$archive")" > "$archive.sha256"
echo "$archive"
