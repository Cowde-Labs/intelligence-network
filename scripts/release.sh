#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

target=${1:-x86_64-unknown-linux-gnu}
out_root=${2:-dist}
case "$target" in
  x86_64-unknown-linux-gnu) artifact_arch=linux-x86_64; deb_arch=amd64; os=linux ;;
  aarch64-unknown-linux-gnu) artifact_arch=linux-aarch64; deb_arch=arm64; os=linux ;;
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
if [[ "$os" == linux ]]; then
  command -v dpkg-deb >/dev/null 2>&1 || { echo "dpkg-deb is required to build Linux packages" >&2; exit 1; }
  command -v objdump >/dev/null 2>&1 || { echo "objdump is required to determine the glibc baseline" >&2; exit 1; }
  glibc_min=$(objdump -T "$binary" | sed -n 's/.*GLIBC_\([0-9][0-9.]*\).*/\1/p' | sort -Vu | tail -n 1)
  deb_root=$(mktemp -d "${TMPDIR:-/tmp}/intelligence-deb.XXXXXX")
  trap 'rm -rf "$deb_root"' EXIT INT TERM
  install -D -m 0755 "$binary" "$deb_root/usr/bin/intelligence"
  install -D -m 0644 config/node.toml.example "$deb_root/usr/share/doc/intelligence-network/node.toml.example"
  install -D -m 0644 README.md "$deb_root/usr/share/doc/intelligence-network/README.md"
  install -D -m 0644 LICENSE "$deb_root/usr/share/doc/intelligence-network/copyright"
  mkdir -p "$deb_root/DEBIAN"
  {
    printf 'Package: intelligence-network\n'
    printf 'Version: %s\n' "$version"
    printf 'Section: net\nPriority: optional\n'
    printf 'Architecture: %s\n' "$deb_arch"
    printf 'Maintainer: Cowde Labs <github@users.noreply.github.com>\n'
    printf 'Homepage: https://github.com/Cowde-Labs/intelligence-network\n'
    if [[ -n "$glibc_min" ]]; then printf 'Depends: libc6 (>= %s)\n' "$glibc_min"; fi
    printf 'Description: decentralized compute node for the Intelligence Network\n'
    printf ' A small Rust node for contributing bounded CPU/GPU compute to a peer network.\n'
  } > "$deb_root/DEBIAN/control"
  deb_file="$out_root/intelligence-network_${version}_${deb_arch}.deb"
  dpkg-deb --root-owner-group --build "$deb_root" "$deb_file" >/dev/null
  deb_checksum=$(sha256 "$deb_file" | awk '{print $1}')
  printf '%s  %s\n' "$deb_checksum" "$(basename "$deb_file")" > "$deb_file.sha256"
  echo "$deb_file"
fi
echo "$archive"
