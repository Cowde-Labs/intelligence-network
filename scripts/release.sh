#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

target=${1:-x86_64-unknown-linux-gnu}
out_root=${2:-dist}
case "$target" in
  x86_64-unknown-linux-gnu) artifact_arch=linux-x86_64 ;;
  aarch64-unknown-linux-gnu) artifact_arch=linux-aarch64 ;;
  *) echo "unsupported release target: $target" >&2; exit 2 ;;
esac

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-.cache/rust-target}"
export CARGO_INCREMENTAL=0
export INTELLIGENCE_COMMIT="${INTELLIGENCE_COMMIT:-${SOURCE_VERSION:-unknown}}"
cargo build -p intelligence-cli --release --target "$target"

binary="$CARGO_TARGET_DIR/$target/release/intelligence"
[[ -x "$binary" ]] || { echo "release binary missing: $binary" >&2; exit 1; }
if command -v strip >/dev/null 2>&1; then
  strip --strip-all "$binary" 2>/dev/null || true
fi

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
package_dir="$out_root/intelligence-network-v${version}-${artifact_arch}"
rm -rf "$package_dir"
mkdir -p "$package_dir"
install -m 0755 "$binary" "$package_dir/intelligence"
install -m 0644 config/node.toml.example "$package_dir/node.toml.example"
install -m 0644 infra/systemd/intelligence-node.service "$package_dir/intelligence-node.service"
INTELLIGENCE_COMMIT="$INTELLIGENCE_COMMIT" "$package_dir/intelligence" version > "$package_dir/version.json"
sha256sum "$package_dir/intelligence" > "$package_dir/SHA256SUMS"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -czf "$package_dir.tar.gz" -C "$out_root" "$(basename "$package_dir")"
sha256sum "$package_dir.tar.gz" >> "$package_dir/SHA256SUMS"
echo "$package_dir.tar.gz"
