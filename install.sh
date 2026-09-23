#!/bin/sh
set -eu

# The installer deliberately downloads only the release binary and verifies
# its archive checksum. It never installs a model, runtime, or system service.
version=${INTELLIGENCE_VERSION:-v1.0.0}
case "$version" in
    v*) ;;
    *) version="v$version" ;;
esac

os=$(uname -s)
machine=$(uname -m)
case "$os:$machine" in
    Linux:x86_64|Linux:amd64)
        artifact_arch=linux-x86_64
        ;;
    Linux:aarch64|Linux:arm64)
        artifact_arch=linux-aarch64
        ;;
    *)
        echo "Intelligence Network has no release artifact for $os/$machine yet." >&2
        echo "Build from source or choose a published Linux x86_64/aarch64 release." >&2
        exit 2
        ;;
esac

if command -v curl >/dev/null 2>&1; then
    download() {
        curl --fail --silent --show-error --location --retry 3 -o "$2" -- "$1"
    }
elif command -v wget >/dev/null 2>&1; then
    download() {
        wget --quiet --tries=3 --output-document="$2" -- "$1"
    }
else
    echo "install requires curl or wget" >&2
    exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    echo "install requires sha256sum or shasum for checksum verification" >&2
    exit 2
fi

base_url=${INTELLIGENCE_RELEASE_BASE_URL:-"https://github.com/Cowde-Labs/intelligence-network/releases/download/$version"}
artifact="intelligence-network-${version}-${artifact_arch}.tar.gz"
checksum_asset="$artifact.sha256"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/intelligence-install.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

download "$base_url/$artifact" "$tmp_dir/$artifact"
download "$base_url/$checksum_asset" "$tmp_dir/$checksum_asset"

expected=$(awk -v name="$artifact" '$2 == name {print $1; exit}' "$tmp_dir/$checksum_asset")
if [ -z "$expected" ]; then
    expected=$(tr -d '[:space:]' < "$tmp_dir/$checksum_asset")
fi
if ! printf '%s\n' "$expected" | awk 'length($0) != 64 || $0 !~ /^[0-9a-fA-F]+$/ { exit 1 }'; then
    echo "release checksum is missing or malformed" >&2
    exit 1
fi
actual=$(sha256 "$tmp_dir/$artifact")
if [ "$actual" != "$expected" ]; then
    echo "release checksum mismatch for $artifact" >&2
    exit 1
fi

unsafe_entry=$(tar -tzf "$tmp_dir/$artifact" | awk '
    $0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/ { print; exit }
')
if [ -n "$unsafe_entry" ]; then
    echo "release archive contains an unsafe path: $unsafe_entry" >&2
    exit 1
fi
tar -xzf "$tmp_dir/$artifact" -C "$tmp_dir"
package_dir="$tmp_dir/intelligence-network-${version}-${artifact_arch}"
if [ ! -f "$package_dir/intelligence" ]; then
    echo "release archive does not contain the intelligence binary" >&2
    exit 1
fi

home=${HOME:-}
if [ -z "$home" ]; then
    echo "HOME is required to install into ~/.local/bin" >&2
    exit 1
fi
bin_dir="$home/.local/bin"
mkdir -p "$bin_dir"
install -m 0755 "$package_dir/intelligence" "$bin_dir/intelligence"

echo "Installed Intelligence Network $version to $bin_dir/intelligence"
case ":${PATH:-}:" in
    *:"$bin_dir":*) ;;
    *) echo "Add $bin_dir to PATH, then run: intelligence up" ;;
esac
