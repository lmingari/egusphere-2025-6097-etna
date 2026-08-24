#!/usr/bin/env bash
# Downloads files listed in urls.txt into data/, skipping ones that already
# exist with a matching checksum, and verifies everything against
# data/sha256sums.txt at the end.
#
# Expects:
#   urls.txt            -> "<filename> <url>" per line
#   data/sha256sums.txt -> output of `sha256sum data/*`

set -euo pipefail

DATA_DIR="data2"
URLS_FILE="$DATA_DIR/urls.txt"
SUMS_FILE="$DATA_DIR/sha256sums.txt"

if [[ ! -f "$URLS_FILE" ]]; then
    echo "error: $URLS_FILE not found" >&2
    exit 1
fi

if [[ ! -f "$SUMS_FILE" ]]; then
    echo "error: $SUMS_FILE not found" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"

# Look up the expected sha256 for a given filename from sums file.
# sha256sums.txt lines look like: "<hash>  data/<filename>"
expected_hash() {
    local fname="$1"
    # Match on basename of column 2, so it doesn't matter whether the sums
    # file stores "data/file", "./file", or just "file".
    awk -v f="$fname" '{n=split($2,p,"/"); if (p[n]==f) {print $1; found=1}} END{if(!found) exit 1}' "$SUMS_FILE"
}

file_matches_hash() {
    local filepath="$1" want="$2"
    local got
    got=$(sha256sum "$filepath" | awk '{print $1}')
    [[ "$got" == "$want" ]]
}

while IFS=' ' read -r filename url; do
    [[ -z "$filename" || "$filename" == \#* ]] && continue

    dest="$DATA_DIR/$filename"

    want_hash=$(expected_hash "$filename") || {
        echo "warning: no checksum entry for '$filename' in $SUMS_FILE, skipping" >&2
        continue
    }

    if [[ -f "$dest" ]] && file_matches_hash "$dest" "$want_hash"; then
        echo "ok (cached):  $filename"
        continue
    fi

    echo "downloading:  $filename"
    curl -fL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"

    if file_matches_hash "$dest" "$want_hash"; then
        echo "ok (verified): $filename"
    else
        echo "error: checksum mismatch for $filename" >&2
        exit 1
    fi
done < "$URLS_FILE"

echo "all files present and verified."
