#!/usr/bin/env bash
# Verify signature.txt against actual zip files. Run before defense.
set -euo pipefail

if [ ! -f signature.txt ]; then
    echo "signature.txt not found. Run train.py first."
    exit 1
fi

cmd="shasum"
command -v "$cmd" >/dev/null 2>&1 || cmd="sha1sum"

failed=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    expected=$(echo "$line" | awk '{print $1}')
    name=$(echo "$line" | awk '{print $2}')
    if [ ! -f "$name" ]; then
        echo "Missing file: $name"
        failed=1
        continue
    fi
    actual=$("$cmd" "$name" | awk '{print $1}')
    if [ "$actual" = "$expected" ]; then
        echo "OK $name"
    else
        echo "MISMATCH $name"
        echo "   expected: $expected"
        echo "   actual:   $actual"
        failed=1
    fi
done < signature.txt

if [ "$failed" = 0 ]; then
    echo "All signatures verified."
else
    echo "Verification failed."
    exit 1
fi
