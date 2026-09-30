#!/usr/bin/env bash
# Download and unpack the NASA MSL/SMAP telemetry dataset from Hundman et al. (2018).
# Reference: https://github.com/khundman/telemanom
#
# Usage: bash fetch_msl.sh
#
# Skips download if datasets/MSL/labeled_anomalies.csv already exists.

set -e
cd "$(dirname "$0")"
DEST="datasets/MSL"
mkdir -p "$DEST"

if [ -f "$DEST/labeled_anomalies.csv" ] && [ -d "$DEST/train" ] && [ -d "$DEST/test" ]; then
  echo "MSL dataset already present at $DEST — nothing to do."
  exit 0
fi

echo "Downloading MSL/SMAP data.zip (~200 MB) from NASA JPL S3..."
URL="https://s3-us-west-2.amazonaws.com/telemanom/data/data.zip"
TMPDIR=$(mktemp -d)
curl -L -o "$TMPDIR/data.zip" "$URL"

echo "Unzipping..."
unzip -q "$TMPDIR/data.zip" -d "$TMPDIR"

# The zip contains a data/ directory with train/, test/, labeled_anomalies.csv
if [ -d "$TMPDIR/data" ]; then
  cp -r "$TMPDIR/data/train" "$DEST/"
  cp -r "$TMPDIR/data/test" "$DEST/"
  cp "$TMPDIR/data/labeled_anomalies.csv" "$DEST/"
elif [ -d "$TMPDIR/train" ]; then
  # Some versions unpack differently
  cp -r "$TMPDIR/train" "$DEST/"
  cp -r "$TMPDIR/test" "$DEST/"
  cp "$TMPDIR/labeled_anomalies.csv" "$DEST/"
else
  echo "ERROR: unexpected zip contents. See $TMPDIR"
  exit 1
fi

rm -rf "$TMPDIR"

echo "Done. Contents of $DEST:"
ls -la "$DEST"
echo ""
echo "MSL channels available:"
python3 msl_loader.py
