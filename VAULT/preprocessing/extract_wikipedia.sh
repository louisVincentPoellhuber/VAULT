#!/bin/bash

# Variables
# TODO: Make variable
WIKI_DUMP="$STORAGE_DIR/datasets/vault/wikipedia/downloads/enwiki-latest-pages-articles.xml.bz2"
WIKI_OUT_DIR="$STORAGE_DIR/datasets/vault/wikipedia/downloads"

if [ ! -d "$WIKI_OUT_DIR" ]; then
    mkdir -p "$WIKI_OUT_DIR"
fi

# Step 1: Download dump if not present
if [ ! -f "$WIKI_DUMP" ]; then
    echo "Downloading Wikipedia dump..."
    curl -L --progress-bar -C - \
        https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
        -o "$WIKI_DUMP"
else
    echo "Wikipedia dump already exists, skipping download."
fi

# Step 2: Run WikiExtractor if no subfolders exist in output
if [ -z "$(find "$WIKI_OUT_DIR" -mindepth 1 -type d 2>/dev/null)" ]; then
    echo "Extracting Wikipedia dump..."
    python wikiextractor/WikiExtractor.py \
        --json --processes 16 \
        --output "$WIKI_OUT_DIR" \
        "$WIKI_DUMP"
else
    echo "WikiExtractor output already exists, skipping extraction."
fi
