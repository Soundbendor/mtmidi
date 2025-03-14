#!/bin/bash

# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
CURDIR=$(dirname "$(realpath "$0")");
SYNTOML="../syntheory/midi2audio/Cargo.toml"
IN_DEFAULT="$CURDIR/midi"
IN_DIR=${1:-$IN_DEFAULT}
OUT_DEFAULT="/media/dxk/TOSHIBA EXT/wav"
SOUNDFONT="$CURDIR/TimGM6mb.sf2"
OUT_DIR=${2:-$OUT_DEFAULT}
for f in $IN_DIR/*mid;
do 
    F_NAME=$(basename ${f});
    OUT_FILE="${OUT_DIR}/${F_NAME%.*}.wav"
    cargo run --manifest-path "${SYNTOML}" -- "${SOUNDFONT}" "${f}" "${OUT_FILE}";
done;
