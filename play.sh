#!/bin/bash
cd "$(dirname "$0")"

./mac_vrx | ffplay \
  -fs \
  -probesize 32 \
  -analyzeduration 0 \
  -fflags nobuffer \
  -flags low_delay \
  -flags2 +showall \
  -err_detect ignore_err \
  -threads 1 \
  -an -sn \
  -vf setpts=0 \
  -sync ext \
  -framedrop \
  -f hevc \
  -framerate 60 \
  -i -
