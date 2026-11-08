# Playback components

`decoder_worker_bundle.js` and `decoder/wasm/wrapper.wasm` are the unmodified
AOMedia IAMF decoder demo binaries used by Orchestra Space. Upstream:
[decoder demo](https://aomediacodec.github.io/iamf-tools/decoder_demo/) and
[iamf-tools source](https://github.com/AOMediaCodec/iamf-tools). The accompanying
`LICENSE` and `PATENTS` apply to those components.

`monitor-filters.npz` contains the measured impulse responses of that same
build for a lossless 48 kHz 7.1.4 test sequence. It uses no third-party recording.
The regeneration script is `tests/rebuild_monitor_filters.py`. The inputs
follow the FFmpeg channel order used by StemSpace's exporter; the outputs
follow the decoder channel order used by its standalone playback.

`decode.mjs` is the StemSpace transport shim for running the worker under
Node.js, and writes the returned PCM to a temporary WAV.
