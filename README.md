# StemSpace 1.0.0

StemSpace is a local Python application for separating a stereo or mono recording with MLX, positioning and auditioning the resulting tracks, and exporting and playing standalone IAMF. It runs separation on Apple Silicon.

## Quick start on Apple Silicon

Use native arm64 Python 3.11 or later. Python 3.12 works with the pinned separator. Install FFmpeg with IAMF muxing and libopus support, then open a terminal in the unzipped `stemspace` folder:

```bash
brew install ffmpeg node
ffmpeg -hide_banner -muxers 2>/dev/null | grep iamf
ffmpeg -hide_banner -encoders 2>/dev/null | grep libopus

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ui,mlx]'
stemspace ui
```

If `python3.12` isn't on your PATH, use the arm64 Python 3.11+ executable from your working MLX setup. You can also install `-e '.[ui]'` into the virtual environment where the two models already work. The application uses `mlx-audio-separator` on PATH; if you keep separate environments, set `STEMSPACE_MLX_SEPARATOR` to the absolute path to its executable. `STEMSPACE_FFMPEG` and `STEMSPACE_FFPROBE` can override the media tools. Node.js 18+ runs the bundled decoder for saved files; set `STEMSPACE_NODE` if needed. Live editor playback does not require Node.

1. Click **New from audio**, select the recording, and choose a parent folder. The app creates a folder named after your recording. First-time model downloads and conversions can take a while. The app reports which stage it is running.
2. Click a marker in the **3D listener scene** or choose a track from the list. A badge counts overlapping markers; click the same spot again to cycle through those tracks. Drag the marker to change azimuth, or drag its height handle up or down to change elevation. Shift-dragging a marker also changes elevation. The numeric fields remain available for precise placement.
3. Use **Play** and the seek bar below the scene to audition your edits. Check the tracks you want to hear, or use **Solo** for the selected track and **All** to bring the others back. These checks do not affect export. Use **Mute in export** in the placement controls to change the saved mix.
4. Save your scene or click **Export IAMF**. Open that file in the **IAMF Player** tab and press **Play**. To try the player without a project, open `examples/speaker-check.iamf`. It activates each of the 12 decoded channels in turn. The shorter `examples/position-check.iamf` repeats the same sound at front, rear right, elevated rear right, and left.

Drag empty space (or drag with the secondary mouse button) to orbit. Scroll or use **−** / **+** to zoom; **Reset view** restores the starting camera. Camera controls do not edit the mix.

| Scene cue | What it shows |
| --- | --- |
| Marker and dashed drop line | The object's direction and its height above the ear-level plane. Positive azimuth is right, 0° is front, and ±180° is rear. |
| Arc with two endpoints | Stereo spread around the center azimuth. Width 0° is a point. Near overhead, the selected object's dashed floor arc shows the azimuth spread as the 3D arc contracts. |
| Ring around the marker | Gain relative to the selected marker's dotted 0 dB reference ring. The center stays easy to click at any gain. Muted markers are gray and crossed out. |

Already using an earlier version? Install from this folder into the same environment with `python -m pip install -e '.[ui]'`, then open your existing `project.json`. Its format, stem cache and IAMF export are unchanged. `stemspace --version` shows the installed version.

## Audio Output

The transport below the tabs is shared. Each tab keeps its own playhead when you switch. Choose **Audio Output…** to select the device and render layout:

| Layout | Required outputs |
| --- | ---: |
| Binaural headphones | 2 |
| Stereo speakers | 2 |
| 5.1 speakers | 6 |
| 7.1 speakers | 8 |
| 7.1.4 speakers | 12 |

StemSpace offers layouts the selected device can open. A two-channel device can
be headphones or speakers, so choose explicitly. The channel numbers refer to
the device's PCM outputs. Assign each named speaker to a distinct channel and
press **Test** to verify it, particularly the side, rear, and height pairs.
The app remembers mappings per device and layout. If a device or layout is
missing, check that macOS exposes the expected channels in Audio MIDI Setup.

Changing the output pauses playback and keeps the playhead. A saved file is
decoded again if you change its layout; press Play to resume when it is ready.
StemSpace sends rendered PCM to the device. It accepts standalone `.iamf`
files, and does not send an IAMF bitstream to the receiver or demux MP4.

Playback renders audio ahead of the device and starts after filling
a short buffer. The buffer lets playback ride through short render or disk
stalls, at the cost of a brief delay when starting or applying live edits.
If the output device still reports a gap, the transport shows an **Underruns**
count for that playback run. Seeking or restarting playback clears the count.

## Speaker activity

The IAMF Player shows seven ear-level speakers and four overhead speakers in a
3D listener view, with a separate LFE meter. The display uses the decoded 7.1.4
signal from the file, so it also works when listening on headphones. It needs
no project, stems, or companion file. Speaker activity describes channel
signals; it does not identify which instruments contributed to them.

Drag to orbit and scroll or use the buttons to zoom. Click a speaker marker or
use the **Inspect speaker** list to read its RMS level and sample peak in dBFS.
The markers and meters follow playback and seeking and freeze on pause.
Monitor volume and physical output mapping do not affect the readings. Most
StemSpace exports have a silent LFE channel by design; the speaker-check file
includes an LFE burst.

Playback can start while the 7.1.4 view prepares in the background. If playback
already uses 7.1.4, both use the same decoded audio. If a file cannot be
rendered to 7.1.4 by the bundled decoder, the view reports that it is unavailable
and playback can still use a layout the file supports.

You can also run the pipeline from the terminal:

```bash
stemspace separate "/path/to/Mr-Know-It-All.wav" "./Mr-Know-It-All StemSpace"
stemspace check "./Mr-Know-It-All StemSpace"
stemspace ui "./Mr-Know-It-All StemSpace"
stemspace export "./Mr-Know-It-All StemSpace" "./Mr-Know-It-All.iamf"
```

## Cascade configuration

The built-in [cascade.yaml](stemspace/cascade.yaml) runs:

| Stage | Input | Outputs |
| --- | --- | --- |
| `roformer` | Original audio | bass, drums, vocals, guitar, piano, other |
| `drumsep` | `roformer.drums` | kick, snare, toms, hh, ride, crash |

The scene selects 11 tracks. It replaces the parent drums stem with the six drum parts and keeps the parent in the cache. To experiment with more stages, copy the built-in YAML and edit it. Use `stemspace separate AUDIO PROJECT --recipe my-cascade.yaml`, or choose **File → New with YAML recipe** in the app.

- Declare stages in order. A stage reads `source` or a prior `stage.stem`. Several stages may read the same output.
- `outputs` can be a list or a mapping from model output label to stem ID, such as `{"Lead Vocals": lead}`. The names must match the model's actual outputs; stage IDs and stem IDs use lowercase letters, digits, underscores or hyphens.
- `scene` is the ordered list of outputs to edit and export. Don't include both a parent stem and its split parts unless you want both mixed in.
- Optional stage settings are `sample_rate`, `revision`, and `options` with positive integer `mdxc_segment_size`, `mdxc_overlap`, or `mdxc_batch_size`. Change `revision` if you replace model weights without changing the filename.
- Stage cache keys include the stage configuration, input, and separator version or executable identity. Changing only DrumSep settings reuses RoFormer's outputs. Existing positions for matching track IDs are carried forward. If a cache directory is damaged, move that directory aside and rerun separation.

## Project files and reconstruction

A project folder contains `project.json` (track positions, paths, recipe, and check results) and a `cache` folder with the stereo reference, model outputs, final dry WAVs, and `residual.wav`. Keep the whole folder when moving a project. The original recording is needed to rerun separation, but can be absent when reopening or exporting a saved scene.

The reconstruction reference is decoded float WAV at the source sample rate. Mono input is copied to both stereo channels. Models run at their configured rates, and the final WAVs are brought back to the reference timeline. The separate residual is the reference minus the sum of the 11 dry tracks. The app verifies the reconstruction from files on disk with a peak tolerance of 1e-6 full scale. The check says nothing about how clean an individual stem is. You can find and audition `residual.wav` at the relative path in `project.json`, or add it as a wide front layer via the checkbox. It is excluded from the spatial export by default.

## What the IAMF contains

This version exports a **7.1.4 channel-based speaker mix**, encoded with Opus at 48 kHz, with the binaural headphone rendering flag. The panner distributes each track across the speaker positions, including height speakers. Bass and kick go to the normal speaker channels; StemSpace leaves the LFE channel silent. The exporter applies shared headroom when needed. The dry reconstruction guarantee does not apply after panning, gain changes, or lossy encoding.

The editable objects live in the project; the saved IAMF file does not carry independent object metadata. IAMF 2.0 defines native objects, but a native authoring and playback path was not verified here. Exporting an Eclipsa delivery MP4 is also separate from this standalone `.iamf` workflow.

For monitoring, the editor uses the same stem-to-7.1.4 panning math as export and applies the bundled AOMedia renderer's measured response for the chosen layout. It reads the stems in sync and fades placement changes and Listen toggles. The IAMF Player uses the AOMedia decoder itself, runs the decode in the background, and stores temporary PCM for playback and seeking. Live and encoded playback can differ because of export headroom, Opus, and the file's own rendering metadata. See [playback component notices](stemspace/vendor/NOTICES.md).

The export is based on [AOMedia's 7.1.4 FFmpeg workflow](https://github.com/AOMediaCodec/iamf-tools/blob/main/docs/external/encoding_with_external_tools.md#encoding-714-wav-to-iamf). The [IAMF specification](https://aomediacodec.github.io/iamf/) describes the difference between channel-based and native object audio.

## Development checks

```bash
python -m unittest discover -s tests -v
python tests/make_position_demo.py ./demo-project ./demo.iamf
node tests/verify_web_decoder.mjs /path/to/orchestra-space ./demo.iamf
```

The tests cover the cascade, reconstruction, 3D interactions, live playback and
routing with a simulated audio device, standalone decoding in all five
layouts when Node is available, and speaker order and playhead synchronization.
The optional Orchestra Space check verifies its
actual web decoder. Use Audio Output's test tones to verify physical speakers
on your Mac; the development checks do not exercise CoreAudio hardware.
