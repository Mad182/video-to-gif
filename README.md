# video-to-gif

Convert video files to animated GIF using FFmpeg libraries for decoding the video
and custom Wu quantization + Floyd-Steinberg dithering for GIF encoding.

## Usage

```
video-to-gif -i input.mp4 -o output.gif [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i FILE` | Input video file (required) | - |
| `-o FILE` | Output GIF file (required) | - |
| `-r FPS` | Frame rate | 10 |
| `-ss TIME` | Start time in seconds | 0 |
| `-t TIME` | Duration in seconds | full video |
| `-w WIDTH` | Scale to exact width (auto height) | original |
| `-h HEIGHT` | Scale to exact height (auto width) | original |
| `-maxw N` | Scale down to N width only if source is wider | - |
| `-maxh N` | Scale down to N height only if source is taller | - |
| `-loop N` | Loop count: 0=infinite, 1=once, N=N times | 0 |
| `-fuzz N` | Fuzziness for transparent pixel optimization (default: 2) | 2 |

### Examples

```bash
# Basic conversion, 3 seconds at 15fps
video-to-gif -i video.mp4 -o out.gif -r 15 -t 3

# Extract 5 seconds starting at 2.5s, scale to 480px wide
video-to-gif -i clip.webm -o out.gif -r 10 -ss 2.5 -t 5 -w 480

# Scale to 360p height, loop 3 times
video-to-gif -i video.mp4 -o out.gif -h 360 -loop 3

# Fit within 800x600, only resize if the source exceeds this size
video-to-gif -i video.mp4 -o banner.gif -maxw 800 -maxh 600
```

## Features

- **Wu's optimal color quantization** with global palette across all frames
- **Floyd-Steinberg dithering** for high visual quality
- **Delta frame optimization** - only encodes changed pixels between frames
- **LZW compression** with proper GIF89a output
- **Lanczos scaling** via libswscale for high-quality downscaling
- **Rotation metadata** - automatically detects and applies rotation from video metadata
- **All FFmpeg-supported formats** - MP4, WebM, MOV, AVI, MKV, etc.

## Dependencies

- libavformat, libavcodec, libavutil, libswscale (FFmpeg libraries)

## Build setup:

    apt install g++ make pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

## Runtime setup:

    apt install libavformat61 libavcodec61 libavutil59 libswscale8

(Version numbers may vary by Debian release - the -dev packages are the safe bet since they pull the right versions automatically.)

## Build

```bash
make
```

## Install (/usr/local/bin/video-to-gif)

```bash
sudo make install
```

## Disclaimer

I built this while experimenting with various tools and methods for encoding GIFs, but it seems to work pretty well, so I'm releasing it here. I have only tested it on Debian, but there is nothing specific to it, so it should work anywhere provided the build dependencies are met.

It's mostly written by AI and using well know algorithms. I just put them together in a way that works for me. I'm not a lawyer, so I don't know what the legal implications of this are. Use at your own risk. I included the MIT license, as it seems one of the most permissive. I don't really care what you do with this code and I can hardly call it mine anyway, you could reproduce it in half an hour with AI.

For reading and scaling video files, it uses FFmpeg libraries, which are LGPL licensed. I'm not including the FFmpeg code or binaries here, you can get them from your operating system's package manager or from https://ffmpeg.org/download.html
