/*
 * video-to-gif - Convert video to animated GIF directly using FFmpeg libraries.
 *
 * Decodes video frames with libavformat/libavcodec, scales with libswscale,
 * and encodes GIF in-memory using Wu quantization + Floyd-Steinberg dithering.
 *
 * Usage: video-to-gif -i input.mp4 -o output.gif [options]
 *
 *   -i FILE    Input video file (required)
 *   -o FILE    Output GIF file (required)
 *   -r FPS     Frame rate (default: 10)
 *   -ss TIME   Start time in seconds (default: 0)
 *   -t TIME    Duration in seconds (default: entire video)
 *   -w WIDTH   Scale to width, preserving aspect ratio (-1 = auto)
 *   -h HEIGHT  Scale to height, preserving aspect ratio (-1 = auto)
 *   -loop N    Loop count: 0=infinite (default), 1=play once, N=play N times
 *   -maxw N    Maximum width (default: no limit)
 *   -maxh N    Maximum height (default: no limit)
 *   -fuzz N    Fuzziness for transparent pixel optimization (default: 2)
 *
 * Dependencies: libavformat, libavcodec, libavutil, libswscale
 * Build: see Makefile
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <climits>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/display.h>
#include <libswscale/swscale.h>
}

// =====================================================================
// Types
// =====================================================================

struct RGBAFrame {
    std::vector<uint8_t> pixels; // RGBA, row-major
    int width, height;
};

// =====================================================================
// Video orientation - detect and apply rotation metadata
// =====================================================================

// Detect rotation angle from stream side data or metadata.
// Returns a normalized angle: 0, 90, 180, or 270.
static int detect_rotation(AVStream *stream) {
    double theta = 0.0;

    // Method 1: displaymatrix in codecpar side data (preferred)
    const AVPacketSideData *sd = nullptr;
    for (int i = 0; i < stream->codecpar->nb_coded_side_data; i++) {
        if (stream->codecpar->coded_side_data[i].type == AV_PKT_DATA_DISPLAYMATRIX) {
            sd = &stream->codecpar->coded_side_data[i];
            break;
        }
    }
    if (sd) {
        theta = -av_display_rotation_get((const int32_t *)sd->data);
    }

    // Method 2: "rotate" metadata tag (legacy fallback)
    if (theta == 0.0) {
        AVDictionaryEntry *tag = av_dict_get(stream->metadata,
                                             "rotate", nullptr, 0);
        if (tag && tag->value) {
            theta = atof(tag->value);
        }
    }

    // Normalize to 0, 90, 180, 270
    int angle = ((int)round(theta)) % 360;
    if (angle < 0) angle += 360;

    // Snap to nearest 90° increment
    if (angle >= 315 || angle < 45)   return 0;
    if (angle >= 45  && angle < 135)  return 90;
    if (angle >= 135 && angle < 225)  return 180;
    return 270;
}

// Rotate an RGBA pixel buffer in-place.
// rotation must be 0, 90, 180, or 270.
static void rotate_rgba(std::vector<uint8_t> &pixels,
                        int &width, int &height, int rotation) {
    if (rotation == 0) return;

    int w = width, h = height;

    if (rotation == 180) {
        // Reverse all pixels in-place
        int total = w * h;
        for (int i = 0; i < total / 2; i++) {
            int j = total - 1 - i;
            for (int c = 0; c < 4; c++)
                std::swap(pixels[i * 4 + c], pixels[j * 4 + c]);
        }
        // Dimensions unchanged
        return;
    }

    // 90° or 270° - dimensions swap
    int new_w = h;
    int new_h = w;
    std::vector<uint8_t> dst((size_t)new_w * new_h * 4);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int src_idx = (y * w + x) * 4;
            int dx, dy;
            if (rotation == 90) {
                // (x,y) -> (h-1-y, x)
                dx = h - 1 - y;
                dy = x;
            } else { // 270
                // (x,y) -> (y, w-1-x)
                dx = y;
                dy = w - 1 - x;
            }
            int dst_idx = (dy * new_w + dx) * 4;
            memcpy(&dst[dst_idx], &pixels[src_idx], 4);
        }
    }

    pixels = std::move(dst);
    width = new_w;
    height = new_h;
}

// =====================================================================
// Video Decoder - extract RGBA frames using FFmpeg libraries
// =====================================================================

static bool decode_video_frames(const char *filename,
                                double start_time, double duration,
                                double fps,
                                int target_w, int target_h,
                                int max_w, int max_h,
                                std::vector<RGBAFrame> &frames) {
    AVFormatContext *fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, filename, nullptr, nullptr) < 0) {
        fprintf(stderr, "Error: cannot open '%s'\n", filename);
        return false;
    }
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        fprintf(stderr, "Error: cannot find stream info\n");
        avformat_close_input(&fmt_ctx);
        return false;
    }

    // Find best video stream
    int video_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = (int)i;
            break;
        }
    }
    if (video_idx < 0) {
        fprintf(stderr, "Error: no video stream found\n");
        avformat_close_input(&fmt_ctx);
        return false;
    }

    AVStream *stream = fmt_ctx->streams[video_idx];
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "Error: unsupported codec\n");
        avformat_close_input(&fmt_ctx);
        return false;
    }

    AVCodecContext *dec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec_ctx, stream->codecpar);
    if (avcodec_open2(dec_ctx, codec, nullptr) < 0) {
        fprintf(stderr, "Error: cannot open codec\n");
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }

    // Detect rotation from video metadata
    int rotation = detect_rotation(stream);
    if (rotation != 0)
        fprintf(stderr, "Detected video rotation: %d°\n", rotation);

    // Determine source dimensions
    // For 90°/270° rotation, swap dimensions so the scaler produces
    // output with the correct aspect ratio.
    int src_w = dec_ctx->width;
    int src_h = dec_ctx->height;
    if (rotation == 90 || rotation == 270)
        std::swap(src_w, src_h);

    // Compute output dimensions
    int out_w = target_w;
    int out_h = target_h;

    // Apply -maxw/-maxh: scale down only if source exceeds the max.
    // When neither -w nor -h is explicitly set:
    if (out_w <= 0 && out_h <= 0) {
        if (max_w > 0 && max_h > 0) {
            // Both max constraints: fit within bounding box
            if (src_w > max_w || src_h > max_h) {
                double scale_w = (double)max_w / src_w;
                double scale_h = (double)max_h / src_h;
                double scale = std::min(scale_w, scale_h);
                out_w = (int)round(src_w * scale);
                out_h = (int)round(src_h * scale);
            }
        } else if (max_w > 0 && src_w > max_w) {
            out_w = max_w;
        } else if (max_h > 0 && src_h > max_h) {
            out_h = max_h;
        }
    }

    if (out_w > 0 && out_h <= 0) {
        // Scale to width, auto height (match ffmpeg's scale=N:-1 truncation)
        out_h = (int)((int64_t)src_h * out_w / src_w);
    } else if (out_h > 0 && out_w <= 0) {
        // Scale to height, auto width
        out_w = (int)((int64_t)src_w * out_h / src_h);
    } else if (out_w <= 0 && out_h <= 0) {
        out_w = src_w;
        out_h = src_h;
    }
    // else both specified, use as-is

    // Make sure dimensions are at least 1
    if (out_w < 1) out_w = 1;
    if (out_h < 1) out_h = 1;

    // For 90°/270° rotation, the scaler must use the raw (un-swapped)
    // codec dimensions as source, and produce a pre-rotation output
    // that we then rotate after scaling. Swap out_w/out_h for the
    // scaler so it maps the raw frame correctly.
    int scale_src_w = dec_ctx->width;
    int scale_src_h = dec_ctx->height;
    int scale_dst_w = out_w;
    int scale_dst_h = out_h;
    if (rotation == 90 || rotation == 270)
        std::swap(scale_dst_w, scale_dst_h);

    // Set up scaler: use Lanczos when resizing, bilinear when 1:1.
    // SWS_ACCURATE_RND + full chroma interpolation for best YUV->RGB quality.
    int sws_algo = (scale_dst_w == scale_src_w && scale_dst_h == scale_src_h)
                   ? SWS_BILINEAR : SWS_LANCZOS;
    int sws_flags = sws_algo | SWS_ACCURATE_RND
                  | SWS_FULL_CHR_H_INT | SWS_FULL_CHR_H_INP;
    SwsContext *sws_ctx = sws_getContext(
        scale_src_w, scale_src_h, dec_ctx->pix_fmt,
        scale_dst_w, scale_dst_h, AV_PIX_FMT_RGBA,
        sws_flags, nullptr, nullptr, nullptr);
    if (!sws_ctx) {
        fprintf(stderr, "Error: cannot create scaler\n");
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }

    // Seek to start time
    if (start_time > 0.0) {
        int64_t ts = (int64_t)(start_time * AV_TIME_BASE);
        av_seek_frame(fmt_ctx, -1, ts, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(dec_ctx);
    }

    // Determine effective end time
    double vid_duration = 0.0;
    if (duration > 0.0) {
        vid_duration = duration;
    } else {
        // Use stream duration
        if (stream->duration != AV_NOPTS_VALUE) {
            vid_duration = stream->duration * av_q2d(stream->time_base) - start_time;
        } else if (fmt_ctx->duration != AV_NOPTS_VALUE) {
            vid_duration = (double)fmt_ctx->duration / AV_TIME_BASE - start_time;
        }
        if (vid_duration <= 0.0) vid_duration = 300.0; // fallback
    }

    double end_time = start_time + vid_duration;

    // ---------------------------------------------------------------
    // Streaming nearest-frame selection (matches ffmpeg's fps filter):
    //
    // For each output frame at time T = start + i/fps, we pick the
    // source frame with the closest PTS. We do this in a single pass
    // by keeping the current and previous decoded frame, and emitting
    // whichever is closer when the next decoded frame moves past T.
    // ---------------------------------------------------------------

    double frame_interval = 1.0 / fps;

    // Compute number of output frames (match ffmpeg's fps filter)
    int n_output = (int)floor(vid_duration * fps + 0.5);
    if (n_output < 1) n_output = 1;
    if (n_output > 15000) n_output = 15000; // safety cap

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!pkt || !frame) {
        fprintf(stderr, "Error: cannot allocate packet/frame\n");
        sws_freeContext(sws_ctx);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        if (pkt) av_packet_free(&pkt);
        if (frame) av_frame_free(&frame);
        return false;
    }

    // Allocate RGBA buffer
    uint8_t *rgba_data[4] = {nullptr};
    int rgba_linesize[4] = {0};
    if (av_image_alloc(rgba_data, rgba_linesize, scale_dst_w, scale_dst_h, AV_PIX_FMT_RGBA, 32) < 0) {
        fprintf(stderr, "Error: cannot allocate RGBA buffer\n");
        av_frame_free(&frame);
        av_packet_free(&pkt);
        sws_freeContext(sws_ctx);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }

    // State for nearest-frame selection
    struct CandidateFrame {
        double pts = -1e9;
        std::vector<uint8_t> pixels;
        bool valid = false;
    };

    CandidateFrame prev_frame, curr_frame;
    int output_idx = 0;  // next output frame to emit

    // Pre-rotation output dimensions (before rotating the pixels).
    // For 90°/270°, the scaler produces rotated dimensions that we
    // then rotate back to the expected orientation.
    int pre_rot_w = out_w, pre_rot_h = out_h;
    if (rotation == 90 || rotation == 270)
        std::swap(pre_rot_w, pre_rot_h);

    auto scale_and_store = [&](AVFrame *f, CandidateFrame &cf, double pts) {
        sws_scale(sws_ctx,
                  f->data, f->linesize,
                  0, dec_ctx->height,
                  rgba_data, rgba_linesize);
        cf.pts = pts;
        cf.pixels.resize((size_t)pre_rot_w * pre_rot_h * 4);
        for (int y = 0; y < pre_rot_h; y++) {
            memcpy(cf.pixels.data() + y * pre_rot_w * 4,
                   rgba_data[0] + y * rgba_linesize[0],
                   pre_rot_w * 4);
        }
        // Apply rotation to the RGBA pixels
        int rot_w = pre_rot_w, rot_h = pre_rot_h;
        rotate_rgba(cf.pixels, rot_w, rot_h, rotation);
        cf.valid = true;
    };

    auto emit_frame = [&](const CandidateFrame &cf) {
        RGBAFrame rf;
        rf.width = out_w;
        rf.height = out_h;
        rf.pixels = cf.pixels; // copy (may be reused for dup frames)
        frames.push_back(std::move(rf));
        output_idx++;
    };

    // Try to emit output frames when we have enough information.
    // For target time T, if curr.pts >= T, we compare prev vs curr
    // and emit whichever is closer.
    auto try_emit = [&]() {
        while (output_idx < n_output) {
            double target = start_time + output_idx * frame_interval;

            if (!curr_frame.valid) return;

            // If curr hasn't reached past our target yet, need more frames
            if (curr_frame.pts < target && curr_frame.pts < end_time - 0.001) return;

            // Pick nearest between prev and curr
            if (prev_frame.valid) {
                double dp = fabs(prev_frame.pts - target);
                double dc = fabs(curr_frame.pts - target);
                if (dp <= dc) {
                    emit_frame(prev_frame);
                } else {
                    emit_frame(curr_frame);
                }
            } else {
                emit_frame(curr_frame);
            }
        }
    };

    auto process_decoded_frame = [&](double pts) {
        if (pts < start_time - 1.0 / fps) return; // before our window

        // Shift: prev <- curr, curr <- new
        prev_frame = std::move(curr_frame);
        scale_and_store(frame, curr_frame, pts);

        try_emit();
    };

    // Decode loop
    bool done = false;
    while (!done && output_idx < n_output && av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != video_idx) {
            av_packet_unref(pkt);
            continue;
        }

        if (avcodec_send_packet(dec_ctx, pkt) < 0) {
            av_packet_unref(pkt);
            continue;
        }

        while (avcodec_receive_frame(dec_ctx, frame) == 0) {
            double pts = 0.0;
            if (frame->pts != AV_NOPTS_VALUE)
                pts = frame->pts * av_q2d(stream->time_base);
            else if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
                pts = frame->best_effort_timestamp * av_q2d(stream->time_base);

            process_decoded_frame(pts);

            // Stop if we've emitted all frames and are well past end
            if (output_idx >= n_output) { done = true; break; }
            if (pts > end_time + 1.0) { done = true; break; }
        }
        av_packet_unref(pkt);
    }

    // Flush decoder
    if (!done && output_idx < n_output) {
        avcodec_send_packet(dec_ctx, nullptr);
        while (avcodec_receive_frame(dec_ctx, frame) == 0) {
            double pts = 0.0;
            if (frame->pts != AV_NOPTS_VALUE)
                pts = frame->pts * av_q2d(stream->time_base);
            else if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
                pts = frame->best_effort_timestamp * av_q2d(stream->time_base);

            process_decoded_frame(pts);
            if (output_idx >= n_output) break;
        }
    }

    // Emit remaining output frames from the last available frame
    while (output_idx < n_output && curr_frame.valid) {
        emit_frame(curr_frame);
    }

    av_freep(&rgba_data[0]);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);

    return !frames.empty();
}

// =====================================================================
// Wu's Optimal Color Quantization
// =====================================================================
// Based on Xiaolin Wu's "Greedy orthogonal bipartition of RGB space
// for color quantization" (Graphics Gems vol. II, 1991).
//
// Uses a 3D histogram (33^3) with cumulative moments to efficiently
// find optimal splits that minimize total variance.

static constexpr int HIST_SIZE = 33; // indices 0..32, 0 is unused
static constexpr int MAXCOLOR = 256;

class WuQuantizer {
    // 3D cumulative moments
    long wt[HIST_SIZE][HIST_SIZE][HIST_SIZE]; // weight (pixel count)
    long mr[HIST_SIZE][HIST_SIZE][HIST_SIZE]; // sum of red
    long mg[HIST_SIZE][HIST_SIZE][HIST_SIZE]; // sum of green
    long mb[HIST_SIZE][HIST_SIZE][HIST_SIZE]; // sum of blue
    float m2[HIST_SIZE][HIST_SIZE][HIST_SIZE]; // sum of squared distances

public:
    WuQuantizer() {
        memset(wt, 0, sizeof(wt));
        memset(mr, 0, sizeof(mr));
        memset(mg, 0, sizeof(mg));
        memset(mb, 0, sizeof(mb));
        memset(m2, 0, sizeof(m2));
    }

    struct Box {
        int r0, r1, g0, g1, b0, b1;
        int vol;
    };

    // Accumulate 3D histogram from RGBA pixels (skip transparent).
    // Can be called multiple times to combine pixels from multiple frames.
    void build_histogram(const uint8_t *rgba, int npixels) {
        for (int i = 0; i < npixels; i++) {
            int a = rgba[i*4+3];
            if (a < 128) continue; // skip transparent
            int r = rgba[i*4+0];
            int g = rgba[i*4+1];
            int b = rgba[i*4+2];
            int ri = (r >> 3) + 1;
            int gi = (g >> 3) + 1;
            int bi = (b >> 3) + 1;
            wt[ri][gi][bi]++;
            mr[ri][gi][bi] += r;
            mg[ri][gi][bi] += g;
            mb[ri][gi][bi] += b;
            m2[ri][gi][bi] += (float)(r*r + g*g + b*b);
        }
    }

    // Compute cumulative moments using 3D prefix sums
    void compute_moments() {
        for (int r = 1; r < HIST_SIZE; r++) {
            long area_w[HIST_SIZE] = {};
            long area_r[HIST_SIZE] = {};
            long area_g[HIST_SIZE] = {};
            long area_b[HIST_SIZE] = {};
            float area_2[HIST_SIZE] = {};

            for (int g = 1; g < HIST_SIZE; g++) {
                long line_w = 0, line_r = 0, line_g2 = 0, line_b = 0;
                float line_2 = 0;
                for (int b = 1; b < HIST_SIZE; b++) {
                    line_w += wt[r][g][b];
                    line_r += mr[r][g][b];
                    line_g2 += mg[r][g][b];
                    line_b += mb[r][g][b];
                    line_2 += m2[r][g][b];

                    area_w[b] += line_w;
                    area_r[b] += line_r;
                    area_g[b] += line_g2;
                    area_b[b] += line_b;
                    area_2[b] += line_2;

                    wt[r][g][b] = wt[r-1][g][b] + area_w[b];
                    mr[r][g][b] = mr[r-1][g][b] + area_r[b];
                    mg[r][g][b] = mg[r-1][g][b] + area_g[b];
                    mb[r][g][b] = mb[r-1][g][b] + area_b[b];
                    m2[r][g][b] = m2[r-1][g][b] + area_2[b];
                }
            }
        }
    }

    // Volume of a box (using inclusion-exclusion on cumulative array)
    static long vol(const Box &b, const long d[HIST_SIZE][HIST_SIZE][HIST_SIZE]) {
        return d[b.r1][b.g1][b.b1]
              -d[b.r1][b.g1][b.b0]
              -d[b.r1][b.g0][b.b1]
              +d[b.r1][b.g0][b.b0]
              -d[b.r0][b.g1][b.b1]
              +d[b.r0][b.g1][b.b0]
              +d[b.r0][b.g0][b.b1]
              -d[b.r0][b.g0][b.b0];
    }

    static float vol_f(const Box &b, const float d[HIST_SIZE][HIST_SIZE][HIST_SIZE]) {
        return d[b.r1][b.g1][b.b1]
              -d[b.r1][b.g1][b.b0]
              -d[b.r1][b.g0][b.b1]
              +d[b.r1][b.g0][b.b0]
              -d[b.r0][b.g1][b.b1]
              +d[b.r0][b.g1][b.b0]
              +d[b.r0][b.g0][b.b1]
              -d[b.r0][b.g0][b.b0];
    }

    // Bottom slice along an axis for the maximize function
    static long bottom(const Box &b, int dir,
                       const long d[HIST_SIZE][HIST_SIZE][HIST_SIZE]) {
        switch (dir) {
        case 0: // red
            return -d[b.r0][b.g1][b.b1]
                   +d[b.r0][b.g1][b.b0]
                   +d[b.r0][b.g0][b.b1]
                   -d[b.r0][b.g0][b.b0];
        case 1: // green
            return -d[b.r1][b.g0][b.b1]
                   +d[b.r1][b.g0][b.b0]
                   +d[b.r0][b.g0][b.b1]
                   -d[b.r0][b.g0][b.b0];
        case 2: // blue
            return -d[b.r1][b.g1][b.b0]
                   +d[b.r1][b.g0][b.b0]
                   +d[b.r0][b.g1][b.b0]
                   -d[b.r0][b.g0][b.b0];
        }
        return 0;
    }

    // Top slice for a given cut position
    static long top(const Box &b, int dir, int pos,
                    const long d[HIST_SIZE][HIST_SIZE][HIST_SIZE]) {
        switch (dir) {
        case 0:
            return d[pos][b.g1][b.b1]
                  -d[pos][b.g1][b.b0]
                  -d[pos][b.g0][b.b1]
                  +d[pos][b.g0][b.b0];
        case 1:
            return d[b.r1][pos][b.b1]
                  -d[b.r1][pos][b.b0]
                  -d[b.r0][pos][b.b1]
                  +d[b.r0][pos][b.b0];
        case 2:
            return d[b.r1][b.g1][pos]
                  -d[b.r1][b.g0][pos]
                  -d[b.r0][b.g1][pos]
                  +d[b.r0][b.g0][pos];
        }
        return 0;
    }

    // Variance of a box
    float variance(const Box &b) {
        float dr = (float)vol(b, mr);
        float dg = (float)vol(b, mg);
        float db = (float)vol(b, mb);
        float xx = vol_f(b, m2);
        float w  = (float)vol(b, wt);
        if (w <= 0) return 0.0f;
        return xx - (dr*dr + dg*dg + db*db) / w;
    }

    // Find the best cut along a given axis that maximizes gain
    float maximize(const Box &b, int dir, int first, int last,
                   int &cut, long whole_w, long whole_r,
                   long whole_g, long whole_b) {
        long base_r = bottom(b, dir, mr);
        long base_g = bottom(b, dir, mg);
        long base_b = bottom(b, dir, mb);
        long base_w = bottom(b, dir, wt);

        float max_gain = 0.0f;
        cut = -1;

        for (int i = first; i < last; i++) {
            long half_r = base_r + top(b, dir, i, mr);
            long half_g = base_g + top(b, dir, i, mg);
            long half_b = base_b + top(b, dir, i, mb);
            long half_w = base_w + top(b, dir, i, wt);

            if (half_w == 0) continue;

            float temp = ((float)half_r * half_r +
                          (float)half_g * half_g +
                          (float)half_b * half_b) / (float)half_w;

            long other_w = whole_w - half_w;
            if (other_w == 0) continue;

            long other_r = whole_r - half_r;
            long other_g = whole_g - half_g;
            long other_b = whole_b - half_b;

            temp += ((float)other_r * other_r +
                     (float)other_g * other_g +
                     (float)other_b * other_b) / (float)other_w;

            if (temp > max_gain) {
                max_gain = temp;
                cut = i;
            }
        }
        return max_gain;
    }

    // Try to split a box into two. Returns true if successful.
    bool cut_box(Box &b1, Box &b2) {
        long whole_r = vol(b1, mr);
        long whole_g = vol(b1, mg);
        long whole_b = vol(b1, mb);
        long whole_w = vol(b1, wt);

        if (whole_w == 0) return false;

        int cutr = -1, cutg = -1, cutb = -1;
        float maxr = maximize(b1, 0, b1.r0 + 1, b1.r1, cutr,
                              whole_w, whole_r, whole_g, whole_b);
        float maxg = maximize(b1, 1, b1.g0 + 1, b1.g1, cutg,
                              whole_w, whole_r, whole_g, whole_b);
        float maxb = maximize(b1, 2, b1.b0 + 1, b1.b1, cutb,
                              whole_w, whole_r, whole_g, whole_b);

        int dir;
        if (maxr >= maxg && maxr >= maxb) {
            dir = 0;
            if (cutr < 0) return false;
        } else if (maxg >= maxr && maxg >= maxb) {
            dir = 1;
            if (cutg < 0) return false;
        } else {
            dir = 2;
            if (cutb < 0) return false;
        }

        b2 = b1;
        switch (dir) {
        case 0: b2.r0 = cutr; b1.r1 = cutr; break;
        case 1: b2.g0 = cutg; b1.g1 = cutg; break;
        case 2: b2.b0 = cutb; b1.b1 = cutb; break;
        }
        b1.vol = (b1.r1-b1.r0) * (b1.g1-b1.g0) * (b1.b1-b1.b0);
        b2.vol = (b2.r1-b2.r0) * (b2.g1-b2.g0) * (b2.b1-b2.b0);
        return true;
    }

    // Extract palette from pre-computed moments (after calling
    // build_histogram() one or more times, then compute_moments())
    int quantize_from_moments(int max_colors, uint8_t palette[][3]) {
        Box boxes[MAXCOLOR];
        float vv[MAXCOLOR];
        int nboxes = 1;

        boxes[0] = {0, HIST_SIZE-1, 0, HIST_SIZE-1, 0, HIST_SIZE-1, 0};
        vv[0] = variance(boxes[0]);

        for (int i = 1; i < max_colors; i++) {
            int best = -1;
            float best_var = -1;
            for (int j = 0; j < nboxes; j++) {
                if (vv[j] > best_var) {
                    best_var = vv[j];
                    best = j;
                }
            }
            if (best < 0 || best_var <= 0.0f) break;

            Box new_box;
            if (!cut_box(boxes[best], new_box)) {
                vv[best] = 0.0f;
                i--;
                continue;
            }

            vv[best] = variance(boxes[best]);
            boxes[nboxes] = new_box;
            vv[nboxes] = variance(new_box);
            nboxes++;
        }

        for (int i = 0; i < nboxes; i++) {
            long w = vol(boxes[i], wt);
            if (w > 0) {
                palette[i][0] = (uint8_t)(vol(boxes[i], mr) / w);
                palette[i][1] = (uint8_t)(vol(boxes[i], mg) / w);
                palette[i][2] = (uint8_t)(vol(boxes[i], mb) / w);
            } else {
                palette[i][0] = palette[i][1] = palette[i][2] = 0;
            }
        }
        return nboxes;
    }
};

// =====================================================================
// Nearest color lookup with cache
// =====================================================================

static int find_nearest(const uint8_t palette[][3], int count,
                        int r, int g, int b,
                        std::vector<int16_t> &cache) {
    int key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
    if (cache[key] >= 0) return cache[key];

    int best = 0, best_d = INT_MAX;
    for (int i = 0; i < count; i++) {
        int dr = r - palette[i][0];
        int dg = g - palette[i][1];
        int db = b - palette[i][2];
        // Perceptual weighting: green > red > blue
        int d = 2*dr*dr + 4*dg*dg + 3*db*db;
        if (d < best_d) { best_d = d; best = i; }
    }
    cache[key] = (int16_t)best;
    return best;
}



// =====================================================================
// LZW Encoder
// =====================================================================

class LZWEncoder {
    FILE *fp_;
    uint32_t bit_buf_;
    int bits_in_;
    uint8_t block_[255];
    int block_len_;

    // Hash table (open addressing, double hashing)
    static constexpr int HTAB_SIZE = 5003;
    int htab_key_[HTAB_SIZE];
    int htab_val_[HTAB_SIZE];

    void clear_htab() { memset(htab_key_, 0xFF, sizeof(htab_key_)); }

    int lookup(int prefix, int suffix) {
        int key = (prefix << 8) | suffix;
        int h = (unsigned)key % HTAB_SIZE;
        int step = 1 + ((unsigned)key % (HTAB_SIZE - 2));
        while (htab_key_[h] != -1) {
            if (htab_key_[h] == key) return htab_val_[h];
            h = (h + step) % HTAB_SIZE;
        }
        return -1;
    }

    void insert(int prefix, int suffix, int code) {
        int key = (prefix << 8) | suffix;
        int h = (unsigned)key % HTAB_SIZE;
        int step = 1 + ((unsigned)key % (HTAB_SIZE - 2));
        while (htab_key_[h] != -1)
            h = (h + step) % HTAB_SIZE;
        htab_key_[h] = key;
        htab_val_[h] = code;
    }

    void emit(int code, int code_size) {
        bit_buf_ |= ((uint32_t)code << bits_in_);
        bits_in_ += code_size;
        while (bits_in_ >= 8) {
            block_[block_len_++] = bit_buf_ & 0xFF;
            bit_buf_ >>= 8;
            bits_in_ -= 8;
            if (block_len_ == 255) flush_block();
        }
    }

    void flush_block() {
        if (block_len_ > 0) {
            fputc(block_len_, fp_);
            fwrite(block_, 1, block_len_, fp_);
            block_len_ = 0;
        }
    }

public:
    void encode(FILE *fp, const uint8_t *data, int count, int palette_bits) {
        fp_ = fp;
        bit_buf_ = 0;
        bits_in_ = 0;
        block_len_ = 0;

        int min_code = std::max(palette_bits, 2);
        int clear_code = 1 << min_code;
        int eoi_code = clear_code + 1;
        int code_size = min_code + 1;
        int next_code = eoi_code + 1;

        fputc(min_code, fp_);

        clear_htab();
        emit(clear_code, code_size);

        if (count == 0) {
            emit(eoi_code, code_size);
            if (bits_in_ > 0)
                block_[block_len_++] = bit_buf_ & 0xFF;
            flush_block();
            fputc(0, fp_);
            return;
        }

        int prefix = data[0];
        for (int i = 1; i < count; i++) {
            int suffix = data[i];
            int combined = lookup(prefix, suffix);
            if (combined != -1) {
                prefix = combined;
            } else {
                emit(prefix, code_size);
                if (next_code < 4096) {
                    insert(prefix, suffix, next_code);
                    next_code++;
                    if (next_code > (1 << code_size) && code_size < 12)
                        code_size++;
                } else {
                    emit(clear_code, code_size);
                    clear_htab();
                    code_size = min_code + 1;
                    next_code = eoi_code + 1;
                }
                prefix = suffix;
            }
        }

        emit(prefix, code_size);
        emit(eoi_code, code_size);

        if (bits_in_ > 0)
            block_[block_len_++] = bit_buf_ & 0xFF;
        flush_block();
        fputc(0, fp_); // block terminator
    }
};

// =====================================================================
// GIF Writer
// =====================================================================

static void write_u16le(FILE *fp, uint16_t v) {
    fputc(v & 0xFF, fp);
    fputc((v >> 8) & 0xFF, fp);
}

static int palette_bits_for(int count) {
    int bits = 1;
    while ((1 << bits) < count) bits++;
    return bits;
}

static bool write_gif(const char *output,
                      std::vector<RGBAFrame> &frames,
                      const std::vector<int> &delays_cs,
                      int loop_count,
                      int fuzz_threshold = 0) {
    if (frames.empty()) {
        fprintf(stderr, "Error: no frames to write\n");
        return false;
    }

    int gw = frames[0].width, gh = frames[0].height;

    // ---------------------------------------------------------
    // Delta frame optimization: for frames after the first,
    // mark unchanged pixels as transparent and crop to the
    // bounding box of changed pixels.
    // ---------------------------------------------------------

    struct PreparedFrame {
        std::vector<uint8_t> rgba;
        int left, top, fw, fh;
        bool is_delta;
    };

    size_t nframes = frames.size();
    std::vector<PreparedFrame> prepared(nframes);

    // Frame 0: always full canvas
    prepared[0].rgba = frames[0].pixels;
    prepared[0].left = 0;
    prepared[0].top  = 0;
    prepared[0].fw   = gw;
    prepared[0].fh   = gh;
    prepared[0].is_delta = false;

    // Composed canvas: tracks what the GIF viewer will actually display
    // after compositing each delta frame. When fuzz > 0, pixels deemed
    // "close enough" are made transparent and the viewer keeps whatever
    // was there before. If we compare against the raw original frames,
    // the small errors compound over time causing visible drift.
    // By comparing against the canvas (viewer's ground truth), each
    // frame's fuzz decision is accurate - no accumulation.
    size_t canvas_size = (size_t)gw * gh * 4;
    std::vector<uint8_t> canvas(canvas_size);
    memcpy(canvas.data(), frames[0].pixels.data(), canvas_size);

    int fuzz = fuzz_threshold;

    for (size_t fi = 1; fi < nframes; fi++) {
        const uint8_t *curr = frames[fi].pixels.data();
        const uint8_t *prev = canvas.data();  // what the viewer sees

        // Pixel comparison lambda - exact or lossy (fuzz threshold)
        auto pixels_same = [&](int idx) -> bool {
            if (curr[idx+3] < 128 && prev[idx+3] < 128) return true; // both transparent
            if (curr[idx+3] < 128 || prev[idx+3] < 128) return false; // one transparent
            if (fuzz == 0) {
                return curr[idx]==prev[idx] && curr[idx+1]==prev[idx+1] &&
                       curr[idx+2]==prev[idx+2];
            }
            return abs(curr[idx]-prev[idx]) <= fuzz &&
                   abs(curr[idx+1]-prev[idx+1]) <= fuzz &&
                   abs(curr[idx+2]-prev[idx+2]) <= fuzz;
        };

        // Check if any pixel goes opaque -> transparent (prevents delta)
        bool can_delta = true;
        for (int p = 0; p < gw * gh; p++) {
            if (prev[p * 4 + 3] >= 128 && curr[p * 4 + 3] < 128) {
                can_delta = false;
                break;
            }
        }

        if (!can_delta) {
            prepared[fi].rgba.assign(curr, curr + (size_t)gw * gh * 4);
            prepared[fi].left = 0;
            prepared[fi].top  = 0;
            prepared[fi].fw   = gw;
            prepared[fi].fh   = gh;
            prepared[fi].is_delta = false;
            // Full frame replaces everything on canvas
            memcpy(canvas.data(), curr, canvas_size);
            continue;
        }

        // Find bounding box of changed pixels
        int min_x = gw, min_y = gh, max_x = -1, max_y = -1;
        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < gw; x++) {
                int idx = (y * gw + x) * 4;
                if (!pixels_same(idx)) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }

        if (max_x < 0) {
            // Frames are identical - emit a 1x1 transparent pixel
            prepared[fi].rgba = {0, 0, 0, 0};
            prepared[fi].left = 0;
            prepared[fi].top  = 0;
            prepared[fi].fw   = 1;
            prepared[fi].fh   = 1;
            prepared[fi].is_delta = true;
            // Canvas stays unchanged (viewer sees same thing)
            continue;
        }

        int sw = max_x - min_x + 1;
        int sh = max_y - min_y + 1;

        // Build sub-frame: copy changed pixels, mark unchanged as transparent
        std::vector<uint8_t> sub((size_t)sw * sh * 4);
        for (int y = 0; y < sh; y++) {
            for (int x = 0; x < sw; x++) {
                int src = ((min_y + y) * gw + (min_x + x)) * 4;
                int dst = (y * sw + x) * 4;
                if (pixels_same(src)) {
                    sub[dst] = sub[dst+1] = sub[dst+2] = sub[dst+3] = 0;
                    // Canvas pixel stays as-is (viewer keeps old value)
                } else {
                    sub[dst]   = curr[src];
                    sub[dst+1] = curr[src+1];
                    sub[dst+2] = curr[src+2];
                    sub[dst+3] = curr[src+3];
                    // Update canvas to reflect what the viewer will see
                    canvas[src]   = curr[src];
                    canvas[src+1] = curr[src+1];
                    canvas[src+2] = curr[src+2];
                    canvas[src+3] = curr[src+3];
                }
            }
        }

        prepared[fi].rgba = std::move(sub);
        prepared[fi].left = min_x;
        prepared[fi].top  = min_y;
        prepared[fi].fw   = sw;
        prepared[fi].fh   = sh;
        prepared[fi].is_delta = true;
    }

    // Free original frames
    frames.clear();
    frames.shrink_to_fit();

    // ---------------------------------------------------------
    // Build global palette from all frames
    // ---------------------------------------------------------

    bool any_alpha = false;
    for (size_t fi = 0; fi < nframes; fi++) {
        const uint8_t *rgba = prepared[fi].rgba.data();
        int npx = prepared[fi].fw * prepared[fi].fh;
        for (int i = 0; i < npx && !any_alpha; i++)
            if (rgba[i*4+3] < 128) any_alpha = true;
    }

    int global_max_colors = any_alpha ? 255 : 256;
    uint8_t global_palette[256][3];
    int global_pal_count;
    {
        WuQuantizer wu;
        for (size_t fi = 0; fi < nframes; fi++) {
            wu.build_histogram(prepared[fi].rgba.data(),
                               prepared[fi].fw * prepared[fi].fh);
        }
        wu.compute_moments();
        global_pal_count = wu.quantize_from_moments(global_max_colors,
                                                     global_palette);
    }

    // Add transparency slot to global palette if needed
    int global_trans_idx = -1;
    if (any_alpha) {
        global_trans_idx = global_pal_count;
        global_palette[global_pal_count][0] = 0;
        global_palette[global_pal_count][1] = 0;
        global_palette[global_pal_count][2] = 0;
        global_pal_count++;
    }

    int global_pbits = palette_bits_for(global_pal_count);
    int global_pal_size = 1 << global_pbits;

    // ---------------------------------------------------------
    // Write GIF
    // ---------------------------------------------------------

    FILE *fp = fopen(output, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot create '%s'\n", output);
        return false;
    }

    // GIF89a Header
    fwrite("GIF89a", 1, 6, fp);

    // Logical Screen Descriptor with Global Color Table
    write_u16le(fp, gw);
    write_u16le(fp, gh);
    uint8_t gct_packed = 0x80              // GCT flag
                       | ((global_pbits - 1) << 4)  // color resolution
                       | (global_pbits - 1);        // GCT size
    fputc(gct_packed, fp);
    fputc(0x00, fp); // background color index
    fputc(0x00, fp); // pixel aspect ratio

    // Global Color Table
    for (int i = 0; i < global_pal_size; i++) {
        if (i < global_pal_count) {
            fputc(global_palette[i][0], fp);
            fputc(global_palette[i][1], fp);
            fputc(global_palette[i][2], fp);
        } else {
            fputc(0, fp); fputc(0, fp); fputc(0, fp);
        }
    }

    // Netscape Application Extension (looping)
    if (loop_count != 1) {
        fputc(0x21, fp);
        fputc(0xFF, fp);
        fputc(11, fp);
        fwrite("NETSCAPE2.0", 1, 11, fp);
        fputc(3, fp);
        fputc(1, fp);
        uint16_t ns_count = (loop_count == 0) ? 0 : (uint16_t)(loop_count - 1);
        write_u16le(fp, ns_count);
        fputc(0, fp);
    }

    // Color lookup cache (shared across frames since palette is the same)
    std::vector<int16_t> cache(32768, -1);
    int opaque_count = global_pal_count - (any_alpha ? 1 : 0);

    LZWEncoder lzw;



    for (size_t fi = 0; fi < nframes; fi++) {
        auto &pf = prepared[fi];

        // Dither frame against global palette
        const uint8_t *rgba = pf.rgba.data();
        int fw = pf.fw, fh = pf.fh;
        std::vector<uint8_t> indexed(fw * fh);

        // Floyd-Steinberg dithering
        std::vector<int> err_r0(fw + 2, 0), err_g0(fw + 2, 0), err_b0(fw + 2, 0);
        std::vector<int> err_r1(fw + 2, 0), err_g1(fw + 2, 0), err_b1(fw + 2, 0);

        for (int y = 0; y < fh; y++) {
            std::fill(err_r1.begin(), err_r1.end(), 0);
            std::fill(err_g1.begin(), err_g1.end(), 0);
            std::fill(err_b1.begin(), err_b1.end(), 0);

            for (int x = 0; x < fw; x++) {
                int idx = y * fw + x;
                if (rgba[idx*4+3] < 128) {
                    indexed[idx] = global_trans_idx;
                    continue;
                }

                int r = std::clamp((int)rgba[idx*4+0] + err_r0[x+1], 0, 255);
                int g = std::clamp((int)rgba[idx*4+1] + err_g0[x+1], 0, 255);
                int b = std::clamp((int)rgba[idx*4+2] + err_b0[x+1], 0, 255);

                int ci = find_nearest(global_palette, opaque_count, r, g, b, cache);
                indexed[idx] = ci;

                int er = r - global_palette[ci][0];
                int eg = g - global_palette[ci][1];
                int eb = b - global_palette[ci][2];

                err_r0[x+2] += er * 7 / 16;
                err_g0[x+2] += eg * 7 / 16;
                err_b0[x+2] += eb * 7 / 16;
                err_r1[x]   += er * 3 / 16;
                err_g1[x]   += eg * 3 / 16;
                err_b1[x]   += eb * 3 / 16;
                err_r1[x+1] += er * 5 / 16;
                err_g1[x+1] += eg * 5 / 16;
                err_b1[x+1] += eb * 5 / 16;
                err_r1[x+2] += er * 1 / 16;
                err_g1[x+2] += eg * 1 / 16;
                err_b1[x+2] += eb * 1 / 16;
            }
            std::swap(err_r0, err_r1);
            std::swap(err_g0, err_g1);
            std::swap(err_b0, err_b1);
        }

        // Determine disposal method
        uint8_t disposal;
        if (fi + 1 < nframes && prepared[fi + 1].is_delta) {
            disposal = 1; // do not dispose
        } else {
            disposal = 2; // restore to background
        }

        // Graphic Control Extension
        fputc(0x21, fp);
        fputc(0xF9, fp);
        fputc(4, fp);
        uint8_t gce_packed = (disposal << 2);
        if (global_trans_idx >= 0) gce_packed |= 0x01;
        fputc(gce_packed, fp);
        int frame_delay = (fi < delays_cs.size()) ? delays_cs[fi] : 0;
        write_u16le(fp, (uint16_t)frame_delay);
        fputc(global_trans_idx >= 0 ? global_trans_idx : 0, fp);
        fputc(0, fp);

        // Image Descriptor (no local color table - using GCT)
        fputc(0x2C, fp);
        write_u16le(fp, (uint16_t)pf.left);
        write_u16le(fp, (uint16_t)pf.top);
        write_u16le(fp, (uint16_t)pf.fw);
        write_u16le(fp, (uint16_t)pf.fh);
        fputc(0x00, fp); // no local color table

        // LZW image data
        lzw.encode(fp, indexed.data(), fw * fh, global_pbits);

        // Free frame data early
        pf.rgba.clear();
        pf.rgba.shrink_to_fit();
    }

    fputc(0x3B, fp); // GIF Trailer
    fclose(fp);
    return true;
}

// =====================================================================
// Main
// =====================================================================

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -i input -o output.gif [options]\n\n"
        "Options:\n"
        "  -i FILE    Input video file (required)\n"
        "  -o FILE    Output GIF file (required)\n"
        "  -r FPS     Frame rate (default: 10)\n"
        "  -ss TIME   Start time in seconds (default: 0)\n"
        "  -t TIME    Duration in seconds (default: full video)\n"
        "  -w WIDTH   Scale to exact width (default: original)\n"
        "  -h HEIGHT  Scale to exact height (default: original)\n"
        "  -maxw N    Scale down to N width only if source is wider\n"
        "  -maxh N    Scale down to N height only if source is taller\n"
        "  -loop N    Loop count: 0=infinite (default), 1=once, N=N times\n"
        "  -fuzz N    Lossy delta threshold (0-255, default: 2)\n\n"
        "Examples:\n"
        "  %s -i video.mp4 -o out.gif -r 15 -ss 2.5 -t 5 -w 480\n"
        "  %s -i clip.webm -o out.gif -r 10 -maxw 800 -fuzz 2\n",
        prog, prog, prog);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }

    std::string input_file;
    std::string output_file;
    double fps = 10.0;
    double start_time = 0.0;
    double duration = -1.0; // -1 = full video
    int target_w = -1;
    int target_h = -1;
    int max_w = -1;
    int max_h = -1;
    int loop_count = 0; // 0 = infinite
    int fuzz = 2;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            fps = atof(argv[++i]);
            if (fps <= 0) fps = 10.0;
        } else if (strcmp(argv[i], "-ss") == 0 && i + 1 < argc) {
            start_time = atof(argv[++i]);
            if (start_time < 0) start_time = 0;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            duration = atof(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            target_w = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            target_h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-maxw") == 0 && i + 1 < argc) {
            max_w = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-maxh") == 0 && i + 1 < argc) {
            max_h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-loop") == 0 && i + 1 < argc) {
            loop_count = atoi(argv[++i]);
            if (loop_count < 0) loop_count = 0;
        } else if (strcmp(argv[i], "-fuzz") == 0 && i + 1 < argc) {
            fuzz = atoi(argv[++i]);
            if (fuzz < 0) fuzz = 0;
            if (fuzz > 255) fuzz = 255;
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (input_file.empty()) {
        fprintf(stderr, "Error: no input file specified (use -i)\n");
        return 1;
    }
    if (output_file.empty()) {
        fprintf(stderr, "Error: no output file specified (use -o)\n");
        return 1;
    }

    // Decode video frames
    std::vector<RGBAFrame> frames;
    fprintf(stderr, "Decoding video: %s (fps=%.1f, start=%.2f",
            input_file.c_str(), fps, start_time);
    if (duration > 0)
        fprintf(stderr, ", duration=%.2f", duration);
    if (target_w > 0)
        fprintf(stderr, ", w=%d", target_w);
    if (target_h > 0)
        fprintf(stderr, ", h=%d", target_h);
    fprintf(stderr, ")...\n");

    if (!decode_video_frames(input_file.c_str(), start_time, duration,
                             fps, target_w, target_h, max_w, max_h, frames)) {
        fprintf(stderr, "Error: failed to decode video frames\n");
        return 1;
    }

    fprintf(stderr, "Decoded %zu frames (%dx%d)\n",
            frames.size(), frames[0].width, frames[0].height);

    // Compute per-frame delays in centiseconds.
    // GIF delays are in centiseconds (1/100s). For many FPS values the
    // exact frame interval can't be represented as an integer number of
    // centiseconds. Instead of rounding every frame to the same value
    // (which drifts the total duration), we distribute the remainder
    // across frames: e.g. at 7fps the ideal delay is 14.286cs, so we
    // alternate 14 and 15 to keep the running total accurate.
    size_t n = frames.size();
    std::vector<int> delays_cs(n);
    {
        int assigned = 0;
        for (size_t i = 0; i < n; i++) {
            // Target cumulative centiseconds at end of frame i+1
            int target = (int)round((double)(i + 1) * 100.0 / fps);
            int d = target - assigned;
            if (d < 0) d = 0;
            if (d > 65535) d = 65535;
            delays_cs[i] = d;
            assigned += d;
        }
    }

    // Save count before write_gif (which clears the vector)
    size_t frame_count = frames.size();

    // Write GIF
    if (!write_gif(output_file.c_str(), frames, delays_cs, loop_count, fuzz)) {
        fprintf(stderr, "Error: failed to write GIF\n");
        return 1;
    }

    fprintf(stderr, "Created %s (%zu frames)\n",
            output_file.c_str(), frame_count);
    return 0;
}
