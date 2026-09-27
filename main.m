/*
 * High-Speed USB Bulk Video Receiver (VRX) + Keyboard Plane Control +
 * ArduPilot CRSF Telemetry, for macOS.
 */

#define _DARWIN_C_SOURCE

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreGraphics/CoreGraphics.h>
#include <Metal/Metal.h>
#include <simd/simd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/ioctl.h>
#include <libusb-1.0/libusb.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------- Video (vendor bulk) config ---------------- */
#define TARGET_VID      0x2207
#define TARGET_PID      0x0011
#define BUFFER_SIZE     (64 * 1024)

/* ---------------- ArduPilot CRSF link (USB-CDC) config ------ */
#define ARDUPILOT_SERIAL_DEV_DEFAULT "/dev/cu.usbmodem14201" /* override via argv[1]; Linux equivalent is usually /dev/ttyACM0 */
#define ARDUPILOT_BAUD        B115200
#define CRSF_TX_HZ            50     /* how often we send an RC_CHANNELS_PACKED frame */
#define SERIAL_RETRY_MS        2000  /* how often to retry opening the port if ArduPilot isn't there yet */

#define CRSF_SYNC_FC                    0xC8  /* address we send OUR frames to (Flight Controller) */
#define CRSF_MAX_FRAME_LEN               64
#define CRSF_FRAMETYPE_GPS               0x02
#define CRSF_FRAMETYPE_BATTERY_SENSOR    0x08
#define CRSF_FRAMETYPE_LINK_STATISTICS   0x14
#define CRSF_FRAMETYPE_RC_CHANNELS       0x16
#define CRSF_FRAMETYPE_ATTITUDE          0x1E
#define CRSF_FRAMETYPE_FLIGHT_MODE       0x21

/* ---------------- Keyboard control mapping ------------------- */
#define KEY_ROLL_LEFT   'a'
#define KEY_ROLL_RIGHT  'd'
#define KEY_PITCH_DOWN  'w'
#define KEY_PITCH_UP    's'
#define KEY_SW1         '1'
#define KEY_SW2         '2'
#define KEY_SW3         '3'
#define KEY_CENTER      'c'
#define KEY_QUIT        'q'
/* Throttle = Up/Down arrows, Yaw = Left/Right arrows (handled as ANSI escape sequences) */

#define AXIS_STEP       0.15f   /* roll/pitch/yaw nudge per keypress, -1..+1 scale */
#define THROTTLE_STEP   0.03f   /* throttle nudge per keypress, 0..1 scale */
#define DECAY_TAU_SEC   0.35f   /* roll/pitch/yaw spring-back time constant */



/* Embedded Metal Shading Language (MSL) Source */
static const char *kMetalShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct VertexOut {\n"
"    float4 position [[position]];\n"
"    float2 texCoords;\n"
"};\n"
"\n"
"// Fullscreen quad generator (no vertex buffer needed)\n"
"vertex VertexOut yuvVertexShader(uint vertexID [[vertex_id]]) {\n"
"    float4 pos[4] = {\n"
"        float4(-1.0, -1.0, 0.0, 1.0),\n"
"        float4( 1.0, -1.0, 0.0, 1.0),\n"
"        float4(-1.0,  1.0, 0.0, 1.0),\n"
"        float4( 1.0,  1.0, 0.0, 1.0)\n"
"    };\n"
"    float2 tex[4] = {\n"
"        float2(0.0, 1.0),\n"
"        float2(1.0, 1.0),\n"
"        float2(0.0, 0.0),\n"
"        float2(1.0, 0.0)\n"
"    };\n"
"    VertexOut out;\n"
"    out.position = pos[vertexID];\n"
"    out.texCoords = tex[vertexID];\n"
"    return out;\n"
"}\n"
"\n"
"// YUV420P to RGB BT.709 matrix conversion on GPU\n"
"fragment float4 yuvFragmentShader(VertexOut in [[stage_in]],\n"
"                                  texture2d<float> yTexture [[texture(0)]],\n"
"                                  texture2d<float> uTexture [[texture(1)]],\n"
"                                  texture2d<float> vTexture [[texture(2)]]) {\n"
"    constexpr sampler s(address::clamp_to_edge, filter::linear);\n"
"    float y = yTexture.sample(s, in.texCoords).r;\n"
"    float u = uTexture.sample(s, in.texCoords).r - 0.5;\n"
"    float v = vTexture.sample(s, in.texCoords).r - 0.5;\n"
"\n"
"    float3 rgb;\n"
"    rgb.r = y + 1.5748 * v;\n"
"    rgb.g = y - 0.1873 * u - 0.4681 * v;\n"
"    rgb.b = y + 1.8556 * u;\n"
"    return float4(clamp(rgb, 0.0, 1.0), 1.0);\n"
"}\n";

/* --- Metal GPU Pipeline State --- */
typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        commandQueue;
    id<MTLRenderPipelineState> pipelineState;

    /* Double-buffered textures (index 0 and 1) to eliminate GPU stalling */
    id<MTLTexture> yTex[2];
    id<MTLTexture> uTex[2];
    id<MTLTexture> vTex[2];
    int            activeIdx;    /* which texture slot holds the newest completed frame */

    int            width;
    int            height;
    pthread_mutex_t tex_lock;
    bool           has_frame;
} metal_video_pipeline_t;

static metal_video_pipeline_t g_metal;

static int metal_pipeline_init(id<MTLDevice> device) {
    memset(&g_metal, 0, sizeof(g_metal));
    g_metal.device = device;
    g_metal.commandQueue = [device newCommandQueue];
    pthread_mutex_init(&g_metal.tex_lock, NULL);

    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:kMetalShaderSource];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
    if (!lib) {
        NSLog(@"[metal] Shader compilation failed: %@", error);
        return -1;
    }

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction   = [lib newFunctionWithName:@"yuvVertexShader"];
    desc.fragmentFunction = [lib newFunctionWithName:@"yuvFragmentShader"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    g_metal.pipelineState = [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!g_metal.pipelineState) {
        NSLog(@"[metal] Pipeline state creation failed: %@", error);
        return -1;
    }

    return 0;
}

static void ensure_textures(int w, int h) {
    /* If textures already exist and match the video resolution, do nothing */
    if (g_metal.width == w && g_metal.height == h && g_metal.yTex[0] != nil) {
        return;
    }

    g_metal.width = w;
    g_metal.height = h;

    for (int i = 0; i < 2; i++) {
        /* Y plane: full resolution (width x height) */
        MTLTextureDescriptor *yd = [MTLTextureDescriptor 
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
            width:w height:h mipmapped:NO];
        yd.usage = MTLTextureUsageShaderRead;
        yd.storageMode = MTLStorageModeShared;
        g_metal.yTex[i] = [g_metal.device newTextureWithDescriptor:yd];

        /* U and V planes: half resolution (w/2 x h/2) for YUV420P */
        MTLTextureDescriptor *uvd = [MTLTextureDescriptor 
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
            width:w / 2 height:h / 2 mipmapped:NO];
        uvd.usage = MTLTextureUsageShaderRead;
        uvd.storageMode = MTLStorageModeShared;
        g_metal.uTex[i] = [g_metal.device newTextureWithDescriptor:uvd];
        g_metal.vTex[i] = [g_metal.device newTextureWithDescriptor:uvd];
    }
}

static void metal_upload_yuv420p_frame(AVFrame *frame) {
    if (frame->format != AV_PIX_FMT_YUV420P) {
        return;
    }

    pthread_mutex_lock(&g_metal.tex_lock);

    int w = frame->width;
    int h = frame->height;
    ensure_textures(w, h);

    /* Write to the inactive buffer slot while the GPU displays the active one */
    int writeIdx = 1 - g_metal.activeIdx;

    /* 1. Upload Y plane directly from decoder memory */
    [g_metal.yTex[writeIdx] replaceRegion:MTLRegionMake2D(0, 0, w, h)
                              mipmapLevel:0
                                withBytes:frame->data[0]
                              bytesPerRow:frame->linesize[0]];

    /* 2. Upload U plane */
    [g_metal.uTex[writeIdx] replaceRegion:MTLRegionMake2D(0, 0, w / 2, h / 2)
                              mipmapLevel:0
                                withBytes:frame->data[1]
                              bytesPerRow:frame->linesize[1]];

    /* 3. Upload V plane */
    [g_metal.vTex[writeIdx] replaceRegion:MTLRegionMake2D(0, 0, w / 2, h / 2)
                              mipmapLevel:0
                                withBytes:frame->data[2]
                              bytesPerRow:frame->linesize[2]];

    /* Swap the active buffer index so the render loop picks up the new frame */
    g_metal.activeIdx = writeIdx;
    g_metal.has_frame = true;

    pthread_mutex_unlock(&g_metal.tex_lock);
}


/* ================================================================
 * Shared state
 * ================================================================ */
static volatile sig_atomic_t g_quit = 0;
static const char *g_serial_dev_path = ARDUPILOT_SERIAL_DEV_DEFAULT;

static void sigint_handler(int sig) { (void)sig; g_quit = 1; }

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

typedef struct {
    pthread_mutex_t lock;
    float roll, pitch, yaw;   /* -1..+1, spring-centered */
    float throttle;           /* 0..1, holds value */
    bool  sw1, sw2, sw3;      /* hold value */
} control_state_t;

static control_state_t g_ctrl;

/* Flight Sim Continuous Key State */
typedef struct {
    bool roll_left;
    bool roll_right;
    bool pitch_down;
    bool pitch_up;
    bool yaw_left;
    bool yaw_right;
    bool throttle_up;
    bool throttle_down;
} sim_keys_t;

static sim_keys_t g_keys = {0};

static void ctrl_init(control_state_t *c) {
    memset(c, 0, sizeof(*c));
    pthread_mutex_init(&c->lock, NULL);
}

typedef enum { AXIS_ROLL, AXIS_PITCH, AXIS_YAW } axis_t;

static void ctrl_nudge_axis(axis_t axis, float delta) {
    pthread_mutex_lock(&g_ctrl.lock);
    float *v = (axis == AXIS_ROLL) ? &g_ctrl.roll
             : (axis == AXIS_PITCH) ? &g_ctrl.pitch
             : &g_ctrl.yaw;
    *v = clampf(*v + delta, -1.0f, 1.0f);
    pthread_mutex_unlock(&g_ctrl.lock);
}

static void ctrl_nudge_throttle(float delta) {
    pthread_mutex_lock(&g_ctrl.lock);
    g_ctrl.throttle = clampf(g_ctrl.throttle + delta, 0.0f, 1.0f);
    pthread_mutex_unlock(&g_ctrl.lock);
}

static bool ctrl_toggle_sw(int which) {
    bool val;
    pthread_mutex_lock(&g_ctrl.lock);
    if (which == 1)      { g_ctrl.sw1 = !g_ctrl.sw1; val = g_ctrl.sw1; }
    else if (which == 2) { g_ctrl.sw2 = !g_ctrl.sw2; val = g_ctrl.sw2; }
    else                 { g_ctrl.sw3 = !g_ctrl.sw3; val = g_ctrl.sw3; }
    pthread_mutex_unlock(&g_ctrl.lock);
    return val;
}

static void ctrl_center(void) {
    pthread_mutex_lock(&g_ctrl.lock);
    g_ctrl.roll = g_ctrl.pitch = g_ctrl.yaw = 0.0f;
    pthread_mutex_unlock(&g_ctrl.lock);
}

/* Called once per serial-thread tick to spring roll/pitch/yaw back to 0.
 * Throttle and the switches are untouched -- they hold their last value. */
static void ctrl_apply_decay(double dt) {
    float factor = expf((float)(-dt / DECAY_TAU_SEC));
    pthread_mutex_lock(&g_ctrl.lock);
    g_ctrl.roll  *= factor;
    g_ctrl.pitch *= factor;
    g_ctrl.yaw   *= factor;
    pthread_mutex_unlock(&g_ctrl.lock);
}

typedef struct {
    float roll, pitch, yaw, throttle;
    bool sw1, sw2, sw3;
} ctrl_snapshot_t;

static void ctrl_snapshot(ctrl_snapshot_t *out) {
    pthread_mutex_lock(&g_ctrl.lock);
    out->roll = g_ctrl.roll; out->pitch = g_ctrl.pitch; out->yaw = g_ctrl.yaw;
    out->throttle = g_ctrl.throttle;
    out->sw1 = g_ctrl.sw1; out->sw2 = g_ctrl.sw2; out->sw3 = g_ctrl.sw3;
    pthread_mutex_unlock(&g_ctrl.lock);
}

/* ================================================================
 * CRSF encode (channels we send TO ArduPilot)
 * ================================================================ */
static uint16_t encode_axis(float v /* -1..1 */) {
    int raw = 992 + (int)(v * 820.0f);
    if (raw < 172) raw = 172;
    if (raw > 1811) raw = 1811;
    return (uint16_t)raw;
}
static uint16_t encode_throttle(float v /* 0..1 */) {
    int raw = 172 + (int)(v * (1811 - 172));
    if (raw < 172) raw = 172;
    if (raw > 1811) raw = 1811;
    return (uint16_t)raw;
}
static uint16_t encode_switch(bool on) { return on ? 1811 : 172; }

/* Verified against the standard published CRC-8/DVB-S2 test vector:
 * crsf_crc8("123456789", 9) == 0xBC. */
static uint8_t crc8_dvb_s2(uint8_t crc, uint8_t a) {
    crc ^= a;
    for (int i = 0; i < 8; i++)
        crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xD5) : (uint8_t)(crc << 1);
    return crc;
}
static uint8_t crsf_crc8(const uint8_t *data, int len) {
    uint8_t crc = 0;
    for (int i = 0; i < len; i++) crc = crc8_dvb_s2(crc, data[i]);
    return crc;
}

/* Pack 16 x 11-bit channel values into 22 bytes, per the CRSF spec. */
static void crsf_pack_channels(const uint16_t ch[16], uint8_t out[22]) {
    uint32_t bitbuf = 0;
    int bitcount = 0, idx = 0;
    for (int i = 0; i < 16; i++) {
        bitbuf |= ((uint32_t)(ch[i] & 0x7FF)) << bitcount;
        bitcount += 11;
        while (bitcount >= 8) {
            out[idx++] = (uint8_t)(bitbuf & 0xFF);
            bitbuf >>= 8;
            bitcount -= 8;
        }
    }
}

/* Builds a full RC_CHANNELS_PACKED frame: sync + len + type + 22B payload + crc = 26 bytes. */
static int crsf_build_rc_frame(const ctrl_snapshot_t *s, uint8_t out[26]) {
    uint16_t ch[16];
    ch[0] = encode_axis(s->roll);
    ch[1] = encode_axis(s->pitch);
    ch[2] = encode_throttle(s->throttle);
    ch[3] = encode_axis(s->yaw);
    ch[4] = encode_switch(s->sw1);
    ch[5] = encode_switch(s->sw2);
    ch[6] = encode_switch(s->sw3);
    for (int i = 7; i < 16; i++) ch[i] = 992;

    out[0] = CRSF_SYNC_FC;
    out[1] = 22 + 2; /* len = type(1) + payload(22) + crc(1) */
    out[2] = CRSF_FRAMETYPE_RC_CHANNELS;
    crsf_pack_channels(ch, &out[3]);
    out[25] = crsf_crc8(&out[2], 1 + 22);
    return 26;
}

/* --- Thread-Safe Telemetry Cache for HUD Rendering --- */
typedef struct {
    pthread_mutex_t lock;
    float pitch_deg;
    float roll_deg;
    float yaw_deg;
    float voltage;
    float current;
    uint32_t mah;
    uint8_t battery_pct;
    uint8_t lq;
    int8_t  rssi;
    char    flight_mode[16];
} telem_cache_t;

static telem_cache_t g_telem_cache = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .flight_mode = "MANUAL"
};

static void telem_update_attitude(float pitch, float roll, float yaw) {
    pthread_mutex_lock(&g_telem_cache.lock);
    g_telem_cache.pitch_deg = pitch;
    g_telem_cache.roll_deg  = roll;
    g_telem_cache.yaw_deg   = yaw;
    pthread_mutex_unlock(&g_telem_cache.lock);
}

static void telem_update_battery(float v, float a, uint32_t mah, uint8_t pct) {
    pthread_mutex_lock(&g_telem_cache.lock);
    g_telem_cache.voltage = v;
    g_telem_cache.current = a;
    g_telem_cache.mah = mah;
    g_telem_cache.battery_pct = pct;
    pthread_mutex_unlock(&g_telem_cache.lock);
}

static void telem_update_link(int8_t rssi, uint8_t lq) {
    pthread_mutex_lock(&g_telem_cache.lock);
    g_telem_cache.rssi = rssi;
    g_telem_cache.lq = lq;
    pthread_mutex_unlock(&g_telem_cache.lock);
}

static void telem_update_mode(const char *mode) {
    pthread_mutex_lock(&g_telem_cache.lock);
    strncpy(g_telem_cache.flight_mode, mode, sizeof(g_telem_cache.flight_mode) - 1);
    g_telem_cache.flight_mode[sizeof(g_telem_cache.flight_mode) - 1] = '\0';
    pthread_mutex_unlock(&g_telem_cache.lock);
}

/* ================================================================
 * CRSF parse (telemetry ArduPilot sends back on the same link)
 * ================================================================ */
static void handle_crsf_frame(const uint8_t *f, int len) {
    uint8_t type = f[0];
    const uint8_t *payload = &f[1];
    int payload_len = len - 2; /* minus type and crc */

    switch (type) {
    case CRSF_FRAMETYPE_ATTITUDE:
        if (payload_len >= 6) {
            int16_t pitch = (int16_t)((payload[0] << 8) | payload[1]);
            int16_t roll  = (int16_t)((payload[2] << 8) | payload[3]);
            int16_t yaw   = (int16_t)((payload[4] << 8) | payload[5]);
            telem_update_attitude((float)pitch / 10000.0f * 180.0f / (float)M_PI,
                                  (float)roll  / 10000.0f * 180.0f / (float)M_PI,
                                  (float)yaw   / 10000.0f * 180.0f / (float)M_PI);
        }
        break;

    case CRSF_FRAMETYPE_BATTERY_SENSOR:
        if (payload_len >= 8) {
            uint16_t mv  = (uint16_t)((payload[0] << 8) | payload[1]);
            uint16_t ca  = (uint16_t)((payload[2] << 8) | payload[3]);
            uint32_t mah = ((uint32_t)payload[4] << 16) | ((uint32_t)payload[5] << 8) | payload[6];
            uint8_t  pct = payload[7];
            telem_update_battery(mv / 10.0f, ca / 10.0f, mah, pct);
        }
        break;

    case CRSF_FRAMETYPE_FLIGHT_MODE: {
        char mode[17];
        int n = payload_len < 16 ? payload_len : 16;
        if (n < 0) n = 0;
        memcpy(mode, payload, (size_t)n);
        mode[n] = '\0';
        telem_update_mode(mode);
        break;
    }

    case CRSF_FRAMETYPE_LINK_STATISTICS:
        if (payload_len >= 4) {
            uint8_t up_rssi1 = payload[0];
            uint8_t up_lq    = payload[2];
            telem_update_link((int8_t)up_rssi1, up_lq);
        }
        break;

    default:
        break;
    }
}

typedef enum { CRSF_WAIT_SYNC, CRSF_WAIT_LEN, CRSF_WAIT_DATA } crsf_rx_state_t;

typedef struct {
    crsf_rx_state_t state;
    uint8_t buf[CRSF_MAX_FRAME_LEN];
    int len;  /* expected type+payload+crc byte count, from the length field */
    int pos;  /* bytes collected so far in this frame */
} crsf_parser_t;

static void crsf_parser_init(crsf_parser_t *p) { p->state = CRSF_WAIT_SYNC; p->pos = 0; p->len = 0; }

static void crsf_parser_feed(crsf_parser_t *p, uint8_t byte) {
    switch (p->state) {
    case CRSF_WAIT_SYNC:
        /* Being lenient about the address byte here since it's not fully
         * pinned down which one your ArduPilot build uses for outbound
         * telemetry -- see the assumptions note at the top of this file. */
        if (byte == 0xC8 || byte == 0xEE || byte == 0xEA || byte == 0xEC) {
            p->pos = 0;
            p->state = CRSF_WAIT_LEN;
        }
        break;
    case CRSF_WAIT_LEN:
        if (byte < 2 || byte > CRSF_MAX_FRAME_LEN - 2) { p->state = CRSF_WAIT_SYNC; break; }
        p->len = byte;
        p->pos = 0;
        p->state = CRSF_WAIT_DATA;
        break;
    case CRSF_WAIT_DATA:
        p->buf[p->pos++] = byte;
        if (p->pos == p->len) {
            uint8_t crc_calc = crsf_crc8(p->buf, p->len - 1);
            if (crc_calc == p->buf[p->len - 1]) {
                handle_crsf_frame(p->buf, p->len);
            }
            p->state = CRSF_WAIT_SYNC;
        }
        break;
    }
}

/* --- Low-Latency HEVC Decoder Context --- */
typedef struct {
    const AVCodec        *codec;
    AVCodecContext       *ctx;
    AVCodecParserContext *parser;
    AVFrame              *frame;
    AVPacket             *pkt;
} hevc_decoder_t;

static int hevc_decoder_init(hevc_decoder_t *dec) {
    memset(dec, 0, sizeof(*dec));

    /* 1. Find the H.265/HEVC decoder */
    dec->codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    if (!dec->codec) {
        fprintf(stderr, "[decoder] H.265/HEVC codec not found\n");
        return -1;
    }

    /* 2. Create the H.265 bitstream parser (reassembles NALs across USB chunks) */
    dec->parser = av_parser_init(dec->codec->id);
    if (!dec->parser) {
        fprintf(stderr, "[decoder] Failed to create parser\n");
        return -1;
    }

    /* 3. Allocate the decoder context */
    dec->ctx = avcodec_alloc_context3(dec->codec);
    if (!dec->ctx) {
        av_parser_close(dec->parser);
        return -1;
    }

    /* 4. Intra-Refresh Low-Latency Tuning */
    dec->ctx->flags  |= AV_CODEC_FLAG_LOW_DELAY;
    dec->ctx->flags2 |= AV_CODEC_FLAG2_FAST; /* Notice: NO SHOW_ALL (stops grey flashing) */

    /* Pure single-thread decoding: eliminates HEVC intra-refresh slice bugs */
    dec->ctx->thread_type = 0;
    dec->ctx->thread_count = 1;

    /* Eliminate B-frame buffering */
    dec->ctx->max_b_frames = 0;
    dec->ctx->has_b_frames = 0;
    dec->ctx->delay = 0;

    AVDictionary *opts = NULL;
    av_dict_set(&opts, "tune", "zerolatency", 0);

    /* 5. Open codec */
    if (avcodec_open2(dec->ctx, dec->codec, &opts) < 0) {
        fprintf(stderr, "[decoder] Failed to open HEVC codec\n");
        av_dict_free(&opts);
        avcodec_free_context(&dec->ctx);
        av_parser_close(dec->parser);
        return -1;
    }
    av_dict_free(&opts);

    /* 6. Scratchpad buffers */
    dec->frame = av_frame_alloc();
    dec->pkt   = av_packet_alloc();
    if (!dec->frame || !dec->pkt) {
        return -1;
    }

    return 0;
}


static void on_frame_decoded(AVFrame *frame) {
    static uint64_t frame_count = 0;
    static uint64_t last_count = 0;
    static double last_log_t = 0;

    frame_count++;
    double t = now_seconds();

    /* Print real-time FPS counter every 1.0 second */
    if (t - last_log_t >= 1.0) {
        double fps = (double)(frame_count - last_count) / (t - last_log_t);
        fprintf(stderr, "\r[decoder] Live FPS: %5.1f | Total Frames: %llu | Res: %dx%d   ",
                fps, (unsigned long long)frame_count, frame->width, frame->height);
        fflush(stderr);
        last_log_t = t;
        last_count = frame_count;
    }

    /* Push decoded planes to Metal GPU textures immediately */
    metal_upload_yuv420p_frame(frame);
}

static void hevc_decoder_feed_chunk(hevc_decoder_t *dec, const uint8_t *data, int size) {
    while (size > 0 && !g_quit) {
        /* Parse raw bytes into a discrete NAL packet */
        int consumed = av_parser_parse2(dec->parser, dec->ctx,
                                        &dec->pkt->data, &dec->pkt->size,
                                        data, size,
                                        AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (consumed < 0) {
            break;
        }

        data += consumed;
        size -= consumed;

        /* When a complete NAL unit is ready, push it immediately to the decoder */
        if (dec->pkt->size > 0) {
            if (avcodec_send_packet(dec->ctx, dec->pkt) == 0) {
                while (avcodec_receive_frame(dec->ctx, dec->frame) == 0) {
                    on_frame_decoded(dec->frame);
                    /* Crucial for low latency: free buffer references immediately */
                    av_frame_unref(dec->frame);
                }
            }
        }
    }
}

static void hevc_decoder_destroy(hevc_decoder_t *dec) {
    if (dec->frame)  av_frame_free(&dec->frame);
    if (dec->pkt)    av_packet_free(&dec->pkt);
    if (dec->ctx)    avcodec_free_context(&dec->ctx);
    if (dec->parser) av_parser_close(dec->parser);
}

/* ================================================================
 * Serial Thread: ArduPilot CRSF Link + Flight Sim Stick Physics
 * ================================================================ */
static void *serial_thread_fn(void *arg) {
    (void)arg;
    crsf_parser_t parser;
    crsf_parser_init(&parser);

    int fd = -1;
    double last_tx_t = now_seconds();
    double last_physics_t = now_seconds();
    double last_log_t = now_seconds();
    uint64_t tx_packets = 0;
    uint64_t rx_bytes = 0;

    fprintf(stderr, "[serial-diag] Thread started for device: %s\n", g_serial_dev_path);

    while (!g_quit) {
        /* 1. Attempt connection if port is not yet open */
        if (fd < 0) {
            fd = open(g_serial_dev_path, O_RDWR | O_NOCTTY | O_NONBLOCK);
            if (fd < 0) {
                double t = now_seconds();
                if (t - last_log_t >= 2.0) {
                    fprintf(stderr, "[serial-diag] Cannot open %s: %s (retrying...)\n",
                            g_serial_dev_path, strerror(errno));
                    last_log_t = t;
                }
                usleep(SERIAL_RETRY_MS * 1000);
                continue;
            }

            /* Configure raw 115200 8N1 serial port */
            struct termios tty;
            memset(&tty, 0, sizeof(tty));
            if (tcgetattr(fd, &tty) == 0) {
                cfsetspeed(&tty, ARDUPILOT_BAUD);
                tty.c_cflag |= (CLOCAL | CREAD);
                tty.c_cflag &= ~CSIZE;
                tty.c_cflag |= CS8;
                tty.c_cflag &= ~PARENB;
                tty.c_cflag &= ~CSTOPB;
                tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
                tty.c_iflag &= ~(IXON | IXOFF | IXANY);
                tty.c_oflag &= ~OPOST;
                tcsetattr(fd, TCSANOW, &tty);

                /* Assert DTR & RTS so Linux CDC-ACM brings up the carrier */
                int modem_ctrl = TIOCM_DTR | TIOCM_RTS;
                ioctl(fd, TIOCMBIS, &modem_ctrl);
            }
            fprintf(stderr, "[serial-diag] Port OPENED & DTR/RTS asserted on %s!\n", g_serial_dev_path);
        }

        /* 2. Read incoming telemetry bytes from ArduPilot */
        uint8_t rx_buf[128];
        ssize_t n = read(fd, rx_buf, sizeof(rx_buf));
        if (n > 0) {
            rx_bytes += n;
            for (ssize_t i = 0; i < n; i++) {
                crsf_parser_feed(&parser, rx_buf[i]);
            }
        } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            fprintf(stderr, "[serial-diag] Connection lost: %s\n", strerror(errno));
            close(fd);
            fd = -1;
            continue;
        }

        /* 3. Flight Simulator Proportional Control Physics (100 Hz loop) */
        double t = now_seconds();
        double dt = t - last_physics_t;
        if (dt >= 0.01) {
            float target_roll  = (g_keys.roll_right ? 1.0f : 0.0f) - (g_keys.roll_left ? 1.0f : 0.0f);
            float target_pitch = (g_keys.pitch_up   ? 1.0f : 0.0f) - (g_keys.pitch_down ? 1.0f : 0.0f);
            float target_yaw   = (g_keys.yaw_right  ? 1.0f : 0.0f) - (g_keys.yaw_left   ? 1.0f : 0.0f);

            float blend = 1.0f - expf((float)(-dt / DECAY_TAU_SEC));

            pthread_mutex_lock(&g_ctrl.lock);
            g_ctrl.roll  += (target_roll  - g_ctrl.roll)  * blend;
            g_ctrl.pitch += (target_pitch - g_ctrl.pitch) * blend;
            g_ctrl.yaw   += (target_yaw   - g_ctrl.yaw)   * blend;

            const float THROTTLE_RATE = 0.50f;
            if (g_keys.throttle_up)   g_ctrl.throttle = clampf(g_ctrl.throttle + THROTTLE_RATE * (float)dt, 0.0f, 1.0f);
            if (g_keys.throttle_down) g_ctrl.throttle = clampf(g_ctrl.throttle - THROTTLE_RATE * (float)dt, 0.0f, 1.0f);
            pthread_mutex_unlock(&g_ctrl.lock);

            last_physics_t = t;
        }

        /* 4. Transmit CRSF RC channel packets to ArduPilot at CRSF_TX_HZ */
        if (t - last_tx_t >= (1.0 / CRSF_TX_HZ)) {
            ctrl_snapshot_t snap;
            ctrl_snapshot(&snap);
            uint8_t tx_frame[26];
            int frame_len = crsf_build_rc_frame(&snap, tx_frame);
            ssize_t w = write(fd, tx_frame, frame_len);
            if (w > 0) {
                tx_packets++;
            } else if (w < 0 && errno != EAGAIN) {
                fprintf(stderr, "[serial-diag] Write error: %s\n", strerror(errno));
            }
            last_tx_t = t;
        }

        /* Print periodic TX/RX heartbeat */
        if (t - last_log_t >= 2.0) {
            fprintf(stderr, "\r[serial-diag] TX CRSF: %llu pkts | RX Telem: %llu bytes   \n",
                    (unsigned long long)tx_packets, (unsigned long long)rx_bytes);
            last_log_t = t;
        }

        usleep(1000);
    }

    if (fd >= 0) close(fd);
    return NULL;
}


/* ================================================================
 * Video thread: Luckfox vendor bulk endpoint -> Low-Latency Decoder
 * ================================================================ */
static void *video_thread_fn(void *arg) {
    (void)arg;
    libusb_context *ctx = NULL;
    if (libusb_init(&ctx) < 0) {
        fprintf(stderr, "[video-diag] libusb_init failed\n");
        return NULL;
    }

    /* 1. Initialize HEVC Decoder */
    hevc_decoder_t dec;
    if (hevc_decoder_init(&dec) != 0) {
        fprintf(stderr, "[video-diag] Decoder init failed\n");
        libusb_exit(ctx);
        return NULL;
    }
    fprintf(stderr, "[video-diag] HEVC Decoder initialized.\n");

    uint8_t *usb_buffer = malloc(BUFFER_SIZE);
    if (!usb_buffer) {
        hevc_decoder_destroy(&dec);
        libusb_exit(ctx);
        return NULL;
    }

    fprintf(stderr, "[video-diag] Searching for USB device VID:0x%04X PID:0x%04X...\n", TARGET_VID, TARGET_PID);

    double last_poll_warn = 0;

    /* 2. USB Ingestion & Decode Loop */
    while (!g_quit) {
        libusb_device_handle *dev_handle = libusb_open_device_with_vid_pid(ctx, TARGET_VID, TARGET_PID);
        if (!dev_handle) {
            double t = now_seconds();
            if (t - last_poll_warn >= 3.0) {
                fprintf(stderr, "[video-diag] Waiting for USB device (0x%04X:0x%04X)... Is it plugged in?\n",
                        TARGET_VID, TARGET_PID);
                last_poll_warn = t;
            }
            usleep(250 * 1000);
            continue;
        }

        fprintf(stderr, "[video-diag] Device found! Opening descriptors...\n");

        libusb_device *dev = libusb_get_device(dev_handle);
        struct libusb_config_descriptor *config = NULL;
        if (libusb_get_active_config_descriptor(dev, &config) < 0) {
            fprintf(stderr, "[video-diag] Failed to get config descriptor\n");
            libusb_close(dev_handle);
            usleep(500 * 1000);
            continue;
        }

        int target_intf = -1;
        uint8_t ep_in_addr = 0;
        for (int i = 0; i < config->bNumInterfaces && target_intf < 0; i++) {
            const struct libusb_interface *intf = &config->interface[i];
            for (int a = 0; a < intf->num_altsetting && target_intf < 0; a++) {
                const struct libusb_interface_descriptor *alt = &intf->altsetting[a];
                if (alt->bInterfaceClass == LIBUSB_CLASS_VENDOR_SPEC) {
                    for (int e = 0; e < alt->bNumEndpoints; e++) {
                        const struct libusb_endpoint_descriptor *ep = &alt->endpoint[e];
                        if ((ep->bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_IN &&
                            (ep->bmAttributes & LIBUSB_TRANSFER_TYPE_MASK) == LIBUSB_TRANSFER_TYPE_BULK) {
                            target_intf = alt->bInterfaceNumber;
                            ep_in_addr = ep->bEndpointAddress;
                            break;
                        }
                    }
                }
            }
        }
        libusb_free_config_descriptor(config);

        if (target_intf < 0 || ep_in_addr == 0) {
            fprintf(stderr, "[video-diag] Could not locate vendor bulk IN endpoint on this device\n");
            libusb_close(dev_handle);
            usleep(1000 * 1000);
            continue;
        }

        /* Detach OS kernel driver if attached */
        libusb_detach_kernel_driver(dev_handle, target_intf);

        int claim_ret = libusb_claim_interface(dev_handle, target_intf);
        if (claim_ret < 0) {
            fprintf(stderr, "[video-diag] Failed to claim interface %d: %s (Try running with sudo?)\n",
                    target_intf, libusb_strerror(claim_ret));
            libusb_close(dev_handle);
            usleep(1000 * 1000);
            continue;
        }

        fprintf(stderr, "[video-diag] Connected! Claimed interface %d, listening on endpoint 0x%02X\n",
                target_intf, ep_in_addr);

        uint64_t bytes_rx = 0;
        double last_stat_t = now_seconds();

        while (!g_quit) {
            int actual_length = 0;
            int ret = libusb_bulk_transfer(dev_handle, ep_in_addr, usb_buffer, BUFFER_SIZE, &actual_length, 1000);

            if (ret == 0 && actual_length > 0) {
                bytes_rx += actual_length;
                double t = now_seconds();
                if (t - last_stat_t >= 2.0) {
                    fprintf(stderr, "[video-diag] Receiving stream: %.1f KB/s\n",
                            (bytes_rx / (t - last_stat_t)) / 1024.0);
                    bytes_rx = 0;
                    last_stat_t = t;
                }
                hevc_decoder_feed_chunk(&dec, usb_buffer, actual_length);
            } else if (ret == LIBUSB_ERROR_TIMEOUT) {
                continue;
            } else if (ret == LIBUSB_ERROR_NO_DEVICE || ret == LIBUSB_ERROR_IO) {
                fprintf(stderr, "\n[video-diag] Device disconnected\n");
                break;
            } else if (ret < 0) {
                if (g_quit) break;
                fprintf(stderr, "\n[video-diag] Bulk transfer error: %s\n", libusb_strerror(ret));
                break;
            }
        }

        libusb_release_interface(dev_handle, target_intf);
        libusb_close(dev_handle);
    }

    free(usb_buffer);
    hevc_decoder_destroy(&dec);
    libusb_exit(ctx);
    return NULL;
}



/* ================================================================
 * Keyboard input (main thread)
 * ================================================================ */
static struct termios g_orig_termios;
static bool g_termios_saved = false;

static void restore_terminal(void) {
    if (g_termios_saved) tcsetattr(STDIN_FILENO, TCSANOW, &g_orig_termios);
}

static int enable_cbreak_terminal(void) {
    if (tcgetattr(STDIN_FILENO, &g_orig_termios) != 0) return -1;
    g_termios_saved = true;
    struct termios raw = g_orig_termios;
    raw.c_lflag &= (tcflag_t)~(ECHO | ICANON); /* keep ISIG so Ctrl+C still works */
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return -1;
    atexit(restore_terminal);
    return 0;
}

static void keyboard_loop(void) {
    while (!g_quit) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(STDIN_FILENO, &rfds);
        struct timeval tv = { 0, 100 * 1000 }; /* wake up regularly to notice g_quit */
        int sel = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
        if (sel <= 0) continue;

        unsigned char c;
        if (read(STDIN_FILENO, &c, 1) <= 0) continue;

        if (c == 0x1B) { /* possible arrow-key escape sequence: ESC [ A/B/C/D */
            fd_set rfds2; struct timeval tv2 = { 0, 20 * 1000 };
            FD_ZERO(&rfds2); FD_SET(STDIN_FILENO, &rfds2);
            if (select(STDIN_FILENO + 1, &rfds2, NULL, NULL, &tv2) <= 0) continue;
            unsigned char c2;
            if (read(STDIN_FILENO, &c2, 1) <= 0 || c2 != '[') continue;

            FD_ZERO(&rfds2); FD_SET(STDIN_FILENO, &rfds2);
            if (select(STDIN_FILENO + 1, &rfds2, NULL, NULL, &tv2) <= 0) continue;
            unsigned char c3;
            if (read(STDIN_FILENO, &c3, 1) <= 0) continue;

            switch (c3) {
                case 'A': ctrl_nudge_throttle(+THROTTLE_STEP); break; /* Up */
                case 'B': ctrl_nudge_throttle(-THROTTLE_STEP); break; /* Down */
                case 'C': ctrl_nudge_axis(AXIS_YAW, +AXIS_STEP); break; /* Right */
                case 'D': ctrl_nudge_axis(AXIS_YAW, -AXIS_STEP); break; /* Left */
                default: break;
            }
            continue;
        }

        switch (c) {
            case KEY_ROLL_LEFT:  ctrl_nudge_axis(AXIS_ROLL, -AXIS_STEP); break;
            case KEY_ROLL_RIGHT: ctrl_nudge_axis(AXIS_ROLL, +AXIS_STEP); break;
            case KEY_PITCH_DOWN: ctrl_nudge_axis(AXIS_PITCH, -AXIS_STEP); break;
            case KEY_PITCH_UP:   ctrl_nudge_axis(AXIS_PITCH, +AXIS_STEP); break;
            case KEY_SW1: fprintf(stderr, "[input] SW1 -> %s\n", ctrl_toggle_sw(1) ? "HIGH" : "LOW"); break;
            case KEY_SW2: fprintf(stderr, "[input] SW2 -> %s\n", ctrl_toggle_sw(2) ? "HIGH" : "LOW"); break;
            case KEY_SW3: fprintf(stderr, "[input] SW3 -> %s\n", ctrl_toggle_sw(3) ? "HIGH" : "LOW"); break;
            case KEY_CENTER: ctrl_center(); break;
            case KEY_QUIT: g_quit = 1; break;
            default: break;
        }
    }
}


/* ================================================================
 * Telemetry Heads-Up Display (HUD) Overlay View
 * ================================================================ */
@interface HUDOverlayView : NSView
@end

@implementation HUDOverlayView

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    CGContextRef ctx = [[NSGraphicsContext currentContext] CGContext];
    if (!ctx) return;

    /* Snapshot current telemetry and pilot controls without stalling threads */
    telem_cache_t snap;
    pthread_mutex_lock(&g_telem_cache.lock);
    snap = g_telem_cache;
    pthread_mutex_unlock(&g_telem_cache.lock);

    ctrl_snapshot_t ctrl;
    ctrl_snapshot(&ctrl);

    CGFloat w = self.bounds.size.width;
    CGFloat h = self.bounds.size.height;
    CGPoint center = CGPointMake(w * 0.5, h * 0.5);

    /* 1. Artificial Horizon / Pitch Ladder (Green HUD style) */
    CGContextSaveGState(ctx);
    CGContextTranslateCTM(ctx, center.x, center.y);
    CGContextRotateCTM(ctx, -snap.roll_deg * (CGFloat)M_PI / 180.0);

    CGFloat pitchOffset = snap.pitch_deg * 6.0;
    CGContextSetStrokeColorWithColor(ctx, [NSColor colorWithCalibratedRed:0.0 green:1.0 blue:0.4 alpha:0.85].CGColor);
    CGContextSetLineWidth(ctx, 2.0);

    /* Center bore sight reticle & horizon wings */
    CGContextStrokeRect(ctx, CGRectMake(-8, -8, 16, 16));
    CGContextMoveToPoint(ctx, -140, pitchOffset);
    CGContextAddLineToPoint(ctx, -40, pitchOffset);
    CGContextMoveToPoint(ctx, 40, pitchOffset);
    CGContextAddLineToPoint(ctx, 140, pitchOffset);
    CGContextStrokePath(ctx);
    CGContextRestoreGState(ctx);

    /* 2. Throttle Level Indicator (Vertical Bar on Left) */
    CGFloat thH = 200.0;
    CGFloat thW = 12.0;
    CGFloat thX = 40.0;
    CGFloat thY = (h - thH) * 0.5;

    CGContextSetStrokeColorWithColor(ctx, [NSColor whiteColor].CGColor);
    CGContextStrokeRect(ctx, CGRectMake(thX, thY, thW, thH));
    CGFloat fillH = thH * ctrl.throttle;
    CGContextSetFillColorWithColor(ctx, [NSColor colorWithCalibratedRed:0.2 green:0.8 blue:1.0 alpha:0.75].CGColor);
    CGContextFillRect(ctx, CGRectMake(thX + 1, thY + 1, thW - 2, fillH));

    /* 3. Top & Bottom OSD Telemetry Strings */
    NSString *osdTop = [NSString stringWithFormat:@"MODE: %s   LQ: %u%%   RSSI: -%ddBm",
                        snap.flight_mode, snap.lq, snap.rssi];
    NSString *osdBot = [NSString stringWithFormat:@"BAT: %.1fV  %.1fA  (%u%%)   THR: %d%%",
                        snap.voltage, snap.current, snap.battery_pct, (int)(ctrl.throttle * 100.0f)];

    NSDictionary *attr = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:14 weight:NSFontWeightBold],
        NSForegroundColorAttributeName: [NSColor colorWithCalibratedRed:0.0 green:1.0 blue:0.4 alpha:0.95]
    };

    [osdTop drawAtPoint:CGPointMake(40, h - 35) withAttributes:attr];
    [osdBot drawAtPoint:CGPointMake(40, 20) withAttributes:attr];

    /* 4. High-Precision Latency Stopwatch (Top-Right Corner) */
    uint64_t current_ms = (uint64_t)(now_seconds() * 1000.0);
    uint32_t ms_rollover = (uint32_t)(current_ms % 10000); /* 4 digits: 0000 to 9999 ms */

    NSString *msString = [NSString stringWithFormat:@"%04u ms", ms_rollover];

    /* Dark high-contrast background box for easy camera reading */
    CGFloat boxW = 140.0;
    CGFloat boxH = 40.0;
    CGFloat boxX = w - boxW - 30.0;
    CGFloat boxY = h - boxH - 25.0;

    CGContextSetFillColorWithColor(ctx, [NSColor colorWithCalibratedRed:0.0 green:0.0 blue:0.0 alpha:0.75].CGColor);
    CGContextFillRect(ctx, CGRectMake(boxX, boxY, boxW, boxH));
    CGContextSetStrokeColorWithColor(ctx, [NSColor colorWithCalibratedRed:1.0 green:0.8 blue:0.0 alpha:0.9].CGColor);
    CGContextSetLineWidth(ctx, 1.5);
    CGContextStrokeRect(ctx, CGRectMake(boxX, boxY, boxW, boxH));

    NSDictionary *msAttr = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:22 weight:NSFontWeightHeavy],
        NSForegroundColorAttributeName: [NSColor colorWithCalibratedRed:1.0 green:0.85 blue:0.1 alpha:1.0] /* Bright amber */
    };

    [msString drawAtPoint:CGPointMake(boxX + 16, boxY + 8) withAttributes:msAttr];
}

@end


/* ================================================================
 * Metal Video Presentation View (Lowest-Latency CAMetalLayer)
 * ================================================================ */
@interface FPVMetalView : NSView
@property (nonatomic, strong) CAMetalLayer *metalLayer;
@end

@implementation FPVMetalView

- (instancetype)initWithFrame:(NSRect)frameRect {
    if ((self = [super initWithFrame:frameRect])) {
        self.wantsLayer = YES; /* Tell macOS to use GPU layer-backing */
    }
    return self;
}

- (CALayer *)makeBackingLayer {
    CAMetalLayer *layer = [CAMetalLayer layer];
    layer.device = g_metal.device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;        /* Disables readback for maximum GPU throughput */
    layer.presentsWithTransaction = NO; /* Bypasses WindowServer sync for lowest latency */
    layer.maximumDrawableCount = 2;     /* CRITICAL: Strict double-buffering (eliminates 1-frame queue lag) */
    self.metalLayer = layer;
    return layer;
}

- (void)renderFrame {
    pthread_mutex_lock(&g_metal.tex_lock);
    if (!g_metal.has_frame || !g_metal.yTex[g_metal.activeIdx]) {
        pthread_mutex_unlock(&g_metal.tex_lock);
        return;
    }

    int idx = g_metal.activeIdx;
    id<MTLTexture> y = g_metal.yTex[idx];
    id<MTLTexture> u = g_metal.uTex[idx];
    id<MTLTexture> v = g_metal.vTex[idx];
    pthread_mutex_unlock(&g_metal.tex_lock);

    /* Grab the next screen buffer ready on the display */
    id<CAMetalDrawable> drawable = [self.metalLayer nextDrawable];
    if (!drawable) return;

    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawable.texture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cmdBuf renderCommandEncoderWithDescriptor:rpd];

    /* Bind our compiled shader and the 3 YUV textures */
    [enc setRenderPipelineState:g_metal.pipelineState];
    [enc setFragmentTexture:y atIndex:0];
    [enc setFragmentTexture:u atIndex:1];
    [enc setFragmentTexture:v atIndex:2];

    /* Draw full screen quad (4 vertices generated by our shader) */
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [enc endEncoding];

    /* Submit immediately to the display hardware */
    [cmdBuf presentDrawable:drawable];
    [cmdBuf commit];
}

- (void)displayLinkDidFire:(CADisplayLink *)link {
    (void)link;
    [self renderFrame];
}

@end


/* ================================================================
 * Window with Instant Zero-Latency Fullscreen / Windowed Toggle
 * ================================================================ */
@interface FullscreenWindow : NSWindow
@property (nonatomic, assign) BOOL isFullscreen;
@property (nonatomic, assign) NSRect savedWindowedRect;
- (void)toggleFullscreenMode;
@end

@implementation FullscreenWindow

- (BOOL)canBecomeKeyWindow  { return YES; }
- (BOOL)canBecomeMainWindow { return YES; }

- (void)toggleFullscreenMode {
    if (self.isFullscreen) {
        /* 1. Switch back to Windowed Mode (allows QGroundControl side-by-side) */
        self.styleMask = NSWindowStyleMaskTitled | 
                         NSWindowStyleMaskClosable | 
                         NSWindowStyleMaskMiniaturizable | 
                         NSWindowStyleMaskResizable;
        [self setLevel:NSNormalWindowLevel];
        [self setTitle:@"FPV Video + Telemetry HUD"];

        NSRect target = (self.savedWindowedRect.size.width > 0) ? self.savedWindowedRect : NSMakeRect(100, 100, 1280, 720);
        [self setFrame:target display:YES animate:NO];
        self.isFullscreen = NO;
    } else {
        /* 2. Switch to Instant Borderless Fullscreen (Direct Hardware Scanout) */
        self.savedWindowedRect = self.frame;
        NSScreen *scr = [self screen] ? [self screen] : [NSScreen mainScreen];
        NSRect screenRect = [scr frame];

        self.styleMask = NSWindowStyleMaskBorderless;
        [self setLevel:NSMainMenuWindowLevel + 1]; /* Bypasses menu bar and dock */
        [self setFrame:screenRect display:YES animate:NO];
        self.isFullscreen = YES;
    }
    [self makeKeyAndOrderFront:nil];
}

- (void)keyDown:(NSEvent *)event {
    if ([event isARepeat]) return;

    NSString *chars = [[event charactersIgnoringModifiers] lowercaseString];
    if ([chars length] == 0) return;
    unichar c = [chars characterAtIndex:0];

    switch (c) {
        /* Fullscreen / Windowed Toggle */
        case 'f':
            [self toggleFullscreenMode];
            break;

        /* Roll & Pitch (WASD) */
        case KEY_ROLL_LEFT:  g_keys.roll_left  = true; break;
        case KEY_ROLL_RIGHT: g_keys.roll_right = true; break;
        case KEY_PITCH_DOWN: g_keys.pitch_down = true; break;
        case KEY_PITCH_UP:   g_keys.pitch_up   = true; break;

        /* Throttle (Up / Down Arrows) */
        case NSUpArrowFunctionKey:   g_keys.throttle_up   = true; break;
        case NSDownArrowFunctionKey: g_keys.throttle_down = true; break;

        /* Yaw (Left / Right Arrows) */
        case NSLeftArrowFunctionKey:  g_keys.yaw_left  = true; break;
        case NSRightArrowFunctionKey: g_keys.yaw_right = true; break;

        /* Switches & Helpers */
        case KEY_SW1: ctrl_toggle_sw(1); break;
        case KEY_SW2: ctrl_toggle_sw(2); break;
        case KEY_SW3: ctrl_toggle_sw(3); break;
        case KEY_CENTER: ctrl_center(); break;

        /* Quit */
        case KEY_QUIT:
        case 27: /* ESC */
            g_quit = 1;
            [NSApp stop:nil];
            break;

        default:
            [super keyDown:event];
            break;
    }
}

- (void)keyUp:(NSEvent *)event {
    NSString *chars = [[event charactersIgnoringModifiers] lowercaseString];
    if ([chars length] == 0) return;
    unichar c = [chars characterAtIndex:0];

    switch (c) {
        case KEY_ROLL_LEFT:  g_keys.roll_left  = false; break;
        case KEY_ROLL_RIGHT: g_keys.roll_right = false; break;
        case KEY_PITCH_DOWN: g_keys.pitch_down = false; break;
        case KEY_PITCH_UP:   g_keys.pitch_up   = false; break;

        case NSUpArrowFunctionKey:   g_keys.throttle_up   = false; break;
        case NSDownArrowFunctionKey: g_keys.throttle_down = false; break;

        case NSLeftArrowFunctionKey:  g_keys.yaw_left  = false; break;
        case NSRightArrowFunctionKey: g_keys.yaw_right = false; break;

        default:
            [super keyUp:event];
            break;
    }
}

@end


/* ================================================================
 * Main Entry Point: Cocoa Fullscreen App + DisplayLink + Threads
 * ================================================================ */
int main(int argc, char **argv) {
    if (argc > 1) g_serial_dev_path = argv[1];

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    signal(SIGPIPE, SIG_IGN);

    ctrl_init(&g_ctrl);

    @autoreleasepool {
        /* 1. Initialize native macOS Cocoa Application */
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        /* 2. Initialize Apple Silicon Metal GPU Device */
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || metal_pipeline_init(device) != 0) {
            fprintf(stderr, "[fatal] Metal initialization failed\n");
            return -1;
        }

        /* 3. Start in Borderless Fullscreen (Direct Hardware Scanout, Lowest Latency) */
        NSScreen *mainScreen = [NSScreen mainScreen];
        NSRect screenRect = [mainScreen frame];

        FullscreenWindow *window = [[FullscreenWindow alloc] initWithContentRect:screenRect
            styleMask:NSWindowStyleMaskBorderless
            backing:NSBackingStoreBuffered
            defer:NO];

        [window setLevel:NSMainMenuWindowLevel + 1]; /* Above Dock and Menu Bar */
        [window setOpaque:YES];
        [window setHidesOnDeactivate:NO];
        window.isFullscreen = YES;
        /* Default windowed size to fall back to when pressing 'F' */
        window.savedWindowedRect = NSMakeRect(100, 100, 1280, 720);

        /* 4. Create Video View & HUD (Origin pinned strictly to (0,0)) */
        FPVMetalView   *metalView = [[FPVMetalView alloc] initWithFrame:screenRect];
        /* Use metalView.bounds so HUD origin is strictly (0, 0) without any pixel offset */
        HUDOverlayView *hudView   = [[HUDOverlayView alloc] initWithFrame:metalView.bounds];

        metalView.autoresizesSubviews = YES;
        metalView.autoresizingMask    = NSViewWidthSizable | NSViewHeightSizable;
        hudView.autoresizingMask      = NSViewWidthSizable | NSViewHeightSizable;

        [metalView addSubview:hudView];
        [window setContentView:metalView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        /* Both layers dynamically scale when toggling between window and fullscreen */
        metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        hudView.autoresizingMask   = NSViewWidthSizable | NSViewHeightSizable;

        [metalView addSubview:hudView];
        [window setContentView:metalView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];
        
        /* Stack the transparent HUD on top of the Metal video view */
        [metalView addSubview:hudView];
        [window setContentView:metalView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        /* 5. Modern macOS 15 DisplayLink synchronized to display refresh */
        CADisplayLink *displayLink = [metalView displayLinkWithTarget:metalView selector:@selector(displayLinkDidFire:)];
        [displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];

        /* 6. 60Hz Timer to refresh the Telemetry HUD */
        [NSTimer scheduledTimerWithTimeInterval:0.016 repeats:YES block:^(NSTimer * _Nonnull timer) {
            (void)timer;
            [hudView setNeedsDisplay:YES];
        }];

        /* 7. Launch Background Video & Serial Communication Threads */
        pthread_t video_tid, serial_tid;
        pthread_create(&video_tid, NULL, video_thread_fn, NULL);
        pthread_create(&serial_tid, NULL, serial_thread_fn, NULL);

        fprintf(stderr,
            "\n========================================\n"
            " Low-Latency Metal FPV Pipeline Running \n"
            "========================================\n"
            "Controls:\n"
            "  Roll:      A / D (smooth bank)\n"
            "  Pitch:     W / S (smooth climb/dive)\n"
            "  Throttle:  Up / Down Arrows (smooth ramp)\n"
            "  Yaw:       Left / Right Arrows\n"
            "  Switches:  1 / 2 / 3 (toggle)\n"
            "  Center:    C\n"
            "  Quit:      Q or Esc\n"
            "Serial Link: %s\n\n", g_serial_dev_path);

        /* 8. Run macOS Event Loop (main thread pumps UI & keyboard events) */
        [NSApp run];

        /* 9. Clean Shutdown */
        g_quit = 1;
        [displayLink invalidate];
        pthread_join(video_tid, NULL);
        pthread_join(serial_tid, NULL);
    }

    fprintf(stderr, "\n>>> System shutdown cleanly.\n");
    return 0;
}