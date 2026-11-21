/*
 * High-Speed USB Bulk Video Receiver (VRX) + Telemetry HUD
 * Ultra-Low Latency Pipeline with Non-blocking Metal Presentation
 * Pure P-frame / Immediate Access Unit Ingest
 */

#define _DARWIN_C_SOURCE

#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>
#import <QuartzCore/QuartzCore.h>
#include <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <math.h>
#include <pthread.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <libusb-1.0/libusb.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/mem.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * HUD Color Configuration
 *   0 = Phosphor Green  (R: 0.00, G: 1.00, B: 0.40)
 *   1 = Cyber Magenta   (R: 1.00, G: 0.15, B: 0.85)
 * To customize further, edit HUD_PRIMARY_R / G / B in render_hud_to_texture().
 * ================================================================ */
#define HUD_COLOR_THEME         0

/* ---------------- Video (vendor bulk) config ---------------- */
#define TARGET_VID              0x2207
#define TARGET_PID              0x0011
#define BUFFER_SIZE             (64 * 1024)
#define MAX_AU_PAYLOAD          (2 * 1024 * 1024)

/* ---------------- ArduPilot CRSF link config ---------------- */
#define ARDUPILOT_SERIAL_DEV_DEFAULT "/dev/cu.usbmodem2103"
#define ARDUPILOT_BAUD          B115200
#define CRSF_TX_HZ              100
#define SERIAL_RETRY_MS         2000

#define CRSF_SYNC_FC            0xC8
#define CRSF_MAX_FRAME_LEN      64
#define CRSF_FRAMETYPE_GPS      0x02
#define CRSF_FRAMETYPE_BATTERY  0x08
#define CRSF_FRAMETYPE_LINK_STAT 0x14
#define CRSF_FRAMETYPE_RC_CHANNELS 0x16
#define CRSF_FRAMETYPE_ATTITUDE 0x1E
#define CRSF_FRAMETYPE_FLIGHT_MODE 0x21
#define CRSF_FRAMETYPE_BARO_VARIO 0x09

#define DECAY_TAU_SEC           0.35f
#define THROTTLE_RATE           0.50f

#define SHOW_LATENCY_TIMER      1



/* ---------------- Embedded Metal Shader Source -------------- */
static const char *kMetalShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct VertexOut {\n"
"    float4 position [[position]];\n"
"    float2 texCoords;\n"
"};\n"
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
"fragment float4 yuvFragmentShader(VertexOut in [[stage_in]],\n"
"                                  texture2d<float> yTexture   [[texture(0)]],\n"
"                                  texture2d<float> uTexture   [[texture(1)]],\n"
"                                  texture2d<float> vTexture   [[texture(2)]],\n"
"                                  texture2d<float> hudTexture [[texture(3)]]) {\n"
"    constexpr sampler sVideo(address::clamp_to_edge, filter::linear);\n"
"    constexpr sampler sHud  (address::clamp_to_edge, filter::nearest);\n"
"    float y = yTexture.sample(sVideo, in.texCoords).r;\n"
"    float u = uTexture.sample(sVideo, in.texCoords).r - 0.5;\n"
"    float v = vTexture.sample(sVideo, in.texCoords).r - 0.5;\n"
"    float3 rgb;\n"
"    rgb.r = y + 1.5748 * v;\n"
"    rgb.g = y - 0.1873 * u - 0.4681 * v;\n"
"    rgb.b = y + 1.8556 * u;\n"
"    float4 video = float4(clamp(rgb, 0.0, 1.0), 1.0);\n"
"\n"
"    float4 hud = hudTexture.sample(sHud, in.texCoords);\n"
"    float3 blended = video.rgb * (1.0 - hud.a) + hud.rgb;\n"
"    return float4(blended, 1.0);\n"
"}\n";

@interface FPVMetalView : NSView
@property (nonatomic, strong) CAMetalLayer *metalLayer;
- (void)renderFrame;
@end

static FPVMetalView *g_metal_view = nil;

/* --- Metal Pipeline State --- */
typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        commandQueue;
    id<MTLRenderPipelineState> pipelineState;
    id<MTLTexture>             yTex[3];
    id<MTLTexture>             uTex[3];
    id<MTLTexture>             vTex[3];
    id<MTLTexture>             hudTex[2];
    int                        activeHudIdx;
    int                        activeIdx;
    int                        width;
    int                        height;
    pthread_mutex_t            tex_lock;
    pthread_mutex_t            render_lock;
    bool                       has_frame;
} metal_video_pipeline_t;

static metal_video_pipeline_t g_metal;

static int metal_pipeline_init(id<MTLDevice> device) {
    memset(&g_metal, 0, sizeof(g_metal));
    g_metal.device = device;
    g_metal.commandQueue = [device newCommandQueue];
    pthread_mutex_init(&g_metal.tex_lock, NULL);
    pthread_mutex_init(&g_metal.render_lock, NULL);

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
    return g_metal.pipelineState ? 0 : -1;
}

static void ensure_textures(int w, int h) {
    if (g_metal.width != w || g_metal.height != h || g_metal.yTex[0] == nil) {
        g_metal.width = w;
        g_metal.height = h;

        for (int i = 0; i < 3; i++) {
            MTLTextureDescriptor *yd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:w height:h mipmapped:NO];
            yd.usage = MTLTextureUsageShaderRead;
            yd.storageMode = MTLStorageModeShared;
            g_metal.yTex[i] = [g_metal.device newTextureWithDescriptor:yd];

            MTLTextureDescriptor *uvd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:w / 2 height:h / 2 mipmapped:NO];
            uvd.usage = MTLTextureUsageShaderRead;
            uvd.storageMode = MTLStorageModeShared;
            g_metal.uTex[i] = [g_metal.device newTextureWithDescriptor:uvd];
            g_metal.vTex[i] = [g_metal.device newTextureWithDescriptor:uvd];
        }
    }
}

static void metal_init_black_textures(void) {
    int w = 1280, h = 720;
    ensure_textures(w, h);

    uint8_t *black_y = (uint8_t *)calloc(w * h, 1);
    uint8_t *gray_uv = (uint8_t *)malloc((w / 2) * (h / 2));
    memset(gray_uv, 128, (w / 2) * (h / 2));

    pthread_mutex_lock(&g_metal.tex_lock);
    for (int i = 0; i < 3; i++) {
        [g_metal.yTex[i] replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:black_y bytesPerRow:w];
        [g_metal.uTex[i] replaceRegion:MTLRegionMake2D(0, 0, w / 2, h / 2) mipmapLevel:0 withBytes:gray_uv bytesPerRow:w / 2];
        [g_metal.vTex[i] replaceRegion:MTLRegionMake2D(0, 0, w / 2, h / 2) mipmapLevel:0 withBytes:gray_uv bytesPerRow:w / 2];
    }
    g_metal.has_frame = true;
    pthread_mutex_unlock(&g_metal.tex_lock);

    free(black_y);
    free(gray_uv);
}

static void metal_upload_yuv420p_frame(AVFrame *frame) {
    if (frame->format != AV_PIX_FMT_YUV420P && frame->format != AV_PIX_FMT_YUVJ420P) return;

    pthread_mutex_lock(&g_metal.tex_lock);
    int w = frame->width;
    int h = frame->height;
    ensure_textures(w, h);

    int writeIdx = (g_metal.activeIdx + 1) % 3;
    [g_metal.yTex[writeIdx] replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:frame->data[0] bytesPerRow:frame->linesize[0]];
    [g_metal.uTex[writeIdx] replaceRegion:MTLRegionMake2D(0, 0, w / 2, h / 2) mipmapLevel:0 withBytes:frame->data[1] bytesPerRow:frame->linesize[1]];
    [g_metal.vTex[writeIdx] replaceRegion:MTLRegionMake2D(0, 0, w / 2, h / 2) mipmapLevel:0 withBytes:frame->data[2] bytesPerRow:frame->linesize[2]];

    g_metal.activeIdx = writeIdx;
    g_metal.has_frame = true;
    pthread_mutex_unlock(&g_metal.tex_lock);
}

/* ================================================================
 * Shared State & Telemetry Structs
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

/* --- Pilot Controls --- */
typedef struct {
    float roll, pitch, yaw;
    float throttle;
    float cam_pan;
    float cam_tilt;
    bool  armed;
    bool  rtl;
    int   flight_mode;
} ctrl_state_t;

static ctrl_state_t    g_ctrl = { .flight_mode = 1 };
static pthread_mutex_t g_ctrl_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    bool roll_left, roll_right;
    bool pitch_down, pitch_up;
    bool yaw_left, yaw_right;
    bool throttle_up, throttle_down;
    bool cam_pan_left, cam_pan_right;
    bool cam_tilt_up, cam_tilt_down;
} sim_keys_t;

static sim_keys_t g_keys = {0};

/* --- Telemetry Snapshot Data --- */
typedef struct {
    float    pitch_deg;
    float    roll_deg;
    float    yaw_deg;
    float    voltage;
    float    current;
    uint32_t mah;
    uint8_t  battery_pct;
    uint8_t  rssi;
    uint8_t  lq;
    double   latitude;
    double   longitude;
    double   home_lat;
    double   home_lon;
    bool     has_home;
    float    altitude_m;
    float    gps_altitude_m;
    float    groundspeed_kmh;
    float    gps_heading_deg;
    uint8_t  sats;
    char     flight_mode[16];
} telem_data_t;

static telem_data_t    g_telem = { .flight_mode = "MANUAL" };
static pthread_mutex_t g_telem_lock = PTHREAD_MUTEX_INITIALIZER;

static telem_data_t telem_get_snapshot(void) {
    pthread_mutex_lock(&g_telem_lock);
    telem_data_t snap = g_telem;
    pthread_mutex_unlock(&g_telem_lock);
    return snap;
}

static void ctrl_center(void) {
    pthread_mutex_lock(&g_ctrl_lock);
    g_ctrl.roll = g_ctrl.pitch = g_ctrl.yaw = 0.0f;
    g_ctrl.cam_pan = g_ctrl.cam_tilt = 0.0f;
    pthread_mutex_unlock(&g_ctrl_lock);
}

/* ================================================================
 * CRSF Protocol Handling
 * ================================================================ */
static uint16_t encode_axis(float v) {
    int raw = 992 + (int)(v * 820.0f);
    return (uint16_t)clampf((float)raw, 172.0f, 1811.0f);
}

static uint16_t encode_throttle(float v) {
    int raw = 172 + (int)(v * (1811 - 172));
    return (uint16_t)clampf((float)raw, 172.0f, 1811.0f);
}

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

static int crsf_build_rc_frame(ctrl_state_t s, uint8_t out[26]) {
    uint16_t ch[16];
    ch[0] = encode_axis(s.roll);
    ch[1] = encode_axis(s.pitch);
    ch[2] = encode_throttle(s.throttle);
    ch[3] = encode_axis(s.yaw);
    ch[4] = s.armed ? 1811 : 172;
    ch[5] = encode_axis(s.cam_pan);
    ch[6] = s.rtl ? 1811 : 172;
    ch[7] = (s.flight_mode == 1) ? 172 : ((s.flight_mode == 2) ? 500 : 827);
    ch[8] = encode_axis(s.cam_tilt);
    for (int i = 9; i < 16; i++) ch[i] = 992;

    out[0] = CRSF_SYNC_FC;
    out[1] = 22 + 2;
    out[2] = CRSF_FRAMETYPE_RC_CHANNELS;
    crsf_pack_channels(ch, &out[3]);
    out[25] = crsf_crc8(&out[2], 1 + 22);
    return 26;
}


static void handle_crsf_frame(const uint8_t *f, int len) {
    uint8_t type = f[0];
    const uint8_t *payload = &f[1];
    int payload_len = len - 2;

    static double last_p = 0;
    if (now_seconds() - last_p > 0.5) {
        last_p = now_seconds();
        printf("[mac] CRSF frame received: Type 0x%02X, len=%d\n", type, len);
    }

    pthread_mutex_lock(&g_telem_lock);
    switch (type) {
    case CRSF_FRAMETYPE_ATTITUDE:
        if (payload_len >= 6) {
            g_telem.pitch_deg = (int16_t)((payload[0] << 8) | payload[1]) / 10000.0f * 180.0f / (float)M_PI;
            g_telem.roll_deg  = (int16_t)((payload[2] << 8) | payload[3]) / 10000.0f * 180.0f / (float)M_PI;
            g_telem.yaw_deg   = (int16_t)((payload[4] << 8) | payload[5]) / 10000.0f * 180.0f / (float)M_PI;
        }
        break;
    case CRSF_FRAMETYPE_BATTERY:
        if (payload_len >= 8) {
            g_telem.voltage     = (uint16_t)((payload[0] << 8) | payload[1]) / 10.0f;
            g_telem.current     = (uint16_t)((payload[2] << 8) | payload[3]) / 10.0f;
            g_telem.mah         = ((uint32_t)payload[4] << 16) | ((uint32_t)payload[5] << 8) | payload[6];
            g_telem.battery_pct = payload[7];
        }
        break;
    case CRSF_FRAMETYPE_GPS:
        if (payload_len >= 15) {
            int32_t lat = (int32_t)((payload[0] << 24) | (payload[1] << 16) | (payload[2] << 8) | payload[3]);
            int32_t lon = (int32_t)((payload[4] << 24) | (payload[5] << 16) | (payload[6] << 8) | payload[7]);
            g_telem.latitude        = (double)lat / 1e7;
            g_telem.longitude       = (double)lon / 1e7;
            g_telem.groundspeed_kmh = (uint16_t)((payload[8] << 8) | payload[9]) / 10.0f;
            g_telem.gps_heading_deg = (uint16_t)((payload[10] << 8) | payload[11]) / 100.0f;
            g_telem.gps_altitude_m  = (uint16_t)((payload[12] << 8) | payload[13]) - 1000.0f;
            g_telem.sats            = payload[14];

            if (!g_telem.has_home && g_telem.sats >= 5 &&
                (fabs(g_telem.latitude) > 0.001 || fabs(g_telem.longitude) > 0.001)) {
                g_telem.home_lat = g_telem.latitude;
                g_telem.home_lon = g_telem.longitude;
                g_telem.has_home = true;
            }
        }
        break;
    case CRSF_FRAMETYPE_FLIGHT_MODE: {
        int n = payload_len < 15 ? payload_len : 15;
        if (n > 0) {
            memcpy(g_telem.flight_mode, payload, n);
            g_telem.flight_mode[n] = '\0';
        }
        break;
    }
    case CRSF_FRAMETYPE_LINK_STAT:
        if (payload_len >= 4) {
            g_telem.rssi = payload[0];
            g_telem.lq   = payload[2];
        }
        break;
    case CRSF_FRAMETYPE_BARO_VARIO:
        if (payload_len >= 2) {
            uint16_t raw_alt = ((uint16_t)payload[0] << 8) | payload[1];
            if (raw_alt & 0x8000) {
                g_telem.altitude_m = (float)(raw_alt & 0x7FFF);
            } else {
                g_telem.altitude_m = ((int32_t)raw_alt - 10000) / 10.0f;
            }
        }
        break;
    }
    pthread_mutex_unlock(&g_telem_lock);
}

typedef struct {
    int state, pos, len;
    uint8_t buf[CRSF_MAX_FRAME_LEN];
} crsf_parser_t;

static void crsf_parser_feed(crsf_parser_t *p, uint8_t byte) {
    enum { WAIT_SYNC, WAIT_LEN, WAIT_DATA };
    switch (p->state) {
    case WAIT_SYNC:
        if (byte == 0xC8 || byte == 0xEE || byte == 0xEA || byte == 0xEC) {
            p->pos = 0; p->state = WAIT_LEN;
        }
        break;
    case WAIT_LEN:
        if (byte < 2 || byte > CRSF_MAX_FRAME_LEN - 2) { p->state = WAIT_SYNC; break; }
        p->len = byte; p->pos = 0; p->state = WAIT_DATA;
        break;
    case WAIT_DATA:
        p->buf[p->pos++] = byte;
        if (p->pos == p->len) {
            if (crsf_crc8(p->buf, p->len - 1) == p->buf[p->len - 1]) {
                handle_crsf_frame(p->buf, p->len);
            }
            p->state = WAIT_SYNC;
        }
        break;
    }
}

/* ================================================================
 * Video Decoder Context (Zero-Latency HEVC)
 * ================================================================ */
typedef struct {
    const AVCodec        *codec;
    AVCodecContext       *ctx;
    AVFrame              *frame;
    AVPacket             *pkt;
} hevc_decoder_t;

#define VTX_HDR_CACHE_PATH "/tmp/vtx_hevc_hdr.bin"
static uint8_t g_cached_hdr[512];
static size_t  g_cached_hdr_len = 0;

static const uint8_t k_default_hevc_hdr[82] = {
    0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00,
    0x80, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x5D, 0xBC, 0x09, 0x00, 0x00, 0x00, 0x01, 0x42,
    0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00,
    0x5D, 0xA0, 0x02, 0x80, 0x80, 0x2D, 0x16, 0x36, 0xFB, 0xBC, 0xA6, 0x01, 0x00, 0x00, 0x03, 0x00,
    0x01, 0x00, 0x00, 0x03, 0x00, 0x3C, 0x08, 0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xC0, 0xF3, 0xC0,
    0x4D, 0x90
};

static void load_cached_hdr(void) {
    if (g_cached_hdr_len > 0) return;
    FILE *f = fopen(VTX_HDR_CACHE_PATH, "rb");
    if (f) {
        size_t n = fread(g_cached_hdr, 1, sizeof(g_cached_hdr), f);
        fclose(f);
        if (n > 6) {
            g_cached_hdr_len = n;
            fprintf(stderr, "[decode] Loaded cached VPS/SPS/PPS (%zu bytes) from %s\n",
                    g_cached_hdr_len, VTX_HDR_CACHE_PATH);
            return;
        }
    }
    memcpy(g_cached_hdr, k_default_hevc_hdr, sizeof(k_default_hevc_hdr));
    g_cached_hdr_len = sizeof(k_default_hevc_hdr);
    fprintf(stderr, "[decode] Pre-seeded default 720p60 VPS/SPS/PPS (%zu bytes)\n", g_cached_hdr_len);
}

static size_t extract_hevc_param_sets(const uint8_t *buf, size_t len) {
    if (len < 6) return 0;
    if (buf[0] != 0 || buf[1] != 0 || buf[2] != 0 || buf[3] != 1) return 0;
    if (((buf[4] >> 1) & 0x3F) != 32) return 0; /* Must start with VPS (32) */

    bool seen_sps = false, seen_pps = false;
    for (size_t i = 6; i + 4 < len; i++) {
        if (buf[i] == 0 && buf[i+1] == 0 && buf[i+2] == 0 && buf[i+3] == 1) {
            uint8_t next_type = (buf[i + 4] >> 1) & 0x3F;
            if (next_type == 33) seen_sps = true;
            else if (next_type == 34) seen_pps = true;
            else if (next_type < 32) return (seen_sps && seen_pps) ? i : 0;
            i += 3;
        }
    }
    return 0;
}

static void maybe_latch_hevc_hdr(const uint8_t *pkt_data, size_t pkt_len) {
    size_t hdr_len = extract_hevc_param_sets(pkt_data, pkt_len);
    if (hdr_len == 0 || hdr_len > sizeof(g_cached_hdr)) return;

    if (hdr_len != g_cached_hdr_len || memcmp(g_cached_hdr, pkt_data, hdr_len) != 0) {
        memcpy(g_cached_hdr, pkt_data, hdr_len);
        g_cached_hdr_len = hdr_len;

        FILE *f = fopen(VTX_HDR_CACHE_PATH, "wb");
        if (f) {
            fwrite(g_cached_hdr, 1, g_cached_hdr_len, f);
            fclose(f);
        }

        fprintf(stderr, "[decode] Latched VPS/SPS/PPS (%zu bytes):", g_cached_hdr_len);
        for (size_t i = 0; i < g_cached_hdr_len; i++) {
            fprintf(stderr, " 0x%02X", g_cached_hdr[i]);
        }
        fprintf(stderr, "\n");
    }
}

static int hevc_decoder_init(hevc_decoder_t *dec) {
    memset(dec, 0, sizeof(*dec));
    dec->codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    if (!dec->codec) return -1;

    av_log_set_level(AV_LOG_INFO);
    dec->ctx    = avcodec_alloc_context3(dec->codec);
    if (!dec->ctx) return -1;

    dec->ctx->flags  |= AV_CODEC_FLAG_LOW_DELAY;
    dec->ctx->flags2 |= AV_CODEC_FLAG2_FAST;
    dec->ctx->flags  |= AV_CODEC_FLAG_OUTPUT_CORRUPT;
    dec->ctx->flags2 |= AV_CODEC_FLAG2_SHOW_ALL;

    dec->ctx->thread_type  = FF_THREAD_SLICE;
    dec->ctx->thread_count = 1;
    dec->ctx->max_b_frames = 0;
    dec->ctx->has_b_frames = 0;
    dec->ctx->delay        = 0;

    load_cached_hdr();
    if (g_cached_hdr_len > 0) {
        dec->ctx->extradata = (uint8_t *)av_mallocz(g_cached_hdr_len + AV_INPUT_BUFFER_PADDING_SIZE);
        if (dec->ctx->extradata) {
            memcpy(dec->ctx->extradata, g_cached_hdr, g_cached_hdr_len);
            dec->ctx->extradata_size = (int)g_cached_hdr_len;
        }
    }

    AVDictionary *opts = NULL;
    av_dict_set(&opts, "tune", "zerolatency", 0);
    if (avcodec_open2(dec->ctx, dec->codec, &opts) < 0) return -1;
    av_dict_free(&opts);

    dec->frame = av_frame_alloc();
    dec->pkt   = av_packet_alloc();
    return (dec->frame && dec->pkt) ? 0 : -1;
}

static void hevc_decoder_destroy(hevc_decoder_t *dec) {
    if (dec->frame)  av_frame_free(&dec->frame);
    if (dec->pkt)    av_packet_free(&dec->pkt);
    if (dec->ctx)    avcodec_free_context(&dec->ctx);
}

/* ---------------- Queue Configuration ---------------- */
#define QUEUE_CAP          32
#define QUEUE_SLOT_SIZE    BUFFER_SIZE

typedef struct {
    uint8_t  data[QUEUE_SLOT_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
    uint32_t len;
} au_slot_t;

static au_slot_t       g_queue[QUEUE_CAP];
static int             g_q_head = 0;
static int             g_q_tail = 0;
static int             g_q_count = 0;
static pthread_mutex_t g_q_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_q_cond = PTHREAD_COND_INITIALIZER;

static _Atomic double g_last_video_render_time = 0.0;

static void *decode_render_thread_fn(void *arg) {
    (void)arg;
    hevc_decoder_t dec;
    if (hevc_decoder_init(&dec) != 0) {
        fprintf(stderr, "[decode] Failed to initialize HEVC decoder\n");
        return NULL;
    }
    fprintf(stderr, "[decode] Decoder & Render worker thread active (zero-wait framer)\n");

    uint8_t *au_buf = (uint8_t *)malloc(MAX_AU_PAYLOAD + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!au_buf) {
        hevc_decoder_destroy(&dec);
        return NULL;
    }

    bool   synced    = false;
    size_t au_len    = 4;
    size_t nal_start = 4;
    int    zero_run  = 0;

    while (!g_quit) {
        pthread_mutex_lock(&g_q_lock);
        while (g_q_count == 0 && !g_quit) {
            pthread_cond_wait(&g_q_cond, &g_q_lock);
        }
        if (g_quit) {
            pthread_mutex_unlock(&g_q_lock);
            break;
        }

        int pop_idx = g_q_tail;
        g_q_tail = (g_q_tail + 1) % QUEUE_CAP;
        g_q_count--;
        pthread_mutex_unlock(&g_q_lock);

        const uint8_t *data = g_queue[pop_idx].data;
        uint32_t size = g_queue[pop_idx].len;

        for (uint32_t i = 0; i < size && !g_quit; i++) {
            uint8_t b = data[i];

            /* Wait for first Annex B start code before recording bytes */
            if (!synced) {
                if (b == 0x00) {
                    zero_run++;
                } else if (b == 0x01 && zero_run >= 2) {
                    synced = true;
                    au_buf[0] = 0x00; au_buf[1] = 0x00; au_buf[2] = 0x00; au_buf[3] = 0x01;
                    au_len    = 4;
                    nal_start = 4;
                    zero_run  = 0;
                } else {
                    zero_run = 0;
                }
                continue;
            }

            if (au_len >= MAX_AU_PAYLOAD) {
                fprintf(stderr, "[decode] AU overflow (>%d bytes), resyncing\n", MAX_AU_PAYLOAD);
                synced    = false;
                au_len    = 4;
                nal_start = 4;
                zero_run  = 0;
                continue;
            }

            au_buf[au_len++] = b;

            if (b == 0x00) {
                zero_run++;
            } else if (b == 0x01 && zero_run >= 2) {
                size_t sc_len  = (zero_run >= 3) ? 4 : 3;
                size_t nal_end = au_len - sc_len;
                zero_run = 0;

                size_t nal_len   = (nal_end > nal_start) ? (nal_end - nal_start) : 0;
                if (nal_len == 0) {
                    au_len = nal_start;
                    continue;
                }

                uint8_t nal_type = (au_buf[nal_start] >> 1) & 0x3F;

                /* Keep accumulating non-VCL NALs (VPS=32, SPS=33, PPS=34, AUD=35, SEI=39/40) */
                if (nal_type >= 32 && nal_len <= 512) {
                    if (nal_end + 4 <= MAX_AU_PAYLOAD) {
                        au_buf[nal_end + 0] = 0x00;
                        au_buf[nal_end + 1] = 0x00;
                        au_buf[nal_end + 2] = 0x00;
                        au_buf[nal_end + 3] = 0x01;
                        au_len    = nal_end + 4;
                        nal_start = au_len;
                        continue;
                    }
                }

                /* VCL slice complete -> decode immediately */
                if (nal_end > 4) {
                    memset(au_buf + nal_end, 0, AV_INPUT_BUFFER_PADDING_SIZE);
                    maybe_latch_hevc_hdr(au_buf, nal_end);

                    dec.pkt->data = au_buf;
                    dec.pkt->size = (int)nal_end;

                    if (avcodec_send_packet(dec.ctx, dec.pkt) == 0) {
                        while (avcodec_receive_frame(dec.ctx, dec.frame) == 0) {
                            metal_upload_yuv420p_frame(dec.frame);
                            av_frame_unref(dec.frame);

                            if (g_metal_view) {
                                atomic_store(&g_last_video_render_time, now_seconds());
                                [g_metal_view renderFrame];
                            }
                        }
                    }
                    dec.pkt->data = NULL;
                    dec.pkt->size = 0;
                }

                au_buf[0] = 0x00; au_buf[1] = 0x00; au_buf[2] = 0x00; au_buf[3] = 0x01;
                au_len    = 4;
                nal_start = 4;
            } else {
                zero_run = 0;
            }
        }
    }

    free(au_buf);
    hevc_decoder_destroy(&dec);
    return NULL;
}

/* ================================================================
 * Video Thread: High-Speed Dedicated USB Ingest
 * ================================================================ */
static bool find_bulk_in_endpoint(libusb_device *dev, int *out_intf, uint8_t *out_ep) {
    struct libusb_config_descriptor *config = NULL;
    if (libusb_get_active_config_descriptor(dev, &config) < 0) return false;

    bool found = false;
    for (int i = 0; i < config->bNumInterfaces && !found; i++) {
        const struct libusb_interface *intf = &config->interface[i];
        for (int a = 0; a < intf->num_altsetting && !found; a++) {
            const struct libusb_interface_descriptor *alt = &intf->altsetting[a];
            if (alt->bInterfaceClass != LIBUSB_CLASS_VENDOR_SPEC) continue;

            for (int e = 0; e < alt->bNumEndpoints; e++) {
                const struct libusb_endpoint_descriptor *ep = &alt->endpoint[e];
                bool is_in = (ep->bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_IN;
                bool is_bulk = (ep->bmAttributes & LIBUSB_TRANSFER_TYPE_MASK) == LIBUSB_TRANSFER_TYPE_BULK;
                if (is_in && is_bulk) {
                    *out_intf = alt->bInterfaceNumber;
                    *out_ep   = ep->bEndpointAddress;
                    found = true;
                    break;
                }
            }
        }
    }
    libusb_free_config_descriptor(config);
    return found;
}

static void *video_thread_fn(void *arg) {
    (void)arg;
    libusb_context *ctx = NULL;
    if (libusb_init(&ctx) < 0) return NULL;

    uint8_t *usb_buffer = malloc(BUFFER_SIZE);

    pthread_t dec_tid;
    pthread_create(&dec_tid, NULL, decode_render_thread_fn, NULL);

    while (!g_quit) {
        libusb_device_handle *dev = libusb_open_device_with_vid_pid(ctx, TARGET_VID, TARGET_PID);
        if (!dev) {
            usleep(200 * 1000);
            continue;
        }

        int target_intf = -1;
        uint8_t ep_in_addr = 0;
        if (!find_bulk_in_endpoint(libusb_get_device(dev), &target_intf, &ep_in_addr)) {
            libusb_close(dev);
            usleep(500 * 1000);
            continue;
        }

        libusb_detach_kernel_driver(dev, target_intf);
        if (libusb_claim_interface(dev, target_intf) < 0) {
            libusb_close(dev);
            usleep(500 * 1000);
            continue;
        }

        fprintf(stderr, "[video] Connected! EP 0x%02X streaming\n", ep_in_addr);

        while (!g_quit) {
            int actual = 0;
            int ret = libusb_bulk_transfer(dev, ep_in_addr, usb_buffer, BUFFER_SIZE, &actual, 100);

            static double last_print = 0;
            double now = now_seconds();
            if (now - last_print >= 1.0) {
                last_print = now;
                fprintf(stderr, "[libusb] bulk_transfer ret=%d (%s), actual=%d bytes\n",
                        ret, libusb_error_name(ret), actual);
            }

            if (ret == 0 && actual > 0) {
                pthread_mutex_lock(&g_q_lock);
                if (g_q_count >= 20) {
                    int dropped = 0;
                    while (g_q_count >= 20) {
                        g_q_tail = (g_q_tail + 1) % QUEUE_CAP;
                        g_q_count--;
                        dropped++;
                    }
                    fprintf(stderr, "[video] WARNING: decode queue overflow, dropped %d chunk(s)\n", dropped);
                }

                uint32_t copy_len = (actual > QUEUE_SLOT_SIZE) ? QUEUE_SLOT_SIZE : (uint32_t)actual;
                memcpy(g_queue[g_q_head].data, usb_buffer, copy_len);
                g_queue[g_q_head].len = copy_len;
                g_q_head = (g_q_head + 1) % QUEUE_CAP;
                g_q_count++;
                pthread_cond_signal(&g_q_cond);
                pthread_mutex_unlock(&g_q_lock);
            } else if (ret == LIBUSB_ERROR_NO_DEVICE || ret == LIBUSB_ERROR_IO) {
                fprintf(stderr, "[video] Device disconnected\n");
                break;
            }
        }

        libusb_release_interface(dev, target_intf);
        libusb_close(dev);
    }

    g_quit = 1;
    pthread_mutex_lock(&g_q_lock);
    pthread_cond_broadcast(&g_q_cond);
    pthread_mutex_unlock(&g_q_lock);
    pthread_join(dec_tid, NULL);

    free(usb_buffer);
    libusb_exit(ctx);
    return NULL;
}

/* ================================================================
 * Serial Thread: 100 Hz Physics + CRSF I/O
 * ================================================================ */
static void *serial_thread_fn(void *arg) {
    (void)arg;
    crsf_parser_t parser = {0};
    int fd = -1;
    double last_tx_t = now_seconds();
    double last_physics_t = now_seconds();

    while (!g_quit) {
        if (fd < 0) {
            fd = open(g_serial_dev_path, O_RDWR | O_NOCTTY | O_NONBLOCK);
            if (fd < 0) {
                usleep(SERIAL_RETRY_MS * 1000);
                continue;
            }
            struct termios tty;
            memset(&tty, 0, sizeof(tty));
            if (tcgetattr(fd, &tty) == 0) {
                cfsetspeed(&tty, ARDUPILOT_BAUD);
                tty.c_cflag |= (CLOCAL | CREAD | CS8);
                tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
                tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
                tty.c_iflag &= ~(IXON | IXOFF | IXANY);
                tty.c_oflag &= ~OPOST;
                tty.c_cc[VMIN]  = 1;
                tty.c_cc[VTIME] = 0;
                tcsetattr(fd, TCSANOW, &tty);
                int modem_ctrl = TIOCM_DTR | TIOCM_RTS;
                ioctl(fd, TIOCMBIS, &modem_ctrl);
            }
            fprintf(stderr, "[serial] Connected to %s\n", g_serial_dev_path);
        }

        uint8_t rx_buf[256];
        ssize_t n;
        while ((n = read(fd, rx_buf, sizeof(rx_buf))) > 0) {
            for (ssize_t i = 0; i < n; i++) crsf_parser_feed(&parser, rx_buf[i]);
        }
        if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
            fprintf(stderr, "[serial] Disconnected from %s\n", g_serial_dev_path);
            close(fd);
            fd = -1;
            continue;
        }

        double t = now_seconds();
        double dt = t - last_physics_t;
        if (dt >= 0.01) {
            float blend = 1.0f - expf((float)(-dt / DECAY_TAU_SEC));

            pthread_mutex_lock(&g_ctrl_lock);
            float target_roll  = (g_keys.roll_right ? 1.0f : 0.0f) - (g_keys.roll_left ? 1.0f : 0.0f);
            float target_pitch = (g_keys.pitch_up   ? 1.0f : 0.0f) - (g_keys.pitch_down ? 1.0f : 0.0f);
            float target_yaw   = (g_keys.yaw_right  ? 1.0f : 0.0f) - (g_keys.yaw_left   ? 1.0f : 0.0f);

            g_ctrl.roll  += (target_roll  - g_ctrl.roll)  * blend;
            g_ctrl.pitch += (target_pitch - g_ctrl.pitch) * blend;
            g_ctrl.yaw   += (target_yaw   - g_ctrl.yaw)   * blend;

            if (g_keys.throttle_up)   g_ctrl.throttle = clampf(g_ctrl.throttle + THROTTLE_RATE * (float)dt, 0.0f, 1.0f);
            if (g_keys.throttle_down) g_ctrl.throttle = clampf(g_ctrl.throttle - THROTTLE_RATE * (float)dt, 0.0f, 1.0f);

            float cam_speed = 1.5f;
            if (g_keys.cam_pan_left)   g_ctrl.cam_pan  = clampf(g_ctrl.cam_pan  - cam_speed * (float)dt, -1.0f, 1.0f);
            if (g_keys.cam_pan_right)  g_ctrl.cam_pan  = clampf(g_ctrl.cam_pan  + cam_speed * (float)dt, -1.0f, 1.0f);
            if (g_keys.cam_tilt_down)  g_ctrl.cam_tilt = clampf(g_ctrl.cam_tilt - cam_speed * (float)dt, -1.0f, 1.0f);
            if (g_keys.cam_tilt_up)    g_ctrl.cam_tilt = clampf(g_ctrl.cam_tilt + cam_speed * (float)dt, -1.0f, 1.0f);
            pthread_mutex_unlock(&g_ctrl_lock);

            last_physics_t = t;
        }

        if (t - last_tx_t >= (1.0 / CRSF_TX_HZ)) {
            last_tx_t = t;
            pthread_mutex_lock(&g_ctrl_lock);
            ctrl_state_t current = g_ctrl;
            pthread_mutex_unlock(&g_ctrl_lock);

            uint8_t tx_frame[26];
            int len = crsf_build_rc_frame(current, tx_frame);
            ssize_t wn = write(fd, tx_frame, len);
            if (wn < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                fprintf(stderr, "[serial] Write error (%s), disconnecting\n", strerror(errno));
                close(fd);
                fd = -1;
                continue;
            }
        }

        usleep(1000);
    }

    if (fd >= 0) close(fd);
    return NULL;
}

/* ================================================================
 * Futuristic Retina HUD Renderer
 * ================================================================ */

static void hud_stroke_backed(CGContextRef ctx, CGColorRef fg, CGColorRef shadowCol, CGFloat width) {
    CGPathRef path = CGContextCopyPath(ctx);
    if (!path) return;
    CGContextSetStrokeColorWithColor(ctx, shadowCol);
    CGContextSetLineWidth(ctx, width + 1.6);
    CGContextStrokePath(ctx);

    CGContextAddPath(ctx, path);
    CGContextSetStrokeColorWithColor(ctx, fg);
    CGContextSetLineWidth(ctx, width);
    CGContextStrokePath(ctx);
    CGPathRelease(path);
}

static float wrap_delta_deg(float delta) {
    delta = fmodf(delta + 180.0f, 360.0f);
    if (delta < 0.0f) delta += 360.0f;
    return delta - 180.0f;
}

static void render_hud_to_texture(void) {
    if (!g_metal_view) return;

    @autoreleasepool {
    CGFloat scale = g_metal_view.window.backingScaleFactor ?: [[NSScreen mainScreen] backingScaleFactor];
    NSSize viewSize = g_metal_view.bounds.size;
    CGFloat w = viewSize.width;
    CGFloat h = viewSize.height;
    int pixel_w = (int)(w * scale);
    int pixel_h = (int)(h * scale);
    if (pixel_w <= 0 || pixel_h <= 0) return;

    static uint32_t *s_hud_pixels = NULL;
    static CGContextRef s_cached_ctx = NULL;
    static int s_buf_w = 0, s_buf_h = 0;

    if (s_buf_w != pixel_w || s_buf_h != pixel_h) {
        if (s_cached_ctx) { CGContextRelease(s_cached_ctx); s_cached_ctx = NULL; }
        free(s_hud_pixels);
        s_hud_pixels = (uint32_t *)calloc((size_t)pixel_w * pixel_h, sizeof(uint32_t));
        s_buf_w = pixel_w; s_buf_h = pixel_h;

        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        s_cached_ctx = CGBitmapContextCreate(s_hud_pixels, pixel_w, pixel_h, 8, pixel_w * 4, cs,
                                             kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(cs);

        CGContextSetAllowsAntialiasing(s_cached_ctx, true);
        CGContextSetShouldAntialias(s_cached_ctx, true);
        CGContextSetAllowsFontSmoothing(s_cached_ctx, true);
        CGContextSetShouldSmoothFonts(s_cached_ctx, true);

        MTLTextureDescriptor *hudd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                        width:pixel_w height:pixel_h mipmapped:NO];
        hudd.usage = MTLTextureUsageShaderRead;
        hudd.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex0 = [g_metal.device newTextureWithDescriptor:hudd];
        id<MTLTexture> tex1 = [g_metal.device newTextureWithDescriptor:hudd];

        pthread_mutex_lock(&g_metal.tex_lock);
        g_metal.hudTex[0] = tex0;
        g_metal.hudTex[1] = tex1;
        g_metal.activeHudIdx = 0;
        pthread_mutex_unlock(&g_metal.tex_lock);
    } else {
        memset(s_hud_pixels, 0, (size_t)pixel_w * pixel_h * sizeof(uint32_t));
    }

    CGContextRef ctx = s_cached_ctx;
    if (!ctx) return;

    static NSColor      *s_colPri    = nil;
    static CGColorRef    s_cgPri     = NULL;
    static CGColorRef    s_cgShad    = NULL;
    static CGColorRef    s_cgBg      = NULL;
    static NSDictionary *s_attrPri   = nil;
    static NSDictionary *s_attrSmall = nil;

    if (!s_colPri) {
#if HUD_COLOR_THEME == 1
        const CGFloat HUD_R = 1.00, HUD_G = 0.15, HUD_B = 0.85; /* Magenta */
#else
        const CGFloat HUD_R = 0.00, HUD_G = 1.00, HUD_B = 0.40; /* Green */
#endif
        s_colPri = [NSColor colorWithCalibratedRed:HUD_R green:HUD_G blue:HUD_B alpha:1.0];
        s_cgPri  = CGColorRetain(s_colPri.CGColor);
        s_cgShad = CGColorRetain([NSColor colorWithCalibratedWhite:0.0 alpha:0.75].CGColor);
        s_cgBg   = CGColorRetain([NSColor colorWithCalibratedRed:0.0 green:0.0 blue:0.0 alpha:0.35].CGColor);

        NSShadow *textShadow = [[NSShadow alloc] init];
        textShadow.shadowColor = [NSColor colorWithCalibratedWhite:0.0 alpha:0.9];
        textShadow.shadowOffset = NSMakeSize(1.0, -1.0);
        textShadow.shadowBlurRadius = 1.0;

        s_attrPri = @{
            NSFontAttributeName: [NSFont monospacedSystemFontOfSize:13.5 weight:NSFontWeightBold],
            NSForegroundColorAttributeName: s_colPri, NSShadowAttributeName: textShadow
        };
        s_attrSmall = @{
            NSFontAttributeName: [NSFont monospacedSystemFontOfSize:10.5 weight:NSFontWeightBold],
            NSForegroundColorAttributeName: s_colPri, NSShadowAttributeName: textShadow
        };
#if !__has_feature(objc_arc)
        [s_colPri retain];
        [s_attrPri retain];
        [s_attrSmall retain];
        [textShadow release];
#endif
    }

    CGColorRef cgPri        = s_cgPri;
    CGColorRef cgShad       = s_cgShad;
    CGColorRef cgBg         = s_cgBg;
    NSDictionary *attrPri   = s_attrPri;
    NSDictionary *attrSmall = s_attrSmall;

    telem_data_t telem = telem_get_snapshot();
    pthread_mutex_lock(&g_ctrl_lock);
    ctrl_state_t ctrl = g_ctrl;
    pthread_mutex_unlock(&g_ctrl_lock);

    /* Smooth attitude interpolation (~45 ms time constant, zero-jump 360° wrap) */
    const float ATT_SMOOTH_TAU_SEC = 0.045f;
    static bool   s_att_init   = false;
    static double s_att_last_t = 0.0;
    static float  s_disp_pitch = 0.0f;
    static float  s_disp_roll  = 0.0f;
    static float  s_disp_hdg   = 0.0f;

    double t_now = now_seconds();
    float raw_hdg = fmodf(telem.yaw_deg + 360.0f, 360.0f);
    if (!s_att_init) {
        s_disp_pitch = telem.pitch_deg;
        s_disp_roll  = telem.roll_deg;
        s_disp_hdg   = raw_hdg;
        s_att_init   = true;
    } else {
        float dt = (float)(t_now - s_att_last_t);
        if (dt < 0.0f) dt = 0.0f;
        if (dt > 0.1f) dt = 0.1f;
        float alpha = 1.0f - expf(-dt / ATT_SMOOTH_TAU_SEC);

        s_disp_pitch += wrap_delta_deg(telem.pitch_deg - s_disp_pitch) * alpha;
        s_disp_roll  += wrap_delta_deg(telem.roll_deg  - s_disp_roll)  * alpha;
        s_disp_hdg    = fmodf(s_disp_hdg + wrap_delta_deg(raw_hdg - s_disp_hdg) * alpha + 360.0f, 360.0f);
    }
    s_att_last_t = t_now;

    float disp_pitch = s_disp_pitch;
    float disp_roll  = s_disp_roll;
    float hdg        = s_disp_hdg;
    double home_dist_m = 0.0;
    float home_rel_deg = 0.0f;
    if (telem.has_home) {
        double dlat = (telem.latitude - telem.home_lat) * 111319.9;
        double dlon = (telem.longitude - telem.home_lon) * 111319.9 * cos(telem.home_lat * M_PI / 180.0);
        home_dist_m = hypot(dlat, dlon);
        float home_brg = fmodf((float)(atan2(-dlon, -dlat) * 180.0 / M_PI) + 360.0f, 360.0f);
        home_rel_deg = fmodf(home_brg - hdg + 540.0f, 360.0f) - 180.0f;
    }

    CGContextSaveGState(ctx);
    CGContextScaleCTM(ctx, scale, scale);
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithCGContext:ctx flipped:NO]];

    CGFloat cx = w * 0.5, cy = h * 0.5;

    /* 1. Center: Plain Broken Horizon + 120-Degree Clear-Center Crosshair */
    CGContextSaveGState(ctx);
    CGContextTranslateCTM(ctx, cx, cy);
    CGContextRotateCTM(ctx, -disp_roll * (CGFloat)M_PI / 180.0);
    CGFloat pitchOffset = disp_pitch * 6.0;
    CGContextBeginPath(ctx);
    CGContextMoveToPoint(ctx, -135, pitchOffset); CGContextAddLineToPoint(ctx, -38, pitchOffset);
    CGContextMoveToPoint(ctx,   38, pitchOffset); CGContextAddLineToPoint(ctx, 135, pitchOffset);
    hud_stroke_backed(ctx, cgPri, cgShad, 1.8);
    CGContextRestoreGState(ctx);

    CGFloat rIn = 8.0, rOut = 18.0;
    CGContextBeginPath(ctx);
    for (int i = 0; i < 3; i++) {
        CGFloat a = (90.0 + i * 120.0) * (CGFloat)M_PI / 180.0;
        CGContextMoveToPoint(ctx, cx + cos(a) * rIn,  cy + sin(a) * rIn);
        CGContextAddLineToPoint(ctx, cx + cos(a) * rOut, cy + sin(a) * rOut);
    }
    hud_stroke_backed(ctx, cgPri, cgShad, 1.8);

    /* 2. Left Edge: Slim Throttle Bar */
    CGFloat thH = 180.0, thW = 8.0, thX = 36.0, thY = (h - thH) * 0.5;
    CGContextSetFillColorWithColor(ctx, cgBg);
    CGContextFillRect(ctx, CGRectMake(thX, thY, thW, thH));
    CGContextSetFillColorWithColor(ctx, cgPri);
    CGContextFillRect(ctx, CGRectMake(thX + 1.5, thY + 1.5, thW - 3.0, (thH - 3.0) * ctrl.throttle));
    CGContextBeginPath(ctx); CGContextAddRect(ctx, CGRectMake(thX, thY, thW, thH));
    hud_stroke_backed(ctx, cgPri, cgShad, 1.2);

    /* 3. Top-Center: Concentric Roll Ticks (±45°, no arc line) + Notch Needle */
    CGFloat notchH = 0.0;
    if (@available(macOS 12.0, *)) {
        NSScreen *scr = g_metal_view.window.screen ?: [NSScreen mainScreen];
        notchH = scr.safeAreaInsets.top;
    }
    CGFloat needleBaseY = (notchH > 0.0) ? (h - notchH) : (h - 22.0);
    CGFloat needleTipY  = needleBaseY - 8.0;
    CGContextBeginPath(ctx);
    CGContextMoveToPoint(ctx, cx - 5, needleBaseY);
    CGContextAddLineToPoint(ctx, cx + 5, needleBaseY);
    CGContextAddLineToPoint(ctx, cx, needleTipY);
    CGContextClosePath(ctx);
    hud_stroke_backed(ctx, cgPri, cgShad, 1.4);

    CGFloat rollR = (needleTipY - 4.0) - cy;
    if (rollR > 60.0) {
        CGContextSaveGState(ctx);
        CGContextTranslateCTM(ctx, cx, cy);
        CGContextRotateCTM(ctx, -disp_roll * (CGFloat)M_PI / 180.0);
        CGContextBeginPath(ctx);
        for (int d = -45; d <= 45; d += 5) {
            CGFloat rad = (90.0 + d) * M_PI / 180.0;
            CGFloat tick = (d == 0) ? 12.0 : ((d % 15 == 0) ? 9.0 : 4.5);
            CGContextMoveToPoint(ctx, cos(rad) * rollR, sin(rad) * rollR);
            CGContextAddLineToPoint(ctx, cos(rad) * (rollR - tick), sin(rad) * (rollR - tick));
            if (d != 0 && d % 15 == 0) {
                NSString *rl = [NSString stringWithFormat:@"%d", abs(d)];
                CGSize rsz = [rl sizeWithAttributes:attrSmall];
                [rl drawAtPoint:CGPointMake(cos(rad) * (rollR - 20.0) - rsz.width * 0.5,
                                            sin(rad) * (rollR - 20.0) - rsz.height * 0.5)
                 withAttributes:attrSmall];
            }
        }
        hud_stroke_backed(ctx, cgPri, cgShad, 1.3);
        CGContextRestoreGState(ctx);
    }

    /* 4. Right Edge: Vertical Pitch Tape (±35° linear) */
    CGFloat pitchX = w - 52.0, pitchH = fmin(360.0, h * 0.50), pitchPxPerDeg = 5.0;
    CGContextSaveGState(ctx);
    CGContextClipToRect(ctx, CGRectMake(pitchX - 4.0, cy - pitchH * 0.5, 62.0, pitchH));
    CGContextBeginPath(ctx);
    int basePitch = ((int)floorf(disp_pitch / 5.0f)) * 5;
    for (int d = basePitch - 40; d <= basePitch + 40; d += 5) {
        if (d < -90 || d > 90) continue;
        CGFloat y = cy + ((float)d - disp_pitch) * pitchPxPerDeg;
        CGFloat tick = (d == 0) ? 14.0 : ((d % 10 == 0) ? 10.0 : 5.0);
        CGContextMoveToPoint(ctx, pitchX, y);
        CGContextAddLineToPoint(ctx, pitchX + tick, y);
        if (d % 10 == 0) {
            NSString *pl = [NSString stringWithFormat:@"%+d", d];
            [pl drawAtPoint:CGPointMake(pitchX + 14.0, y - 6.0) withAttributes:attrSmall];
        }
    }
    hud_stroke_backed(ctx, cgPri, cgShad, 1.2);
    CGContextRestoreGState(ctx);

    CGContextBeginPath(ctx); /* Fixed pitch pointer + readout */
    CGContextMoveToPoint(ctx, pitchX - 10, cy + 4);
    CGContextAddLineToPoint(ctx, pitchX - 3, cy);
    CGContextAddLineToPoint(ctx, pitchX - 10, cy - 4);
    CGContextClosePath(ctx);
    hud_stroke_backed(ctx, cgPri, cgShad, 1.3);
    CGRect pBox = CGRectMake(pitchX + 2.0, cy - 9.0, 42.0, 18.0);
    CGContextSetFillColorWithColor(ctx, cgBg);
    CGContextFillRect(ctx, pBox);
    CGContextBeginPath(ctx); CGContextAddRect(ctx, pBox);
    hud_stroke_backed(ctx, cgPri, cgShad, 1.1);
    NSString *pVal = [NSString stringWithFormat:@"%+3d°", (int)roundf(disp_pitch)];
    [pVal drawAtPoint:CGPointMake(pitchX + 6.0, cy - 6.5) withAttributes:attrSmall];

    /* 5. Bottom-Center: Minimal Edge Heading Tape (±60°, readout overlaps tape) */
    CGFloat tapeW = 600.0, tapeY = 32.0, pxPerDeg = 5.0;
    CGContextSaveGState(ctx);
    CGContextClipToRect(ctx, CGRectMake(cx - tapeW * 0.5, tapeY - 18.0, tapeW, 32.0));
    CGContextBeginPath(ctx);
    int baseDeg = ((int)floorf(hdg / 5.0f)) * 5;
    for (int d = baseDeg - 65; d <= baseDeg + 65; d += 5) {
        CGFloat x = cx + ((float)d - hdg) * pxPerDeg;
        int norm = ((d % 360) + 360) % 360;
        bool isCard = (norm % 45 == 0), is30 = (norm % 30 == 0), is15 = (norm % 15 == 0);
        CGContextMoveToPoint(ctx, x, tapeY);
        CGContextAddLineToPoint(ctx, x, tapeY + ((isCard || is15) ? 9.0 : 4.5));
        if (isCard) {
            const char *card = (norm == 0) ? "N" : (norm == 45) ? "NE" : (norm == 90) ? "E" :
                               (norm == 135) ? "SE" : (norm == 180) ? "S" : (norm == 225) ? "SW" :
                               (norm == 270) ? "W" : "NW";
            NSString *cs = [NSString stringWithUTF8String:card];
            CGSize sz = [cs sizeWithAttributes:attrSmall];
            [cs drawAtPoint:CGPointMake(x - sz.width * 0.5, tapeY - 14.0) withAttributes:attrSmall];
        } else if (is30) {
            NSString *ds = [NSString stringWithFormat:@"%03d", norm];
            CGSize sz = [ds sizeWithAttributes:attrSmall];
            [ds drawAtPoint:CGPointMake(x - sz.width * 0.5, tapeY - 14.0) withAttributes:attrSmall];
        }
    }
    hud_stroke_backed(ctx, cgPri, cgShad, 1.2);

    if (telem.has_home) {
        CGFloat hx = cx + clampf(home_rel_deg, -58.0f, 58.0f) * pxPerDeg;
        CGContextBeginPath(ctx);
        CGContextMoveToPoint(ctx, hx, tapeY + 10); CGContextAddLineToPoint(ctx, hx - 4, tapeY + 5);
        CGContextAddLineToPoint(ctx, hx, tapeY);      CGContextAddLineToPoint(ctx, hx + 4, tapeY + 5);
        CGContextClosePath(ctx);
        hud_stroke_backed(ctx, cgPri, cgShad, 1.4);
    }
    CGContextRestoreGState(ctx);

    CGContextBeginPath(ctx); /* Fixed pointer above ticks + overlapping readout box */
    CGContextMoveToPoint(ctx, cx - 4, tapeY + 17); CGContextAddLineToPoint(ctx, cx + 4, tapeY + 17);
    CGContextAddLineToPoint(ctx, cx, tapeY + 11);
    CGContextClosePath(ctx);
    hud_stroke_backed(ctx, cgPri, cgShad, 1.2);

    NSString *hdgStr = telem.has_home
        ? [NSString stringWithFormat:@"%03d° H:%.0fm", (int)roundf(hdg) % 360, home_dist_m]
        : [NSString stringWithFormat:@"%03d°", (int)roundf(hdg) % 360];
    CGSize hdgSz = [hdgStr sizeWithAttributes:attrSmall];
    CGFloat boxW = telem.has_home ? 94.0 : 44.0;
    CGRect hBox = CGRectMake(cx - boxW * 0.5, tapeY - 16.0, boxW, 18.0);
    CGContextSetFillColorWithColor(ctx, cgBg);
    CGContextFillRect(ctx, hBox);
    CGContextBeginPath(ctx); CGContextAddRect(ctx, hBox);
    hud_stroke_backed(ctx, cgPri, cgShad, 1.1);
    [hdgStr drawAtPoint:CGPointMake(cx - hdgSz.width * 0.5, tapeY - 13.5) withAttributes:attrSmall];

    /* 6. Top & Bottom Telemetry Text */
    const char *mode = ctrl.rtl ? "RTL-ACTIVE" : (ctrl.flight_mode == 1 ? "MAN" : (ctrl.flight_mode == 2 ? "STAB" : "FBWA"));
    NSString *osdTop = [NSString stringWithFormat:@"%s [%s]  %s   LQ:%u%%  RSSI:-%udBm",
                        telem.flight_mode, mode, ctrl.armed ? "ARMED" : "DISARMED",
                        telem.lq, telem.rssi];
    [osdTop drawAtPoint:CGPointMake(36, h - 36) withAttributes:attrPri];

#if SHOW_LATENCY_TIMER
    NSString *timerStr = [NSString stringWithFormat:@"%04u ms",
                          (uint32_t)((uint64_t)(now_seconds() * 1000.0) % 10000)];
    CGSize timerSize = [timerStr sizeWithAttributes:attrPri];
    [timerStr drawAtPoint:CGPointMake(w - timerSize.width - 36, h - 36) withAttributes:attrPri];
#endif

    NSString *osdPower = [NSString stringWithFormat:@"BAT: %4.1fV %4.1fA (%u%%) %u mAh   THR: %3d%%",
                          telem.voltage, telem.current, telem.battery_pct, telem.mah,
                          (int)(ctrl.throttle * 100.0f)];
    [osdPower drawAtPoint:CGPointMake(36, 20) withAttributes:attrPri];

    NSString *osdNav = (telem.sats > 0)
        ? [NSString stringWithFormat:@"ALT: %5.1fm  SPD: %3.0fkm/h  SAT:%2u  %.5f, %.5f",
           telem.altitude_m, telem.groundspeed_kmh, telem.sats, telem.latitude, telem.longitude]
        : [NSString stringWithFormat:@"ALT: %5.1fm  SPD: %3.0fkm/h  NO GPS",
           telem.altitude_m, telem.groundspeed_kmh];
    CGSize navSize = [osdNav sizeWithAttributes:attrPri];
    [osdNav drawAtPoint:CGPointMake(w - navSize.width - 36, 20) withAttributes:attrPri];

    [NSGraphicsContext restoreGraphicsState];
    CGContextRestoreGState(ctx);

    pthread_mutex_lock(&g_metal.tex_lock);
    int writeHudIdx = 1 - g_metal.activeHudIdx;
    id<MTLTexture> targetHud = g_metal.hudTex[writeHudIdx];
    pthread_mutex_unlock(&g_metal.tex_lock);

    if (targetHud && [targetHud width] == (NSUInteger)pixel_w && [targetHud height] == (NSUInteger)pixel_h) {
        [targetHud replaceRegion:MTLRegionMake2D(0, 0, pixel_w, pixel_h)
                     mipmapLevel:0
                       withBytes:s_hud_pixels
                     bytesPerRow:pixel_w * 4];

        pthread_mutex_lock(&g_metal.tex_lock);
        if (g_metal.hudTex[writeHudIdx] == targetHud) {
            g_metal.activeHudIdx = writeHudIdx;
        }
        pthread_mutex_unlock(&g_metal.tex_lock);
    }
    } /* @autoreleasepool */
}

/* ================================================================
 * Metal Presentation
 * ================================================================ */
@implementation FPVMetalView

- (instancetype)initWithFrame:(NSRect)frameRect {
    if ((self = [super initWithFrame:frameRect])) {
        self.wantsLayer = YES;
    }
    return self;
}

- (CALayer *)makeBackingLayer {
    CAMetalLayer *layer = [CAMetalLayer layer];
    layer.device = g_metal.device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.presentsWithTransaction = NO;

    layer.maximumDrawableCount = 3;
    layer.displaySyncEnabled = NO;
    layer.opaque = YES;

    CGFloat scale = [[NSScreen mainScreen] backingScaleFactor];
    layer.contentsScale = scale;
    CGSize viewSize = self.bounds.size;
    layer.drawableSize = CGSizeMake(viewSize.width * scale, viewSize.height * scale);

    self.metalLayer = layer;
    return layer;
}

- (void)viewDidChangeBackingProperties {
    [super viewDidChangeBackingProperties];
    [self updateDrawableSize];
}

- (void)setFrameSize:(NSSize)newSize {
    [super setFrameSize:newSize];
    [self updateDrawableSize];
}

- (void)updateDrawableSize {
    pthread_mutex_lock(&g_metal.render_lock);
    CGFloat scale = self.window.backingScaleFactor ?: [[NSScreen mainScreen] backingScaleFactor];
    self.metalLayer.contentsScale = scale;
    CGSize pixelSize = [self convertSizeToBacking:self.bounds.size];
    if (pixelSize.width > 0 && pixelSize.height > 0) {
        self.metalLayer.drawableSize = pixelSize;
    }
    pthread_mutex_unlock(&g_metal.render_lock);
}

- (void)renderFrame {
    pthread_mutex_lock(&g_metal.render_lock);

    pthread_mutex_lock(&g_metal.tex_lock);
    if (!g_metal.has_frame || !g_metal.yTex[g_metal.activeIdx]) {
        pthread_mutex_unlock(&g_metal.tex_lock);
        pthread_mutex_unlock(&g_metal.render_lock);
        return;
    }
    int idx = g_metal.activeIdx;
    id<MTLTexture> y = g_metal.yTex[idx], u = g_metal.uTex[idx], v = g_metal.vTex[idx];
    id<MTLTexture> hud = g_metal.hudTex[g_metal.activeHudIdx];
    pthread_mutex_unlock(&g_metal.tex_lock);

    id<CAMetalDrawable> drawable = [self.metalLayer nextDrawable];
    if (!drawable) {
        pthread_mutex_unlock(&g_metal.render_lock);
        return;
    }

    double t_render_start = CACurrentMediaTime();

    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawable.texture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    id<MTLCommandBuffer> cmdBuf = [g_metal.commandQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cmdBuf renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:g_metal.pipelineState];
    [enc setFragmentTexture:y atIndex:0];
    [enc setFragmentTexture:u atIndex:1];
    [enc setFragmentTexture:v atIndex:2];
    if (hud) {
        [enc setFragmentTexture:hud atIndex:3];
    }
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [enc endEncoding];

    [drawable addPresentedHandler:^(id<MTLDrawable> d) {
        double t_presented = d.presentedTime;
        if (t_presented > 0) {
            double display_delay_ms = (t_presented - t_render_start) * 1000.0;
            static atomic_int pres_cnt = 0;
            if ((atomic_fetch_add(&pres_cnt, 1) + 1) % 60 == 0) {
                printf("[mac] Metal -> Screen Glass delay: %.2f ms\n", display_delay_ms);
            }
        }
    }];

    [cmdBuf presentDrawable:drawable];
    [cmdBuf commit];

    pthread_mutex_unlock(&g_metal.render_lock);
}

@end

/* ================================================================
 * Window & App Setup
 * ================================================================ */
@interface FullscreenWindow : NSWindow
@property (nonatomic, assign) BOOL isFullscreen;
@property (nonatomic, assign) BOOL cursorHidden;
@property (nonatomic, assign) NSRect savedWindowedRect;
- (void)toggleFullscreenMode;
- (void)updateCursorVisibility;
@end

static bool handle_sim_key(unsigned short code, bool down) {
    bool handled = true;
    pthread_mutex_lock(&g_ctrl_lock);
    switch (code) {
        case kVK_ANSI_W:     g_keys.throttle_up   = down; break;
        case kVK_ANSI_S:     g_keys.throttle_down = down; break;
        case kVK_ANSI_A:     g_keys.yaw_left      = down; break;
        case kVK_ANSI_D:     g_keys.yaw_right     = down; break;

        case kVK_UpArrow:    g_keys.pitch_down    = down; break;
        case kVK_DownArrow:  g_keys.pitch_up      = down; break;
        case kVK_LeftArrow:  g_keys.roll_left     = down; break;
        case kVK_RightArrow: g_keys.roll_right    = down; break;

        case kVK_ANSI_J:     g_keys.cam_pan_left  = down; break;
        case kVK_ANSI_L:     g_keys.cam_pan_right = down; break;
        case kVK_ANSI_I:     g_keys.cam_tilt_up   = down; break;
        case kVK_ANSI_K:     g_keys.cam_tilt_down = down; break;
        default:             handled = false; break;
    }
    pthread_mutex_unlock(&g_ctrl_lock);
    return handled;
}

@implementation FullscreenWindow

- (BOOL)canBecomeKeyWindow  { return YES; }
- (BOOL)canBecomeMainWindow { return YES; }

- (void)updateCursorVisibility {
    if (self.isFullscreen && [self isKeyWindow]) {
        if (!self.cursorHidden) {
            [NSCursor hide];
            self.cursorHidden = YES;
        }
    } else {
        if (self.cursorHidden) {
            [NSCursor unhide];
            self.cursorHidden = NO;
        }
    }
}

- (void)becomeKeyWindow {
    [super becomeKeyWindow];
    [self updateCursorVisibility];
}

- (void)resignKeyWindow {
    [super resignKeyWindow];
    pthread_mutex_lock(&g_ctrl_lock);
    memset(&g_keys, 0, sizeof(g_keys));
    pthread_mutex_unlock(&g_ctrl_lock);
    [self updateCursorVisibility];
}

- (void)toggleFullscreenMode {
    if (self.isFullscreen) {
        self.styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
        [self setLevel:NSNormalWindowLevel];
        [self setTitle:@"FPV Video + Telemetry HUD"];
        [self setFrame:(self.savedWindowedRect.size.width > 0 ? self.savedWindowedRect : NSMakeRect(100, 100, 1280, 720)) display:YES animate:NO];
        self.isFullscreen = NO;
    } else {
        self.savedWindowedRect = self.frame;
        self.styleMask = NSWindowStyleMaskBorderless;
        [self setLevel:NSMainMenuWindowLevel + 1];
        [self setFrame:[[self screen] frame] display:YES animate:NO];
        self.isFullscreen = YES;
    }
    [self makeKeyAndOrderFront:nil];
    [self updateCursorVisibility];
}

- (void)keyDown:(NSEvent *)event {
    if ([event isARepeat]) return;
    unsigned short code = [event keyCode];
    if (handle_sim_key(code, true)) return;

    switch (code) {
        case kVK_ANSI_1: case kVK_ANSI_2: case kVK_ANSI_3: {
            pthread_mutex_lock(&g_ctrl_lock);
            g_ctrl.flight_mode = (code == kVK_ANSI_1) ? 1 : ((code == kVK_ANSI_2) ? 2 : 3);
            pthread_mutex_unlock(&g_ctrl_lock);
            break;
        }

        case kVK_ANSI_R: {
            pthread_mutex_lock(&g_ctrl_lock);
            g_ctrl.rtl = !g_ctrl.rtl;
            pthread_mutex_unlock(&g_ctrl_lock);
            break;
        }

        case kVK_Space:
            if ([event modifierFlags] & NSEventModifierFlagShift) {
                pthread_mutex_lock(&g_ctrl_lock);
                g_ctrl.armed = !g_ctrl.armed;
                pthread_mutex_unlock(&g_ctrl_lock);
            }
            break;

        case kVK_ANSI_C: ctrl_center(); break;
        case kVK_ANSI_F: [self toggleFullscreenMode]; break;
        case kVK_Escape:
            g_quit = 1;
            [NSApp stop:nil];
            [NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                                location:NSZeroPoint
                                           modifierFlags:0
                                               timestamp:0
                                            windowNumber:0
                                                 context:nil
                                                 subtype:0
                                                   data1:0
                                                   data2:0] atStart:YES];
            break;

        default: [super keyDown:event]; break;
    }
}

- (void)keyUp:(NSEvent *)event {
    if (!handle_sim_key([event keyCode], false)) {
        [super keyUp:event];
    }
}

@end

/* ================================================================
 * Main Application Entry Point
 * ================================================================ */
int main(int argc, char **argv) {
    if (argc > 1) g_serial_dev_path = argv[1];

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    signal(SIGPIPE, SIG_IGN);

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || metal_pipeline_init(device) != 0) return -1;

        NSRect screenRect = [[NSScreen mainScreen] frame];
        FullscreenWindow *window = [[FullscreenWindow alloc] initWithContentRect:screenRect
            styleMask:NSWindowStyleMaskBorderless backing:NSBackingStoreBuffered defer:NO];
        [window setLevel:NSMainMenuWindowLevel + 1];
        [window setOpaque:YES];
        window.isFullscreen = YES;
        window.savedWindowedRect = NSMakeRect(100, 100, 1280, 720);

        FPVMetalView *metalView = [[FPVMetalView alloc] initWithFrame:screenRect];
        g_metal_view = metalView;
        metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

        metal_init_black_textures();

        fprintf(stderr, "[mac] screen max refresh: %ld Hz\n",
                (long)[[NSScreen mainScreen] maximumFramesPerSecond]);

        [window setContentView:metalView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        [NSTimer scheduledTimerWithTimeInterval:0.016 repeats:YES block:^(NSTimer *timer) {
            if (g_quit) {
                [timer invalidate];
                [NSApp stop:nil];
                [NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                                    location:NSZeroPoint
                                               modifierFlags:0
                                                   timestamp:0
                                                windowNumber:0
                                                     context:nil
                                                     subtype:0
                                                       data1:0
                                                       data2:0] atStart:YES];
                return;
            }
            render_hud_to_texture();

            if (now_seconds() - atomic_load(&g_last_video_render_time) > 0.020) {
                if (g_metal_view) {
                    [g_metal_view renderFrame];
                }
            }
        }];

        pthread_t video_tid, serial_tid;
        pthread_create(&video_tid, NULL, video_thread_fn, NULL);
        pthread_create(&serial_tid, NULL, serial_thread_fn, NULL);

        fprintf(stderr,
            "\n==================================================\n"
            " ArduPilot FPV Controller & Metal HUD Ready \n"
            "==================================================\n"
            " Port: %s\n\n", g_serial_dev_path);

        [NSApp run];

        g_quit = 1;
        pthread_mutex_lock(&g_q_lock);
        pthread_cond_broadcast(&g_q_cond);
        pthread_mutex_unlock(&g_q_lock);
        pthread_join(video_tid, NULL);
        pthread_join(serial_tid, NULL);
    }
    return 0;
}