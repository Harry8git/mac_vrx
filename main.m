/*
 * High-Speed USB Bulk Video Receiver (VRX) + Keyboard Plane Control +
 * ArduPilot CRSF Telemetry & HUD, for macOS.
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
#include <math.h>
#include <pthread.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <libusb-1.0/libusb.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------- Video (vendor bulk) config ---------------- */
#define TARGET_VID              0x2207
#define TARGET_PID              0x0011
#define BUFFER_SIZE             (64 * 1024)

/* ---------------- ArduPilot CRSF link config ---------------- */
#define ARDUPILOT_SERIAL_DEV_DEFAULT "/dev/cu.usbmodem14201"
#define ARDUPILOT_BAUD          B115200
#define CRSF_TX_HZ              50
#define SERIAL_RETRY_MS         2000

#define CRSF_SYNC_FC            0xC8
#define CRSF_MAX_FRAME_LEN      64
#define CRSF_FRAMETYPE_GPS      0x02
#define CRSF_FRAMETYPE_BATTERY  0x08
#define CRSF_FRAMETYPE_LINK_STAT 0x14
#define CRSF_FRAMETYPE_RC_CHANNELS 0x16
#define CRSF_FRAMETYPE_ATTITUDE 0x1E
#define CRSF_FRAMETYPE_FLIGHT_MODE 0x21

#define DECAY_TAU_SEC           0.35f   /* Spring-back decay time constant */
#define THROTTLE_RATE           0.50f   /* 0 to 100% in 2.0s */

#define SHOW_LATENCY_TIMER  1   /* Set to 0 when you are done measuring latency */

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
"                                  texture2d<float> yTexture [[texture(0)]],\n"
"                                  texture2d<float> uTexture [[texture(1)]],\n"
"                                  texture2d<float> vTexture [[texture(2)]]) {\n"
"    constexpr sampler s(address::clamp_to_edge, filter::linear);\n"
"    float y = yTexture.sample(s, in.texCoords).r;\n"
"    float u = uTexture.sample(s, in.texCoords).r - 0.5;\n"
"    float v = vTexture.sample(s, in.texCoords).r - 0.5;\n"
"    float3 rgb;\n"
"    rgb.r = y + 1.5748 * v;\n"
"    rgb.g = y - 0.1873 * u - 0.4681 * v;\n"
"    rgb.b = y + 1.8556 * u;\n"
"    return float4(clamp(rgb, 0.0, 1.0), 1.0);\n"
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
    id<MTLTexture>             yTex[2];
    id<MTLTexture>             uTex[2];
    id<MTLTexture>             vTex[2];
    int                        activeIdx;
    int                        width;
    int                        height;
    pthread_mutex_t            tex_lock;
    bool                       has_frame;
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
    return g_metal.pipelineState ? 0 : -1;
}

static void ensure_textures(int w, int h) {
    if (g_metal.width == w && g_metal.height == h && g_metal.yTex[0] != nil) return;
    g_metal.width = w;
    g_metal.height = h;

    for (int i = 0; i < 2; i++) {
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

static void metal_upload_yuv420p_frame(AVFrame *frame) {
    if (frame->format != AV_PIX_FMT_YUV420P) return;

    pthread_mutex_lock(&g_metal.tex_lock);
    int w = frame->width;
    int h = frame->height;
    ensure_textures(w, h);

    int writeIdx = 1 - g_metal.activeIdx;
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
    float roll, pitch, yaw;   /* -1..+1, spring-centered */
    float throttle;           /*  0..+1, holding/latching */
    bool  armed;              /* CH5: Arm/Disarm */
    bool  rtl;                /* CH7: RTL */
    int   flight_mode;        /* CH8: 1=MAN, 2=STAB, 3=FBWA */
} ctrl_state_t;

static ctrl_state_t   g_ctrl = { .flight_mode = 1 };
static pthread_mutex_t g_ctrl_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    bool roll_left, roll_right;
    bool pitch_down, pitch_up;
    bool yaw_left, yaw_right;
    bool throttle_up, throttle_down;
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
    int8_t   rssi;
    uint8_t  lq;
    double   latitude;
    double   longitude;
    float    altitude_m;
    float    groundspeed_kmh;
    uint8_t  sats;
    char     flight_mode[16];
} telem_data_t;

static telem_data_t    g_telem = { .flight_mode = "MANUAL" };
static pthread_mutex_t g_telem_lock = PTHREAD_MUTEX_INITIALIZER;

/* Telemetry cache helpers (clean snapshotting without copying mutexes) */
static telem_data_t telem_get_snapshot(void) {
    pthread_mutex_lock(&g_telem_lock);
    telem_data_t snap = g_telem;
    pthread_mutex_unlock(&g_telem_lock);
    return snap;
}

static void ctrl_center(void) {
    pthread_mutex_lock(&g_ctrl_lock);
    g_ctrl.roll = g_ctrl.pitch = g_ctrl.yaw = 0.0f;
    pthread_mutex_unlock(&g_ctrl_lock);
}

/* ================================================================
 * CRSF Encoding & Decoding
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
    ch[0] = encode_axis(s.roll);          /* CH1: Roll */
    ch[1] = encode_axis(s.pitch);         /* CH2: Pitch */
    ch[2] = encode_throttle(s.throttle);  /* CH3: Throttle */
    ch[3] = encode_axis(s.yaw);           /* CH4: Yaw */
    ch[4] = s.armed ? 1811 : 172;         /* CH5: Arm/Disarm */
    ch[5] = 992;                          /* CH6: Spare */
    ch[6] = s.rtl ? 1811 : 172;           /* CH7: RTL */
    ch[7] = (s.flight_mode == 1) ? 172 : ((s.flight_mode == 2) ? 992 : 1811); /* CH8: 3-Pos Mode */
    for (int i = 8; i < 16; i++) ch[i] = 992;

    out[0] = CRSF_SYNC_FC;
    out[1] = 22 + 2;
    out[2] = CRSF_FRAMETYPE_RC_CHANNELS;
    crsf_pack_channels(ch, &out[3]);
    out[25] = crsf_crc8(&out[2], 1 + 22);
    return 26;
}

static int crsf_build_link_stats_frame(uint8_t out[14]) {
    out[0] = CRSF_SYNC_FC;
    out[1] = 10 + 2;
    out[2] = CRSF_FRAMETYPE_LINK_STAT;
    out[3] = 45; out[4] = 45; out[5] = 100; out[6] = 10;
    out[7] = 0;  out[8] = 2;  /* 150Hz telemetry unlock */
    out[9] = 0;  out[10] = 45; out[11] = 100; out[12] = 10;
    out[13] = crsf_crc8(&out[2], 1 + 10);
    return 14;
}

static void handle_crsf_frame(const uint8_t *f, int len) {
    uint8_t type = f[0];
    const uint8_t *payload = &f[1];
    int payload_len = len - 2;

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
            g_telem.altitude_m      = (int16_t)((payload[12] << 8) | payload[13]) - 1000.0f;
            g_telem.sats            = payload[14];
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
            g_telem.rssi = (int8_t)payload[0];
            g_telem.lq   = payload[2];
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
 * Video Decoder Context (Low-Latency HEVC)
 * ================================================================ */
typedef struct {
    const AVCodec        *codec;
    AVCodecContext       *ctx;
    AVCodecParserContext *parser;
    AVFrame              *frame;
    AVPacket             *pkt;
} hevc_decoder_t;

static int hevc_decoder_init(hevc_decoder_t *dec) {
    memset(dec, 0, sizeof(*dec));
    dec->codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    if (!dec->codec) return -1;
    av_log_set_level(AV_LOG_FATAL);

    dec->parser = av_parser_init(dec->codec->id);
    dec->ctx    = avcodec_alloc_context3(dec->codec);
    if (!dec->parser || !dec->ctx) return -1;

    dec->ctx->flags  |= AV_CODEC_FLAG_LOW_DELAY;
    dec->ctx->flags2 |= AV_CODEC_FLAG2_FAST;
    dec->ctx->thread_type = 0;
    dec->ctx->thread_count = 1;
    dec->ctx->max_b_frames = 0;
    dec->ctx->delay = 0;

    AVDictionary *opts = NULL;
    av_dict_set(&opts, "tune", "zerolatency", 0);
    if (avcodec_open2(dec->ctx, dec->codec, &opts) < 0) return -1;
    av_dict_free(&opts);

    dec->frame = av_frame_alloc();
    dec->pkt   = av_packet_alloc();
    return (dec->frame && dec->pkt) ? 0 : -1;
}

static void hevc_decoder_feed_chunk(hevc_decoder_t *dec, const uint8_t *data, int size) {
    int frame_decoded = 0;

    while (size > 0 && !g_quit) {
        int consumed = av_parser_parse2(dec->parser, dec->ctx,
                                        &dec->pkt->data, &dec->pkt->size,
                                        data, size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (consumed < 0) break;
        data += consumed;
        size -= consumed;

        if (dec->pkt->data && dec->pkt->size > 0) {
            frame_decoded = 1;
            if (avcodec_send_packet(dec->ctx, dec->pkt) == 0) {
                while (avcodec_receive_frame(dec->ctx, dec->frame) == 0) {
                    metal_upload_yuv420p_frame(dec->frame);
                    av_frame_unref(dec->frame);
                    if (g_metal_view) {
                        [g_metal_view renderFrame];
                    }
                }
            }
        }
    }

    /* Only flush if this chunk didn't yield a frame yet */
    if (!frame_decoded && !g_quit) {
        dec->pkt->data = NULL;
        dec->pkt->size = 0;
        av_parser_parse2(dec->parser, dec->ctx,
                         &dec->pkt->data, &dec->pkt->size,
                         NULL, 0, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);

        if (dec->pkt->data && dec->pkt->size > 0) {
            if (avcodec_send_packet(dec->ctx, dec->pkt) == 0) {
                while (avcodec_receive_frame(dec->ctx, dec->frame) == 0) {
                    metal_upload_yuv420p_frame(dec->frame);
                    av_frame_unref(dec->frame);
                    if (g_metal_view) {
                        [g_metal_view renderFrame];
                    }
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
 * Serial Thread: 100 Hz Physics + CRSF I/O
 * ================================================================ */
static void *serial_thread_fn(void *arg) {
    (void)arg;
    crsf_parser_t parser = {0};
    int fd = -1;
    double last_tx_t = now_seconds();
    double last_physics_t = now_seconds();
    double last_link_t = now_seconds();

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
                tcsetattr(fd, TCSANOW, &tty);
                int modem_ctrl = TIOCM_DTR | TIOCM_RTS;
                ioctl(fd, TIOCMBIS, &modem_ctrl);
            }
            fprintf(stderr, "[serial] Connected to %s\n", g_serial_dev_path);
        }

        /* Read incoming telemetry */
        uint8_t rx_buf[256];
        ssize_t n;
        while ((n = read(fd, rx_buf, sizeof(rx_buf))) > 0) {
            for (ssize_t i = 0; i < n; i++) crsf_parser_feed(&parser, rx_buf[i]);
        }
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            close(fd);
            fd = -1;
            continue;
        }

        /* 100 Hz Proportional Decay Physics Engine */
        double t = now_seconds();
        double dt = t - last_physics_t;
        if (dt >= 0.01) {
            float target_roll  = (g_keys.roll_right ? 1.0f : 0.0f) - (g_keys.roll_left ? 1.0f : 0.0f);
            float target_pitch = (g_keys.pitch_up   ? 1.0f : 0.0f) - (g_keys.pitch_down ? 1.0f : 0.0f);
            float target_yaw   = (g_keys.yaw_right  ? 1.0f : 0.0f) - (g_keys.yaw_left   ? 1.0f : 0.0f);
            float blend = 1.0f - expf((float)(-dt / DECAY_TAU_SEC));

            pthread_mutex_lock(&g_ctrl_lock);
            g_ctrl.roll  += (target_roll  - g_ctrl.roll)  * blend;
            g_ctrl.pitch += (target_pitch - g_ctrl.pitch) * blend;
            g_ctrl.yaw   += (target_yaw   - g_ctrl.yaw)   * blend;

            if (g_keys.throttle_up)   g_ctrl.throttle = clampf(g_ctrl.throttle + THROTTLE_RATE * (float)dt, 0.0f, 1.0f);
            if (g_keys.throttle_down) g_ctrl.throttle = clampf(g_ctrl.throttle - THROTTLE_RATE * (float)dt, 0.0f, 1.0f);
            pthread_mutex_unlock(&g_ctrl_lock);

            last_physics_t = t;
        }

        /* 50 Hz Outbound CRSF Frame */
        if (t - last_tx_t >= (1.0 / CRSF_TX_HZ)) {
            last_tx_t = t;
            pthread_mutex_lock(&g_ctrl_lock);
            ctrl_state_t current = g_ctrl;
            pthread_mutex_unlock(&g_ctrl_lock);

            uint8_t tx_frame[26];
            int len = crsf_build_rc_frame(current, tx_frame);
            write(fd, tx_frame, len);
        }

        /* 10 Hz Link Statistics keep-alive */
        if (t - last_link_t >= 0.1) {
            last_link_t = t;
            uint8_t link_frame[14];
            int l_len = crsf_build_link_stats_frame(link_frame);
            write(fd, link_frame, l_len);
        }

        usleep(1000);
    }

    if (fd >= 0) close(fd);
    return NULL;
}



/* Search descriptors strictly for a Vendor-Specific Bulk IN endpoint */
static bool find_bulk_in_endpoint(libusb_device *dev, int *out_intf, uint8_t *out_ep) {
    struct libusb_config_descriptor *config = NULL;
    if (libusb_get_active_config_descriptor(dev, &config) < 0) {
        return false;
    }

    bool found = false;
    for (int i = 0; i < config->bNumInterfaces && !found; i++) {
        const struct libusb_interface *intf = &config->interface[i];
        for (int a = 0; a < intf->num_altsetting && !found; a++) {
            const struct libusb_interface_descriptor *alt = &intf->altsetting[a];

            /* CRITICAL: Only match Vendor-Specific class (0xFF) to ignore CDC-ACM serial */
            if (alt->bInterfaceClass != LIBUSB_CLASS_VENDOR_SPEC) {
                continue;
            }

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



/* ================================================================
 * Video Thread: Bulk USB Ingest
 * ================================================================ */
static void *video_thread_fn(void *arg) {
    (void)arg;
    libusb_context *ctx = NULL;
    if (libusb_init(&ctx) < 0) return NULL;

    hevc_decoder_t dec;
    if (hevc_decoder_init(&dec) != 0) {
        libusb_exit(ctx);
        return NULL;
    }

    uint8_t *usb_buffer = malloc(BUFFER_SIZE);

    while (!g_quit) {
        libusb_device_handle *dev = libusb_open_device_with_vid_pid(ctx, TARGET_VID, TARGET_PID);
        if (!dev) {
            usleep(250 * 1000);
            continue;
        }

        /* --- Dynamic Descriptor Lookup --- */
        int target_intf = -1;
        uint8_t ep_in_addr = 0;
        if (!find_bulk_in_endpoint(libusb_get_device(dev), &target_intf, &ep_in_addr)) {
            fprintf(stderr, "[video] Could not find Bulk IN endpoint on USB device!\n");
            libusb_close(dev);
            usleep(1000 * 1000);
            continue;
        }

        /* Detach OS kernel driver if macOS assigned a default driver */
        libusb_detach_kernel_driver(dev, target_intf);

        if (libusb_claim_interface(dev, target_intf) < 0) {
            fprintf(stderr, "[video] Failed to claim interface %d\n", target_intf);
            libusb_close(dev);
            usleep(500 * 1000);
            continue;
        }

        fprintf(stderr, "[video] Connected! Claimed interface %d, streaming endpoint 0x%02X\n",
                target_intf, ep_in_addr);

        while (!g_quit) {
            int actual = 0;
            /* Use the discovered ep_in_addr instead of hardcoded 0x81 */
            int ret = libusb_bulk_transfer(dev, ep_in_addr, usb_buffer, BUFFER_SIZE, &actual, 100);
            if (ret == 0 && actual > 0) {
                hevc_decoder_feed_chunk(&dec, usb_buffer, actual);
            } else if (ret == LIBUSB_ERROR_NO_DEVICE || ret == LIBUSB_ERROR_IO) {
                fprintf(stderr, "[video] Device disconnected\n");
                break;
            }
        }

        libusb_release_interface(dev, target_intf);
        libusb_close(dev);
    }

    free(usb_buffer);
    hevc_decoder_destroy(&dec);
    libusb_exit(ctx);
    return NULL;
}

/* ================================================================
 * HUD Overlay View (Horizon, Bat, GPS, Alt, Speed, Throttle)
 * ================================================================ */
@interface HUDOverlayView : NSView
@end

@implementation HUDOverlayView

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    CGContextRef ctx = [[NSGraphicsContext currentContext] CGContext];
    if (!ctx) return;

    telem_data_t telem = telem_get_snapshot();
    pthread_mutex_lock(&g_ctrl_lock);
    ctrl_state_t ctrl = g_ctrl;
    pthread_mutex_unlock(&g_ctrl_lock);

    CGFloat w = self.bounds.size.width;
    CGFloat h = self.bounds.size.height;
    CGPoint center = CGPointMake(w * 0.5, h * 0.5);

    /* 1. Artificial Horizon */
    CGContextSaveGState(ctx);
    CGContextTranslateCTM(ctx, center.x, center.y);
    CGContextRotateCTM(ctx, -telem.roll_deg * (CGFloat)M_PI / 180.0);

    CGFloat pitchOffset = telem.pitch_deg * 6.0;
    CGContextSetStrokeColorWithColor(ctx, [NSColor colorWithCalibratedRed:0.0 green:1.0 blue:0.4 alpha:0.85].CGColor);
    CGContextSetLineWidth(ctx, 2.0);
    CGContextStrokeRect(ctx, CGRectMake(-8, -8, 16, 16));
    CGContextMoveToPoint(ctx, -140, pitchOffset);
    CGContextAddLineToPoint(ctx, -40, pitchOffset);
    CGContextMoveToPoint(ctx, 40, pitchOffset);
    CGContextAddLineToPoint(ctx, 140, pitchOffset);
    CGContextStrokePath(ctx);
    CGContextRestoreGState(ctx);

    /* 2. Left Side: Throttle Indicator */
    CGFloat thH = 200.0, thW = 12.0, thX = 40.0, thY = (h - thH) * 0.5;
    CGContextSetStrokeColorWithColor(ctx, [NSColor whiteColor].CGColor);
    CGContextStrokeRect(ctx, CGRectMake(thX, thY, thW, thH));
    CGContextSetFillColorWithColor(ctx, [NSColor colorWithCalibratedRed:0.2 green:0.8 blue:1.0 alpha:0.75].CGColor);
    CGContextFillRect(ctx, CGRectMake(thX + 1, thY + 1, thW - 2, thH * ctrl.throttle));

    /* 3. Text OSD Elements */
    NSDictionary *attrGreen = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightBold],
        NSForegroundColorAttributeName: [NSColor colorWithCalibratedRed:0.0 green:1.0 blue:0.4 alpha:0.95]
    };

    /* Top Banner: Mode, Arm Status, Signal */
    NSString *osdTop = [NSString stringWithFormat:@"%s [%s]  %s   LQ: %u%%  RSSI: -%ddBm",
                        telem.flight_mode,
                        ctrl.rtl ? "RTL-ACTIVE" : (ctrl.flight_mode == 1 ? "MAN" : (ctrl.flight_mode == 2 ? "STAB" : "FBWA")),
                        ctrl.armed ? "ARMED" : "DISARMED",
                        telem.lq, telem.rssi];
    [osdTop drawAtPoint:CGPointMake(40, h - 35) withAttributes:attrGreen];

#if SHOW_LATENCY_TIMER
    /* 4-digit rollover stopwatch (0000 to 9999 ms) in the top-right corner */
    NSString *timerStr = [NSString stringWithFormat:@"%04u ms", 
                          (uint32_t)((uint64_t)(now_seconds() * 1000.0) % 10000)];
    CGSize timerSize = [timerStr sizeWithAttributes:attrGreen];
    [timerStr drawAtPoint:CGPointMake(w - timerSize.width - 40, h - 35) withAttributes:attrGreen];
#endif

    /* Bottom Left: Power & Motor */
    NSString *osdPower = [NSString stringWithFormat:@"BAT: %4.1fV %4.1fA (%u%%) %u mAh   THR: %3d%%",
                          telem.voltage, telem.current, telem.battery_pct, telem.mah, (int)(ctrl.throttle * 100.0f)];
    [osdPower drawAtPoint:CGPointMake(40, 20) withAttributes:attrGreen];

    /* Bottom Right: GPS & Navigation */
    NSString *osdNav = [NSString stringWithFormat:@"ALT: %4.0fm  SPD: %3.0fkm/h  SATS: %2u  LAT: %9.5f LON: %9.5f",
                        telem.altitude_m, telem.groundspeed_kmh, telem.sats, telem.latitude, telem.longitude];
    CGSize navSize = [osdNav sizeWithAttributes:attrGreen];
    [osdNav drawAtPoint:CGPointMake(w - navSize.width - 40, 20) withAttributes:attrGreen];
}

@end

/* ================================================================
 * Metal Presentation & Application Setup
 * ================================================================ */

@implementation FPVMetalView

- (instancetype)initWithFrame:(NSRect)frameRect {
    if ((self = [super initWithFrame:frameRect])) self.wantsLayer = YES;
    return self;
}

- (CALayer *)makeBackingLayer {
    CAMetalLayer *layer = [CAMetalLayer layer];
    layer.device = g_metal.device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.presentsWithTransaction = NO;
    layer.maximumDrawableCount = 2;
    /* BYPASS V-SYNC WAIT: Immediate scanout for lowest glass-to-glass latency */
    layer.displaySyncEnabled = NO;
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
    id<MTLTexture> y = g_metal.yTex[idx], u = g_metal.uTex[idx], v = g_metal.vTex[idx];
    pthread_mutex_unlock(&g_metal.tex_lock);

    id<CAMetalDrawable> drawable = [self.metalLayer nextDrawable];
    if (!drawable) return;

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
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [enc endEncoding];
    [cmdBuf presentDrawable:drawable];
    [cmdBuf commit];
}

- (void)displayLinkDidFire:(CADisplayLink *)link {
    (void)link;
}

@end

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
}

- (void)resignKeyWindow {
    [super resignKeyWindow];
    memset(&g_keys, 0, sizeof(g_keys));
}

- (void)keyDown:(NSEvent *)event {
    if ([event isARepeat]) return;
    unsigned short code = [event keyCode];

    switch (code) {
        /* Left Hand: Throttle & Rudder */
        case kVK_ANSI_W:       g_keys.throttle_up   = true; break;
        case kVK_ANSI_S:       g_keys.throttle_down = true; break;
        case kVK_ANSI_A:       g_keys.yaw_left      = true; break;
        case kVK_ANSI_D:       g_keys.yaw_right     = true; break;

        /* Right Hand: Pitch & Roll */
        case kVK_UpArrow:      g_keys.pitch_down    = true; break;
        case kVK_DownArrow:    g_keys.pitch_up      = true; break;
        case kVK_LeftArrow:    g_keys.roll_left     = true; break;
        case kVK_RightArrow:   g_keys.roll_right    = true; break;

        /* Modes: 1=MAN, 2=STAB, 3=FBWA */
        case kVK_ANSI_1: case kVK_ANSI_2: case kVK_ANSI_3: {
            pthread_mutex_lock(&g_ctrl_lock);
            g_ctrl.flight_mode = (code == kVK_ANSI_1) ? 1 : ((code == kVK_ANSI_2) ? 2 : 3);
            pthread_mutex_unlock(&g_ctrl_lock);
            break;
        }

        /* Emergency RTL */
        case kVK_ANSI_R: {
            pthread_mutex_lock(&g_ctrl_lock);
            g_ctrl.rtl = !g_ctrl.rtl;
            pthread_mutex_unlock(&g_ctrl_lock);
            break;
        }

        /* Arming: Shift + Space */
        case kVK_Space:
            if ([event modifierFlags] & NSEventModifierFlagShift) {
                pthread_mutex_lock(&g_ctrl_lock);
                g_ctrl.armed = !g_ctrl.armed;
                pthread_mutex_unlock(&g_ctrl_lock);
            }
            break;

        case kVK_ANSI_C:       ctrl_center(); break;
        case kVK_ANSI_F:       [self toggleFullscreenMode]; break;
        case kVK_ANSI_Q: case kVK_Escape:
            g_quit = 1;
            [NSApp stop:nil];
            break;

        default: [super keyDown:event]; break;
    }
}

- (void)keyUp:(NSEvent *)event {
    unsigned short code = [event keyCode];
    switch (code) {
        case kVK_ANSI_W:       g_keys.throttle_up   = false; break;
        case kVK_ANSI_S:       g_keys.throttle_down = false; break;
        case kVK_ANSI_A:       g_keys.yaw_left      = false; break;
        case kVK_ANSI_D:       g_keys.yaw_right     = false; break;

        case kVK_UpArrow:      g_keys.pitch_down    = false; break;
        case kVK_DownArrow:    g_keys.pitch_up      = false; break;
        case kVK_LeftArrow:    g_keys.roll_left     = false; break;
        case kVK_RightArrow:   g_keys.roll_right    = false; break;

        default: [super keyUp:event]; break;
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
        g_metal_view = metalView; /* Store for immediate presentation */
        HUDOverlayView *hudView   = [[HUDOverlayView alloc] initWithFrame:metalView.bounds];
        metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        hudView.autoresizingMask   = NSViewWidthSizable | NSViewHeightSizable;
        [metalView addSubview:hudView];
        [window setContentView:metalView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        CADisplayLink *displayLink = [metalView displayLinkWithTarget:metalView selector:@selector(displayLinkDidFire:)];

        /* UNLOCK 120Hz ProMotion: Cuts presentation interval from 16.6ms down to 8.3ms */
        displayLink.preferredFrameRateRange = CAFrameRateRangeMake(120, 120, 120);

        [displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
        
        [NSTimer scheduledTimerWithTimeInterval:0.016 repeats:YES block:^(NSTimer *timer) {
            (void)timer;
            [hudView setNeedsDisplay:YES];
        }];

        pthread_t video_tid, serial_tid;
        pthread_create(&video_tid, NULL, video_thread_fn, NULL);
        pthread_create(&serial_tid, NULL, serial_thread_fn, NULL);

        fprintf(stderr,
            "\n==================================================\n"
            " ArduPilot FPV Controller & Metal HUD Ready \n"
            "==================================================\n"
            " Left Hand:   W/S -> Throttle | A/D -> Rudder\n"
            " Right Hand:  Up/Down -> Pitch | Left/Right -> Roll\n"
            " Switches:    1: MANUAL | 2: STABILIZE | 3: FBWA\n"
            " Safety:      Shift + Space -> Arm/Disarm\n"
            " Emergency:   R -> Toggle RTL\n"
            " Display:     F -> Fullscreen | Q/Esc -> Exit\n"
            " Target Port: %s\n\n", g_serial_dev_path);

        [NSApp run];

        g_quit = 1;
        [displayLink invalidate];
        pthread_join(video_tid, NULL);
        pthread_join(serial_tid, NULL);
    }
    return 0;
}


// Before testing run:
// v4l2-ctl -d /dev/v4l-subdev2 -c exposure=250,analogue_gain=35