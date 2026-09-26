/*
 * High-Speed USB Bulk Video Receiver (VRX) for macOS
 * Reads raw H.265 stream from Luckfox FunctionFS Vendor Bulk Endpoint
 * and streams to stdout for low-latency pipe to ffplay.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <libusb-1.0/libusb.h>

#define TARGET_VID      0x2207
#define TARGET_PID      0x0011
#define BUFFER_SIZE     (64 * 1024)

static volatile int quit = 0;

static void sigint_handler(int sig) {
    (void)sig;
    quit = 1;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    libusb_context *ctx = NULL;
    libusb_device_handle *dev_handle = NULL;
    int ret;

    ret = libusb_init(&ctx);
    if (ret < 0) {
        fprintf(stderr, "ERROR: libusb_init failed: %s\n", libusb_strerror(ret));
        return 1;
    }

    fprintf(stderr, ">>> Searching for Luckfox VTX (VID: 0x%04X, PID: 0x%04X)...\n",
            TARGET_VID, TARGET_PID);

    dev_handle = libusb_open_device_with_vid_pid(ctx, TARGET_VID, TARGET_PID);
    if (!dev_handle) {
        fprintf(stderr, "ERROR: Could not find/open device. Is the USB cable connected?\n");
        libusb_exit(ctx);
        return 1;
    }

    libusb_device *dev = libusb_get_device(dev_handle);
    struct libusb_config_descriptor *config = NULL;
    ret = libusb_get_active_config_descriptor(dev, &config);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Failed to get config descriptor: %s\n", libusb_strerror(ret));
        libusb_close(dev_handle);
        libusb_exit(ctx);
        return 1;
    }

    int target_intf = -1;
    uint8_t ep_in_addr = 0;

    /* Scan interfaces for Vendor Specific Class (0xFF) with a Bulk IN endpoint */
    for (int i = 0; i < config->bNumInterfaces; i++) {
        const struct libusb_interface *intf = &config->interface[i];
        for (int a = 0; a < intf->num_altsetting; a++) {
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
            if (target_intf >= 0) break;
        }
        if (target_intf >= 0) break;
    }

    if (target_intf < 0 || ep_in_addr == 0) {
        fprintf(stderr, "ERROR: Vendor Bulk IN interface not found in USB descriptor!\n");
        libusb_free_config_descriptor(config);
        libusb_close(dev_handle);
        libusb_exit(ctx);
        return 1;
    }

    fprintf(stderr, ">>> Found Vendor Bulk Interface %d (Endpoint 0x%02X)\n",
            target_intf, ep_in_addr);

    libusb_free_config_descriptor(config);

    /* Claim Vendor Interface */
    ret = libusb_claim_interface(dev_handle, target_intf);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Could not claim interface %d: %s\n",
                target_intf, libusb_strerror(ret));
        libusb_close(dev_handle);
        libusb_exit(ctx);
        return 1;
    }

    fprintf(stderr, ">>> Interface claimed successfully! Streaming to stdout...\n");

    uint8_t *buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        libusb_release_interface(dev_handle, target_intf);
        libusb_close(dev_handle);
        libusb_exit(ctx);
        return 1;
    }

    int actual_length = 0;
    size_t total_bytes = 0;
    struct timespec last_time, now;
    clock_gettime(CLOCK_MONOTONIC, &last_time);

    while (!quit) {
        ret = libusb_bulk_transfer(dev_handle, ep_in_addr, buffer, BUFFER_SIZE,
                                   &actual_length, 1000);

        if (ret == 0 && actual_length > 0) {
            ssize_t written = write(STDOUT_FILENO, buffer, actual_length);
            (void)written;
            total_bytes += actual_length;

            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - last_time.tv_sec) +
                             (now.tv_nsec - last_time.tv_nsec) / 1000000000.0;

            if (elapsed >= 1.0) {
                fprintf(stderr, "\r>>> VRX Receiving: %.2f KB/s (%.2f Mbps) <<<   ",
                        (total_bytes / 1024.0) / elapsed,
                        (total_bytes * 8.0 / 1000000.0) / elapsed);
                fflush(stderr);
                total_bytes = 0;
                last_time = now;
            }
        } else if (ret == LIBUSB_ERROR_TIMEOUT) {
            /* No data within timeout, continue waiting */
            continue;
        } else if (ret < 0) {
            if (quit) break;
            fprintf(stderr, "\nBulk transfer error: %s\n", libusb_strerror(ret));
            break;
        }
    }

    fprintf(stderr, "\n>>> Shutting down VRX...\n");
    free(buffer);
    libusb_release_interface(dev_handle, target_intf);
    libusb_close(dev_handle);
    libusb_exit(ctx);
    return 0;
}
