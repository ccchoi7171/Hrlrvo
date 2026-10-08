#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Redirect only VLC's avcodec module in this process; system files stay intact. */
void *dlopen(const char *filename, int flags)
{
    void *(*original)(const char *, int) = dlsym(RTLD_NEXT, "dlopen");
    const char *replacement = getenv("HRLRVO_VLC_AVCODEC");
    if (filename && replacement) {
        const char *base = strrchr(filename, '/');
        base = base ? base + 1 : filename;
        if (strcmp(base, "libavcodec_plugin.so") == 0) {
            fprintf(stderr, "[VLC16K] Loading isolated 16384-pixel decoder: %s\n", replacement);
            filename = replacement;
        }
    }
    return original(filename, flags);
}
