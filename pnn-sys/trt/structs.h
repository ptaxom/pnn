#pragma once

#include <stddef.h>

struct BindingInfo {
    const char* name;
    size_t batchsize;
    size_t channels;
    size_t height;
    size_t width;
    int is_input;
};