#include <cstdio>

#ifndef LLAI_LOG_H
#define LLAI_LOG_H

#define log_func(prefix, fmt, ...)  fprintf(stderr, prefix ":[%s(%u)]:" fmt "\n", __FILE__, __LINE__, __VA_ARGS__)
#define log_debug(fmt, ...)			log_func("debug", fmt, __VA_ARGS__)
#define log_error(fmt, ...)			log_func("error", fmt, __VA_ARGS__)

namespace llai
{
} // namespace

#endif // LLAI_LOG_H
