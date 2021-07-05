#ifndef SHARED_H
#define SHARED_H
#include <time.h>

void cleanup(char** input, char** output);
void graceful_exit(char** input, char** output);
void error_exit(char** input, char** output);

void help(char const *exec, char const opt, char const *optarg);
void parse_args(int argc, char **argv, unsigned *iterations, unsigned *filterIndex, char**output, char **input);

float time_spent(struct timespec t0, struct timespec t1);

void log_execution(const char *filter_name, unsigned width, unsigned height, unsigned iterations, float spent_time);

#endif
