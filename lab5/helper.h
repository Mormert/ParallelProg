#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int writeppm(char* filename, int width, int height, unsigned char* data);
unsigned char* readppm(char* filename, int* width, int* height);