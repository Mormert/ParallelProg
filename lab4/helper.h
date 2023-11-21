#ifndef HELPER
#define HELPER

//#ifdef __APPLE__
//#include <OpenGL/gl.h>
//#else
//#include <GL/gl.h>
//#endif

#if defined(_WIN32) || defined(_WIN64) 
#include <Windows.h>
#include <stdint.h>

int gettimeofday(struct timeval* tp, struct timezone* tzp);

#elif
Linux: #include <sys/time.h>
#endif

int writeppm(char* filename, int width, int height, unsigned char* data);
unsigned char* readppm(char* filename, int* width, int* height);
//GLint readppmtexture(char* filename, char dofilter, char dorepeat);

#endif