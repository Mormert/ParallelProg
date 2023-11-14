
#include <complex>
#include <array>

#include "scope_profile.h"
#include "stb_image_write.h"

#include "WickedEngine/wiJobSystem.h"

constexpr int cWidth = 1024;
constexpr int cHeight = 1024;
constexpr int cWidthHeight = cWidth * cHeight;

constexpr int cMaxIterations = 256;

constexpr float cZoom = 3.f;
constexpr float cOffsetX = 2 * cWidth / 3.f;
constexpr float cOffsetY = cHeight / 2.f;

constexpr int cImageChannels = 3;
using pixel = std::array<char, cImageChannels>; // [Red, Green, Blue]
using image = std::array<pixel, cWidth * cHeight>;

inline std::array<char, 3> mandelbrot_pixel(int x, int y) {
    std::complex<float> point(cZoom * ((float) x - cOffsetX) / cWidth, cZoom * ((float) y - cOffsetY) / cHeight);
    std::complex<float> z = 0;

    int nb_iter = 0;
    while (abs(z) < 2 && nb_iter <= cMaxIterations) {
        z = z * z + point;
        ++nb_iter;
    }

    if (nb_iter < cMaxIterations)
        // Convert the iterations somehow to a pixel, this is one way of doing it, making it purple-ish
        return pixel{static_cast<char>(nb_iter * 2 % 256), static_cast<char>(nb_iter % 256),
                     static_cast<char>(nb_iter * 3 % 256)};
    else
        return pixel{0, 0, 0};
}

void mandelbrot_image_job(std::unique_ptr<image>& img, const int groupSize)
{
    wi::jobsystem::context ctx;

    wi::jobsystem::Dispatch(
            ctx, cWidth * cHeight, groupSize, [&img](wi::jobsystem::JobArgs args) {
                const int jobIndex = args.jobIndex;
                img->at(jobIndex) = mandelbrot_pixel(jobIndex % cWidth, jobIndex / cWidth);
            });

    wi::jobsystem::Wait(ctx);
}

int main() {

    const int threadAmount = 32;
    wi::jobsystem::Initialize(threadAmount);

    auto img = std::make_unique<image>();

    printf("Using %d threads.\n", threadAmount);

    {
        SCOPED_PROFILE_LOG("Generating mandelbrot set, sequential")
        for (int i = 0; i < cWidth * cHeight; i++) {
            img->at(i) = mandelbrot_pixel(i % cWidth, i / cWidth);
        }
    }

    {
        // Naive parallel implementation, we split the work among X amount of threads.
        SCOPED_PROFILE_LOG("Generating mandelbrot set, split equally among threads")

        const int groupSize = cWidthHeight / threadAmount;
        mandelbrot_image(img, groupSize);
    }

    {
        // Load-balanced implementation, splits the work into 'width' amount of jobs, executed by a jobs system, in parallel
        SCOPED_PROFILE_LOG("Generating mandelbrot set, balanced among jobs")

        const int groupSize = cWidth;
        mandelbrot_image(img, groupSize);
    }

    stbi_write_png("mandelbrot.png", cWidth, cHeight, cImageChannels, img->data(), cWidth * cImageChannels);

    return 0;
}