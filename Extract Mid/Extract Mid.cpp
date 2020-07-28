#include <iostream>
#include <algorithm>
#include <cmath>
#include <complex>
#include <thread>
#include <vector>
#include <fstream>

#include <fftw3.h>

#include "AudioFile.h"

int main(int argc, char** argv)
{
    // Get audio file path from command line arguments
    std::string filePath;
    if (argc > 1) filePath = argv[1];
    else filePath = "C:/Users/User/Music/Right.wav";

    AudioFile<float> audioFile;

    std::cout << "Loading audio file \"" << filePath << "\"" << std::endl;

    audioFile.load(filePath);

    if (audioFile.isMono())
    {
        std::cerr << "Audio is mono. This operation requires stereo audio";
        return 0;
    }

    std::cout << "Audio loaded. Extracting mid..." << std::endl;

    int numSamples = audioFile.getNumSamplesPerChannel();

    // Convert back to samples
    AudioFile<float>::AudioBuffer buffer(1);
    buffer.resize(1);
    buffer[0].resize(numSamples, 0.0f);

    const int WINDOW_SIZE = 8192;

    for (int j = 0; j < numSamples - WINDOW_SIZE; j += (WINDOW_SIZE / 2))
    {
        // Perform FFT on left and right samples
        std::vector <fftwf_complex> leftFourierSeries(WINDOW_SIZE);
        fftwf_plan leftPlan = fftwf_plan_dft_r2c_1d(WINDOW_SIZE, &audioFile.samples[0][j], &leftFourierSeries[0], FFTW_ESTIMATE);
        fftwf_execute_dft_r2c(leftPlan, &audioFile.samples[0][j], &leftFourierSeries[0]);
        fftwf_destroy_plan(leftPlan);

        std::vector <fftwf_complex> rightFourierSeries(WINDOW_SIZE);
        fftwf_plan rightPlan = fftwf_plan_dft_r2c_1d(WINDOW_SIZE, &audioFile.samples[1][j], &rightFourierSeries[0], FFTW_ESTIMATE);
        fftwf_execute_dft_r2c(rightPlan, &audioFile.samples[1][j], &rightFourierSeries[0]);
        fftwf_destroy_plan(rightPlan);

        // Get Side
        std::vector <fftwf_complex> sideFourierSeries(WINDOW_SIZE);
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            float leftMag = sqrt(leftFourierSeries[i][0] * leftFourierSeries[i][0] + leftFourierSeries[i][1] * leftFourierSeries[i][1]);
            float leftAngle = atan2f(leftFourierSeries[i][1], leftFourierSeries[i][0]);

            float rightMag = sqrt(rightFourierSeries[i][0] * rightFourierSeries[i][0] + rightFourierSeries[i][1] * rightFourierSeries[i][1]);
            float rightAngle = atan2f(rightFourierSeries[i][1], rightFourierSeries[i][0]);

            float sideMag = abs(leftMag - rightMag);

            float midReal = (leftFourierSeries[i][0] + rightFourierSeries[i][0]) / 2.0f;
            float midImag = (leftFourierSeries[i][1] + rightFourierSeries[i][1]) / 2.0f;

            float midMag = sqrt(midReal * midReal + midImag * midImag);
            float midAngle = atan2f(midImag, midReal);

            midMag -= sideMag * 0.5f;

            sideFourierSeries[i][0] = cos(midAngle) * midMag;
            sideFourierSeries[i][1] = sin(midAngle) * midMag;

        }

        std::vector<float> tempBuffer(WINDOW_SIZE, 0.0f);
        fftwf_plan sidePlan = fftwf_plan_dft_c2r_1d(WINDOW_SIZE, &sideFourierSeries[0], &tempBuffer[0], FFTW_ESTIMATE);
        fftwf_execute_dft_c2r(sidePlan, &sideFourierSeries[0], &tempBuffer[0]);
        fftwf_destroy_plan(sidePlan);

        for (unsigned int i = 0.0f; i < WINDOW_SIZE; i++)
        {
            tempBuffer[i] = tempBuffer[i] / (float)WINDOW_SIZE;
        }

        for (unsigned int i = WINDOW_SIZE * 0.25f; i < WINDOW_SIZE * 0.75f; i++)
        {
            buffer[0][j + i] = tempBuffer[i];
        }

    }

    audioFile.setAudioBuffer(buffer);

    audioFile.save("output.wav");

}
