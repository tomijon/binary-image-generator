#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BIT_DEPTH 8

constexpr int GreyChannel = 1;
constexpr float Ratio = 0.33f; // % of black pixels.
constexpr std::string_view Padding = "    ";
constexpr unsigned int SampleRate = 10;

using namespace std::chrono_literals;

#if BIT_DEPTH <= 8
typedef stbi_uc Pixel;
#else
typedef stbi_uint16 Pixel;
#endif

/**
 * Pretty print the threshold value and the time elapsed.
 * @param name - A reference name to use in the display output.
 * @param threshold - The threshold value found.
 * @param duration - The length of time the algorithm took (in seconds).
 */
void display(const std::string& name, Pixel threshold, float duration) {
	std::cout << name << std::endl;
	std::cout << Padding << "Threshold: " << (int)threshold << std::endl;
	std::cout << Padding << "Execution Time: " << std::fixed << std::setprecision(3) << duration << 's' << std::endl;
}


/**
 * Normal Distribution Approximation of the threshold value.
 * @param greyscale - the reference image.
 * @param width - the width of the image.
 * @param height - the height of the image.
 * @param ratio - the ratio of black to white pixels.
 */
Pixel normal_estimate(Pixel* greyscale, int width, int height, float ratio) {
	size_t total = 0;
	for (int pixel = 0; pixel < width * height; pixel++) {
		total += greyscale[pixel];
	}
	Pixel average = total / (width * height);

	size_t sigma_total = 0;
	for (int pixel = 0; pixel < width * height; pixel++) {
		sigma_total += std::pow(greyscale[pixel] - average, 2);
	}
	float sigma = std::sqrt((float)sigma_total * (1.0f / (width * height)));

	// Approximate z value using polynomial.
	float approximation = ratio + std::pow(ratio, 3.0f) + std::pow(ratio, 5.0f) + std::pow(ratio, 7.0f);
	float z = std::numbers::sqrt2 * approximation; 
	return (Pixel)((float)average + (z * sigma)); // CDF.
}


/**
 * Approximation of the threshold value using linear interpolation and the
 * average.
 * @param greyscale - the reference image.
 * @param width - the width of the image.
 * @param height - the height of the image.
 * @param ratio - the ratio of black to white pixels.
 */
Pixel weighted_estimate(Pixel* greyscale, int width, int height, float ratio) {
	size_t total = 0;
	for (int pixel = 0; pixel < width * height; pixel++) {
		total += greyscale[pixel];
	}
	Pixel average = total / (width * height);

	Pixel min, max;
	if (ratio > 0.5) {
		min = average;
		max = (1 << BIT_DEPTH) - 1;
	} else {
		min = 0;
		max = average;
	}
	
	if (ratio > 0.5) ratio -= 0.5;
	float percentage = ratio / 0.5f;
	return min + ((max - min) * percentage);
}


/**
 * Sort the image using std::sort to find the threshold value that will
 * produce a binary image with a black-white ratio closest to the given
 * ratio.
 * @param greyscale - the reference image.
 * @param width - the width of the image.
 * @param height - the height of the image.
 * @param ratio - the ratio of black to white pixels.
 */
Pixel std_sort(Pixel* greyscale, int width, int height, float ratio) {
	int n = (width * height) * ratio;
	std::sort(greyscale, greyscale + (width * height));
	return greyscale[n];
}


/**
 * @brief Sort the image using a counting sort to find the threshold value that
 * will produce a binary image with a black-white ratio closest to the given
 * ratio.
 * @param greyscale - the reference image.
 * @param width - the width of the image.
 * @param height - the height of the image.
 * @param ratio - the ratio of black to white pixels.
 */
Pixel counting_sort(Pixel* greyscale, int width, int height, float ratio) {
	std::unique_ptr<size_t[]> count = std::make_unique<size_t[]>(1 << BIT_DEPTH);
	size_t image_size = width * height;

	for (size_t pixel = 0; pixel < image_size; pixel++) {
		count[greyscale[pixel]]++;
	}

	size_t cutoff = image_size * ratio;
	size_t total = 0;
	size_t index = 0;

	while (total < cutoff && index < (1 << BIT_DEPTH)) {
		total += count[index++];
	}
	return std::max((size_t)0, index - 1);
}


/**
 * Find the threshold value using std::nth_element that will produce a binary
 * image with a black-white ratio closest to the given ratio.
 * @param greyscale - the reference image.
 * @param width - the width of the image.
 * @param height - the height of the image.
 * @param ratio - the ratio of black to white pixels.
 */
Pixel nth_element_sort(Pixel* greyscale, int width, int height, float ratio) {
	size_t n = (width * height) * ratio;
	std::nth_element(greyscale, greyscale + n, greyscale + (width * height));
	return greyscale[n];
}


/**
 * Run a counting sort version on a uniform sample of the input image.
 * @param greyscale - the reference image.
 * @param width - the width of the image.
 * @param height - the height of the image.
 * @param sample_rate - how often to sample the image (a value of 10 means
 * 		every 10 pixels)
 * @param ratio - the ratio of black to white pixels.
 */
Pixel uniform_sample(Pixel* greyscale, int width, int height, unsigned int sample_rate, float ratio) {
	std::unique_ptr<size_t[]> count = std::make_unique<size_t[]>(1 << BIT_DEPTH);
	size_t image_size = width * height;

	for (size_t pixel = 0; pixel < image_size; pixel += sample_rate) {
		count[greyscale[pixel]]++;
	}

	size_t cutoff = (image_size / sample_rate) * ratio;
	size_t total = 0;
	size_t index = 0;

	while (total < cutoff && index < (1 << BIT_DEPTH)) {
		total += count[index++];
	}
	return std::max((size_t)0, index - 1);
}


int main(int c, char* argv[]) {
	int width, height, channels;
	const char* greyscale_name = "sample_image.png";
	const char* binary_name = "sample_binary.png";
	Pixel* image;
	Pixel* copy;
	
#if BIT_DEPTH <= 8
	image = stbi_load(greyscale_name, &width, &height, &channels, GreyChannel);
#else
	image = stbi_load_16(greyscale_name, &width, &height, &channels, GreyChannel);
#endif
	assert(image != nullptr && "Failed to open image.");
	copy = (Pixel*)malloc(sizeof(Pixel) * width * height);
	if (copy == nullptr) return 1;
	std::memcpy(copy, image, width * height);

	/*
	Benchmarking a few different methods. Methods that include a memcpy are
	because they sort in place and therefore lose the original image.

		- Counting Sort
			Using the frequency counting part of the counting sort to sort the
			image pixels. Finds a cut off point and then counts to the
			threshold.

		- std::sort
			Using the standard library sorting algorithm to sort the pixels and
			then fetching the threshold from the sorted array.

		- Nth Element
			Using the nth element algorithm from the algorithms module as a
			faster version of the std::sort.

		- Normal Distribution
			Trying to estimate the threshold value using a normal distribution.
			It's not very good.
		
		- Weighted Estimate
			Linearly interpolating between the average and the min / max
			values to find the threshold value.

		- Uniform Sample
			Looks at every n pixels rather than every pixel. Only good for
			larger images or images with little detail.
	*/

	// Benchmarking.
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	std::chrono::duration<float> duration;

	// Counting Sort.
	start = std::chrono::high_resolution_clock::now();
	Pixel counting_sort_threshold = counting_sort(image, width, height, Ratio);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	display("Counting Sort", counting_sort_threshold, duration.count());

	// std::sort.
	start = std::chrono::high_resolution_clock::now();
	Pixel std_sort_threshold = std_sort(image, width, height, Ratio);
	std::memcpy(image, copy, width * height);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	display("std::sort", std_sort_threshold, duration.count());

	// Nth Element.
	start = std::chrono::high_resolution_clock::now();
	Pixel nth_element_sort_threshold = nth_element_sort(image, width, height, Ratio);
	std::memcpy(image, copy, width * height);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	display("Nth Element", nth_element_sort_threshold, duration.count());
	
	// Normal Estimate.
	start = std::chrono::high_resolution_clock::now();
	Pixel estimate_threshold = normal_estimate(image, width, height, Ratio);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	display("Normal Estimate", estimate_threshold, duration.count());
	
	// Weighted Estimate.
	start = std::chrono::high_resolution_clock::now();
	Pixel weighted_threshold = weighted_estimate(image, width, height, Ratio);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	display("Weighted Estimate", weighted_threshold, duration.count());
	
	// Uniform Sample.
	start = std::chrono::high_resolution_clock::now();
	Pixel uniform_sample_threshold = uniform_sample(image, width, height, SampleRate, Ratio);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	display("Uniform Sample", uniform_sample_threshold, duration.count());
	
	// Export Pixel. Do not change pixels that are on the threshold if they are 0 or max BIT_DEPTH.
	for (size_t pixel = 0; pixel < width * height; pixel++) {
		if (image[pixel] > uniform_sample_threshold) {
			image[pixel] = (1 << BIT_DEPTH) - 1;
		} else if (image[pixel] < uniform_sample_threshold) {
			image[pixel] = 0;
		}
		else if (!(image[pixel] == 0 || image[pixel] == ((1 << BIT_DEPTH) - 1))) {
			image[pixel] = 0;
		}
	}
	stbi_write_png(binary_name, width, height, 1, image, width * sizeof(Pixel));
	return 0;
}