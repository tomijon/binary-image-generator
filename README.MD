# Binary Image Generator

Explores different methods for converting a greyscale image to a binary image (black and white) based on a given ratio of black to white pixels. The various methods used include:

- Counting Sort
- std::sort
- std::nth_element
- Normal Distribution (estimation)
- Linear Interpolation (estimation)
- Uniform Sampling

The estimation algorithms were primarily experimental, and were created to see how accurate they could be in comparison to the sorting methods, while also performing faster. `std::sort` and `std::nth_element` sort in place, and so copying the image data to a pre-allocated buffer was included in the benchmark, due to the likelihood of writing the binary image upon finding the threshold value. The nth_element algorithm and the counting sort algorithm are quite close in terms of performance. Uniform sampling is very good, but can produce incorrect results (although the incorrect results are close to the correct values).

## Building

The program can be built with the `build.bat`.  Alternatively, `g++ main.cpp -std=c++20 -o main.exe`. Optionally add the `-O3` flag for optimised build: `g++ main.cpp -std=c++20 -O3 -o main.exe`

## Example Output*

```
Counting Sort
    Threshold: 88
    Execution Time: 0.11s
std::sort
    Threshold: 88
    Execution Time: 1.33s
Nth Element
    Threshold: 88
    Execution Time: 0.12s
Normal Estimate
    Threshold: 139
    Execution Time: 0.24s
Weighted Estimate
    Threshold: 73
    Execution Time: 0.02s
Uniform Sample
    Threshold: 88
    Execution Time: 0.01s
```

*Without -O3 flag.

## Other

This project includes `stb_image.h` and `stb_image_write.h` from the [nothings/stb](https://github.com/nothings/stb) library, which is in the public domain / MIT licensed.