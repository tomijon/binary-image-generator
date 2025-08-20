# Binary Image Generator

Explores different methods for converting a greyscale image to a binary image (black and white) based on a given ratio of black to white pixels. The various methods used include:

- Counting Sort
- std::sort
- std::nth_element
- Normal Distribution (estimation)
- Linear Interpolation (estimation)

The estimation algorithms were primarily experimental, and were created to see how accurate they could be in comparison to the sorting methods, while also performing faster. `std::sort` and `std::nth_element` sort in place, and so copying the image data to a pre-allocated buffer was included in the benchmark, due to the likelihood of writing the binary image upon finding the threshold value. The nth_element algorithm and the counting sort algorithm are quite close in terms of performance.

## Building

The program can be built with the `build.bat`. Optionally add the `-O3` flag for optimised build.

## Example Output

```
Counting Sort
    Threshold: 80
    Execution Time: 0.03s
std::sort
    Threshold: 80
    Execution Time: 0.69s
Nth Element
    Threshold: 80
    Execution Time: 0.07s
Normal Estimate
    Threshold: 136
    Execution Time: 0.15s
Weighted Estimate
    Threshold: 66
    Execution Time: 0.00s
```

## Other

This project includes `stb_image.h` and `stb_image_write.h` from the [nothings/stb](https://github.com/nothings/stb) library, which is in the public domain / MIT licensed.