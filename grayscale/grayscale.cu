#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void grayscale(unsigned char *d_out, unsigned char *d_in, int width, int height, int channels) {
    // get the row and column of the pixel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // check if the pixel is within the image
    if (row < height && col < width) {
        // get the index of the pixel
        int idx = (row * width + col) * channels;

        // get the pixel values
        unsigned char r = d_in[idx];
        unsigned char g = d_in[idx + 1];
        unsigned char b = d_in[idx + 2];

        // convert the pixel to grayscale
        unsigned char gray = 0.21f * r + 0.71f * g + 0.07f * b;

        // write the grayscale pixel to the output image
        d_out[idx] = gray;
        d_out[idx + 1] = gray;
        d_out[idx + 2] = gray;
    }
}

int main(int argc, char const *argv[])
{
    /* code */
    // get the width, height and number of channels of the image

    int width = 0; 
    int height = 0;
    int channels = 0;

    unsigned char *img = stbi_load("1714787723123.jpeg", &width, &height, &channels, 0); // load the image

    if (img == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }

    const char *input_image = "1714787723123.jpeg";
    const char *output_image = "1714787723123_grayscale.jpeg";

    // load image 
    unsigned char *h_img = stbi_load(input_image, &width, &height, &channels, 0);

    // size of the input image in bytes
    size_t img_size = width * height * channels;

    // allocate memory for the output image
    unsigned char *h_out_img = (unsigned char *) malloc(img_size);

    // declare GPU memory pointers
    unsigned char *d_in;
    unsigned char *d_out;

    // allocate memory on the device
    cudaMalloc((void **) &d_in, img_size);
    cudaMalloc((void **) &d_out, img_size);

    // copy the input image from the host to the device

    cudaMemcpy(d_in, h_img, img_size, cudaMemcpyHostToDevice);

    // launch the kernel

    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    grayscale<<<grid, block>>>(d_out, d_in, width, height, channels);

    // copy the output image from the device to the host
    cudaMemcpy(h_out_img, d_out, img_size, cudaMemcpyDeviceToHost);

    // save the output image
    stbi_write_jpg(output_image, width, height, channels, h_out_img, 100);

    // free the memory
    stbi_image_free(h_img);
    stbi_image_free(h_out_img);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;

}

