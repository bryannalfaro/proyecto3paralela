/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
/**
 * @file huoghBaseConstante.cu
 * @author Raul Jimenez
 * @author Bryann Alfaro
 * @author Donaldo Garcia
 * @brief Global constant and shared memory implementation of the Hough transform
 * @version 0.1
 * @date 2023-05-09
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"

// global variables defined here
#define M_PI   3.14159265358979323846264338327950288
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void store_Image(char *name, int w, int h, PGMImage inImg);

/**
 * @brief function to store the image
 * @param name name of file
 * @param w width of image
 * @param h height of image
 * @param inImg image to store
 * @return void
 */
void store_Image(char *name, int w, int h, PGMImage inImg)
{
  FILE *pgmimg;
  int i;
  pgmimg = fopen(name, "wb");
  // Writing Magic Number to the File
  fprintf(pgmimg, "P2\n");

  // Writing Width and Height
  fprintf(pgmimg, "%d %d\n", w, h);

  // Writing the maximum gray value
  fprintf(pgmimg, "255\n");
  for (i = 0; i < h * w; i++)
  {
    fprintf(pgmimg, "%d ", inImg.pixels[i]);
  }
  fclose(pgmimg);
}

/**
 * @brief CPU function to calculate the Hough transform
 * @param pic image to process
 * @param w width of image
 * @param h height of image
 * @return void
 */
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2 max radius equivalent to center -> corner
  *acc = new int[rBins * degreeBins];                // accumulator, 90*180/degInc = 9000
  memset(*acc, 0, sizeof(int) * rBins * degreeBins); // initialize with 0
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) // for every pixel
    for (int j = 0; j < h; j++)
    {
      int idx = j * w + i;
      if (pic[idx] > 0)
      {
        int xCoord = i - xCent;
        int yCoord = yCent - j;                       // y-coord has to be reversed
        float theta = 0;                              // angle
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) // add +1 to every angle
        {
          float r = xCoord * cos(theta) + yCoord * sin(theta);
          int rIdx = (r + rMax) / rScale;
          (*acc)[rIdx * degreeBins + tIdx]++; //+1 for this r and theta
          theta += radInc;
        }
      }
    }
}

//Constant memory
__constant__ float dCos[degreeBins];
__constant__ float dSin[degreeBins];

/**
 * @brief GPU kernel. One thread for every pixel. Constant, shared and global memory
 * @param pic image to process
 * @param w width of image
 * @param h height of image
 * @param acc accumulator
 * @param rMax max radius
 * @param rScale scale of radius
 * @return void
 */
 __global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int locID = threadIdx.x; //local ID
  // we get the global ID
  int gloID =  (blockIdx.x * blockDim.x) + threadIdx.x;


  if (gloID > w * h) return;     // if the global id is greater than the number of pixels we return

  int xCent = w / 2;
  int yCent = h / 2;

 // We get the coordinates of the pixel
  //The x coordinate is obtained by means of the modulo operation with the width of the image. doing the remainder we obtain the column and the subtraction is even to centralize the coordinate
  int xCoord = gloID % w - xCent;
  // The y-coordinate is obtained by means of the integer division operation with the width of the image. We do the integer division to obtain the row and the subtraction is to centralize the coordinate
  int yCoord = yCent - gloID / w;

   __shared__ int localAcc[rBins * degreeBins]; // local accumulator
   int i;
  //loop to initialize localAcc
  for (i = locID; i < rBins * degreeBins; i+= blockDim.x){
    localAcc[i] = 0;
  }
  __syncthreads ();

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          // we calculate the radius
          float r = xCoord * dCos[tIdx] + yCoord * dSin[tIdx];
          // we calculate the index of the radius
          int rIdx = (r + rMax) / rScale;
          // Because it is done based on the angles, it may be that at some point they will touch, which is why an atomic add is done
          atomicAdd (localAcc + (rIdx * degreeBins + tIdx), 1);
        }
}
  __syncthreads ();

  //loop to add localAcc to acc
  for (i = locID; i < rBins * degreeBins; i += blockDim.x){
    atomicAdd (acc +i, localAcc[i] );
  }
}


int main (int argc, char **argv)
{
  int i;

  // read image
  PGMImage inImg (argv[1]);

  int *cpuht;
  // Get image dimensions
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // Store the image
  store_Image("result.pgm", w, h, inImg);

// Print out the dimensions of the image
  printf ("Image size is %d x %d\n", w, h);
  cudaEvent_t start, stop;
  // Create events for timing
cudaEventCreate(&start);
cudaEventCreate(&stop);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  // we calculate the value of the sine and cosine of the angles
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // copy to constant memory
  cudaMemcpyToSymbol(dCos, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol(dSin, pcSin, sizeof (float) * degreeBins);


  // setup and copy data from host to device
  unsigned char *dIn, *hIn;
  int *dHough, *hHough;

  hIn = inImg.pixels; //Contains the pixels of the image

  hHough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &dIn, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &dHough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (dIn, hIn, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (dHough, 0, sizeof (int) * degreeBins * rBins);

  // each block handles 256 pixels
  int blockNum = ceil ((double)w * (double)h / (double)256);
  cudaEventRecord(start);
  GPU_HoughTranShared <<< blockNum, 256 >>> (dIn, w, h, dHough, rMax, rScale);
  cudaEventRecord(stop);


  //  we get the results of the GPU and copy them to the host
  cudaMemcpy (hHough, dHough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Tiempo de ejecucion: %f ms\n", milliseconds);


  // we compare the results of the CPU and the GPU
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != hHough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], hHough[i]);
  }
  printf("Done!\n");

  // free memory
  cudaFree (dIn);
  cudaFree (dHough);
  cudaFree (dCos);
  cudaFree (dSin);
  free (hIn);
  free (hHough);
  free (cpuht);
  free (pcCos);
  free (pcSin);


  return 0;
}
