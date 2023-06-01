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
 * @file huoghBase.cu
 * @author Raul Jimenez
 * @author Bryann Alfaro
 * @author Donaldo Garcia
 * @brief Implementacion de la transformada de Hough en CUDA
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
// declaracion de variables globales
#define M_PI 3.14159265358979323846264338327950288
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void store_Image(char *name, int w, int h, PGMImage inImg);

/**
 * @brief pintar la imagen en un pgm
 * @param name nombre del archivo
 * @param w ancho de la imagen
 * @param h alto de la imagen
 * @param inImg imagen a pintar
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

// The CPU function returns a pointer to the accummulator
/**
 * @brief funcion de CPU para la transformada de Hough
 * @param pic imagen a procesar
 * @param w ancho de la imagen
 * @param h alto de la imagen
 * @return void
 */
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];                // el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset(*acc, 0, sizeof(int) * rBins * degreeBins); // inicializamos con 0s
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) // por cada pixel
    for (int j = 0; j < h; j++)
    {
      int idx = j * w + i;
      if (pic[idx] > 0) // si pasa thresh, entonces lo marca
      {
        int xCoord = i - xCent;
        int yCoord = yCent - j;                       // y-coord has to be reversed
        float theta = 0;                              // angulo
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) // se agrega +1 a cada angulo
        {
          float r = xCoord * cos(theta) + yCoord * sin(theta);
          int rIdx = (r + rMax) / rScale;
          (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
          theta += radInc;
        }
      }
    }
}

/**
 * @brief GPU kernel. Un hilo por pixel de la imagen es creado. La memoria del acumulador debe ser asignada por el host en memoria global
 * @param pic imagen a procesar
 * @param w ancho de la imagen
 * @param h alto de la imagen
 * @param acc acumulador
 * @param rMax radio maximo
 * @param rScale escala del radio
 * @param dCos cosenos
 * @param dSin senos
 * @return void
 */
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *dCos, float *dSin)
{
  // Obtenemos el global ID
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;

  if (gloID > w * h)
    return; // en caso de que el id global sea mayor se cierra

  int xCent = w / 2;
  int yCent = h / 2;

  // Obtenemos las coordenadas del pixel
  // Se obtiene la coordenada x, por medio de la operacion modulo con el ancho de la imagen. haciendo el residuo obtenemos la columna y la resta es par centralizar la coordenada
  int xCoord = gloID % w - xCent;
  // Se obtiene la coordenada y, por medio de la operacion division entera con el ancho de la imagen. Hacemos la division entera para obtener la fila y la resta es par centralizar la coordenada
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      // calculamos el radio
      float r = xCoord * dCos[tIdx] + yCoord * dSin[tIdx];
      // calculamos el indice del radio
      int rIdx = (r + rMax) / rScale;
      // Debido a que se hace en base a los angulos puede ser que en algun punto se vayan a tocar es por esto que se hace un atomic add
      atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}

int main(int argc, char **argv)
{
  int i, j;

  // obtenemos la imagen
  PGMImage inImg(argv[1]);

  int *cpuht;
  // Obtenemos el ancho y el alto de la imagen
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // Guardamos la imagen original
  store_Image("result.pgm", w, h, inImg);

  // Imprimimos el ancho y el alto de la imagen
  printf("Image size is %d x %d\n", w, h);
  cudaEvent_t start, stop;
  // Creamos los eventos para calcular el tiempo
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *dCos;
  float *dSin;

  // Reservamos memoria en el device
  cudaMalloc((void **)&dCos, sizeof(float) * degreeBins);
  cudaMalloc((void **)&dSin, sizeof(float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
  float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
  float rad = 0;
  // calculamos los valores de los cosenos y senos
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // copiamos la data del device
  cudaMemcpy(dCos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(dSin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

  // sconfigurar y copiar la data al device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

  cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
  cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
  cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

  // cada bloque tiene 256 hilos y 1 hilo tiene 1 pixel
  int blockNum = ceil((double)w * (double)h / (double)256);
  cudaEventRecord(start);
  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, dCos, dSin);
  cudaEventRecord(stop);

  // se obtienen los resultados del device
  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Tiempo de ejecucion: %f ms\n", milliseconds);

  // comparamos los resultados de CPU y GPU
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");

  // Funciones de limpieza
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(dCos);
  cudaFree(dSin);
  free(h_in);
  free(h_hough);
  free(cpuht);
  free(pcCos);
  free(pcSin);

  return 0;
}
