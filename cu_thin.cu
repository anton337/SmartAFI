#include <cuda.h>
#include <cufft.h>
#include <cuda_profiler_api.h>
#include <stdio.h>

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
	return __ldg(ptr);
#else
	return *ptr;
#endif
}

#ifndef PI_8
#define PI_8 0.39269908169
#endif

extern "C"
__global__
void Thin(
int nz
, int ny
, int nx
, float * theta // theta (in)
, float * like // like (in)
, float * thin // thin (out)
)
{
	int kz = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kz < nz && ky+1 < ny && ky >= 1)
	{
		float val;
		int ind;
		for (int kx = 1; kx+1 < nx; kx++)
		{
			ind = (kz*ny + ky)*nx + kx;
      float central = like[ind];
      float orientation = theta[ind];
      if(orientation<PI_8||orientation>7*PI_8)
      {
        // x oriented
        // compare (x,y,z) to (x+1,y,z) and (x-1,y,z)
        if(central>like[ind+1] && central>like[ind-1])
        {
          thin[ind] = like[ind];
        }
        else
        {
          thin[ind] = 0;
        }
      }
      else if(orientation<3*PI_8)
      {
        // xy diagonal
        // compare (x,y,z) to (x+1,y+1,z) and (x-1,y-1,z)
        if(central>like[ind+nx+1] && central>like[ind-nx-1])
        {
          thin[ind] = like[ind];
        }
        else
        {
          thin[ind] = 0;
        }
      }
      else if(orientation<5*PI_8)
      {
        // y oriented
        // compare (x,y,z) to (x,y+1,z) and (x,y-1,z)
        if(central>like[ind+nx] && central>like[ind-nx])
        {
          thin[ind] = like[ind];
        }
        else
        {
          thin[ind] = 0;
        }
      }
      else 
      {
        // -xy diagonal
        // compare (x,y,z) to (x-1,y+1,z) and (x+1,y-1,z)
        if(central>like[ind+nx-1] && central>like[ind-nx+1])
        {
          thin[ind] = like[ind];
        }
        else
        {
          thin[ind] = 0;
        }
      }
		}
	}
}
