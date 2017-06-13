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

extern "C"
__global__
void SemblanceDiv(
int nx
, int ny
, int nz
, float *   numerator // num (in)
, float * denominator // den (in)
, float * output // (out)
)
{
	int kx = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kx < nx && ky < ny)
	{
		int k_1 = nx*ky + kx;
		for (int i = 0; i < nz; i++, k_1 += nx*ny)
		{
			output[k_1] = ldg(&numerator[k_1]) / (ldg(&denominator[k_1]) + 0.001f);
			output[k_1] *= output[k_1];
			output[k_1] *= output[k_1];
			output[k_1] *= output[k_1];
			output[k_1] = 1.0f - output[k_1];
		}
	}
}

extern "C"
__global__
void SemblanceMax(
int nx
, int ny
, int nz
, float theta
, float * data // (in)
, float * max_data // (out)
, float * max_theta // (out)
)
{
	int kx = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kx < nx && ky < ny)
	{
		int k_1 = nx*ky + kx;
		for (int i = 0; i < nz; i++, k_1 += nx*ny)
		{
			if (ldg(&data[k_1]) > max_data[k_1])
			{
				max_data[k_1] = ldg(&data[k_1]);
        max_theta[k_1] = theta;
			}
		}
	}
}

extern "C"
__global__
void Semblance(
int win
, int nz
, int ny
, int nx
, float * data // data (in)
, float * num // num (out)
, float * den // den (out)
)
{
	int kz = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kz < nz && ky+win < ny && ky >= win)
	{
		int factor = 2 * win + 1;
		factor *= factor;
		float val;
		int ind;
		for (int kx = win; kx+win < nx; kx++)
		{
			ind = (kz*ny + ky)*nx + kx;
			for (int dy = -win; dy <= win; dy++)
			{
				for (int dx = -win; dx <= win; dx++)
				{
					val = ldg(&data[(kz*ny + ky+dy)*nx + kx+dx]);
					num[ind] += val;
					den[ind] += val*val;
				}
			}
			num[ind] *= num[ind];
			den[ind] *= factor;
		}
	}
}


