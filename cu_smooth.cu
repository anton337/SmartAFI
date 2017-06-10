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
void zSmooth(
int nz
, int ny
, int nx
, float alpha
, float * data // data (in/out)
)
{
	int kx = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kx < nx && ky < ny)
	{
		
		int k_0 = nx*ky + kx;
		int k_1 = nx*ny + nx*ky + kx;
		for (int i = 0; i + 1 < nz; i++, k_0 += nx*ny, k_1 += nx*ny)
		{
			data[k_1] += data[k_0] * alpha;
		}
		k_0 -= nx*ny; k_1 -= nx*ny;
		for (int i = 0; i + 1 < nz && k_0 >= 0 && k_1 >= 0; i++, k_0 -= nx*ny, k_1 -= nx*ny)
		{
			data[k_0] += data[k_1] * alpha;
		}
		
	}
}
