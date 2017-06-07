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
void data_transfer_real_cplx(
int nx
, int ny
, int nz
, float * in
, cufftComplex * out 
)
{
	int kx = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kx < nx && ky < ny)
	{
		for (std::size_t kz = 0; kz < nz; kz++)
		{
			out[(kx*ny + ky)*nz + kz].x = ldg(&in[(kx*ny + ky)*nz + kz]);
			out[(kx*ny + ky)*nz + kz].y = 0;
		}
	}
}

extern "C"
__global__
void data_transfer_cplx_real(
int nx
, int ny
, int nz
, cufftComplex * in
, float * out
)
{
	int kx = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kx < nx && ky < ny)
	{
		int k_1 = nx*ky + kx;
		for (int i = 0; i < nz; i++, k_1 += nx*ny)
		{
			out[k_1] = ldg(&in[k_1]).x;
		}
	}
}
