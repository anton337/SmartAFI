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
void Hadamard(
int nx
, int ny
, int nz
, cufftComplex *  input_fk // input
, cufftComplex * kernel_fk // kernel
, cufftComplex * output_fk // output
)
{
	int kz = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kz < nz && ky < ny)
	{
		int k = nx*ny*kz + nx*ky;
		for (int i = 0; i < nx; i++)
		{
			output_fk[k+i].x = ldg(&input_fk[k+i].x)*ldg(&kernel_fk[k+i].x) - ldg(&input_fk[k+i].y)*ldg(&kernel_fk[k+i].y);
			output_fk[k+i].y = ldg(&input_fk[k+i].x)*ldg(&kernel_fk[k+i].y) + ldg(&input_fk[k+i].y)*ldg(&kernel_fk[k+i].x);
		}
	}
}

extern "C"
__global__
void Hadamard_slice_kernel(
int nx
, int ny
, int nz
, cufftComplex *  input_fk // input
, cufftComplex * kernel_fk // kernel
, cufftComplex * output_fk // output
)
{
	int kz = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kz < nz && ky < ny)
	{
		int k = nx*ny*kz + nx*ky;
		int k_slice = nx*ky;
		for (int i = 0; i < nx; i++)
		{
			output_fk[k + i].x = ldg(&input_fk[k + i].x)*ldg(&kernel_fk[k_slice + i].x) - ldg(&input_fk[k + i].y)*ldg(&kernel_fk[k_slice + i].y);
			output_fk[k + i].y = ldg(&input_fk[k + i].x)*ldg(&kernel_fk[k_slice + i].y) + ldg(&input_fk[k + i].y)*ldg(&kernel_fk[k_slice + i].x);
		}
	}
}

