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

#ifndef PI
#define PI 3.14159265359
#endif

extern "C"
__global__
void Shear(
float center_shift_y
, float center_shift_x
, float shift_y
, float shift_x
, int nz
, int ny
, int nx
, cufftComplex *   input_fk // input
, cufftComplex * sheared_fk // sheared
)
{
	int kx = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kx < nx && ky < ny)
	{
		int k_1 = nx*ky + kx;
		float sz = 0;
		float dz = 1.0f / nz;
		cufftComplex kernel;
		for (int i = 0,kz=0; i < nz; i++, k_1 += nx*ny,kz++)
		{
			sz += dz;
			kernel.x = cosf((2 * PI*(kx*(center_shift_x + sz*shift_x) + ky*(center_shift_y + sz*shift_y))) * dz);
			kernel.y = -sinf((2 * PI*(kx*(center_shift_x + sz*shift_x) + ky*(center_shift_y + sz*shift_y))) * dz);
			sheared_fk[k_1].x = ldg(&input_fk[k_1].x)*kernel.x - ldg(&input_fk[k_1].y)*kernel.y;
			sheared_fk[k_1].y = ldg(&input_fk[k_1].x)*kernel.y + ldg(&input_fk[k_1].y)*kernel.x;
		}
	}
}

extern "C"
__global__
void ShearTimeDomain(
float center_shift_y
, float center_shift_x
, float shift_y
, float shift_x
, int nz
, int ny
, int nx
, float *   input // input
, float * sheared // sheared
)
{
	int kz = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kz < nz && ky+1 < ny && ky)
	{
		float sz = (float)(kz) / (nz);
		int k = nx*ny*kz + nx*ky;
		float w0, w1, _w0, _w1, w00, w01, w10, w11;
		float X, Y;
		for (int i = 0, kx = 0; i+1 < nx; i++, kx++)
		{
			X = kx + sz*shift_x + center_shift_x;
			Y = ky + sz*shift_y + center_shift_y;
			w0 = X - (int)X; w1 = Y - (int)Y;
			_w0 = 1 - w0; _w1 = 1 - w1;
			//w0 *= w0;w1 *= w1;w0 *= w0;w1 *= w1;
			//_w0 *= _w0;_w1 *= _w1;_w0 *= _w0;_w1 *= _w1;
			//s0 = 1.0f / (w0 + _w0);s1 = 1.0f / (w1 + _w1);
			//w0 *= s0;w1 *= s1;_w0 *= s0;_w1 *= s1;
			w00 = _w0*_w1; w01 = _w0*w1; w10 = w0*_w1; w11 = w0*w1;
			sheared[k + i] = w00*ldg(&input[k + i]) + w10*ldg(&input[k + i + 1]) + w01*ldg(&input[k + i + nx]) + w11*ldg(&input[k + i + 1 + nx]);
		}
	}
}
