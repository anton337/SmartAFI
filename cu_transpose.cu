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
void transpose_constY(
int nx
, int ny
, int nz
, float * in 
, float * out // XYZ -> ZYX
)
{
	int kx = blockIdx.x*blockDim.x + threadIdx.x;
	int ky = blockIdx.y*blockDim.y + threadIdx.y;
	if (kx < nx && ky < ny)
	{
		//  input is ordered by {X,Y,Z}
		// output is ordered by {Z,Y,X}
		// out[z*num_y*num_x + y*num_x + x] := in[x*num_y*num_z + y*num_z + z]
		//float *  in_xy = &in[kx*ny*nz + ky*nz];
		for (std::size_t kz = 0; kz < nz; kz++)
		{
			out[(kz*ny + ky)*nx + kx] = ldg(&in[(kx*ny + ky)*nz + kz]);
		}
	}
}
