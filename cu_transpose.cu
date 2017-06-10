#include <cuda.h>


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
		int k_0 = kx + ky*nx;
		int k_1 = kx*ny*nz + ky*nz;
		int i = 0;
		for (; i < nz; i++, k_0+=ny*nx, k_1++)
		{
			out[k_0] = ldg(&in[k_1]);
		}
	}
	
}
