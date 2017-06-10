#ifndef cuda_stuff_h
#define cuda_stuff_h

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#ifndef _checkCudaErrors
#define _checkCudaErrors(err)  ___checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void ___checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err)
	{
		fprintf(stdout, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
			err, getCudaDrvErrorString(err), file, line);
		char ch;
		std::cin >> ch;
	}
}
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		char ch;
		std::cin >> ch;
		if (abort) exit(code);
	}
}

// CUDA STUFF

// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}

inline void cufftErrchk(cufftResult error, bool abort = true)
{
	if (error != CUFFT_SUCCESS)
	{
		fprintf(stderr, "cufft assert: %s\n", _cudaGetErrorEnum(error));
		char ch;
		std::cin >> ch;
		if (abort) exit(error);
	}
}

#endif

