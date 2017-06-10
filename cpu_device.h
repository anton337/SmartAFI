#ifndef cpu_device_h
#define cpu_device_h

#include "cpu_array.h"

#include "freq_cpu_array.h"

class CPUDevice : public ComputeDevice
{
public:
	CPUDevice(std::string _token)
		: ComputeDevice(_token)
	{

	}
	void fft(
		int nz
		, int ny
		, int nx
		, Token in_token
		, Token out_token
		)
	{
		std::cout << "cpu fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	bool create(
		Token token
		, int nx
		, int ny
		, int nz
		, float * dat = NULL
		, bool freq = false
		, bool keep_data = false
		)
	{
		if (freq)
		{
			MESSAGE_ASSERT(token.type == FREQ_GPU,"create data array : type mismatch");
			MESSAGE_ASSERT(token.nx == nx, "create data array : nx mismatch");
			MESSAGE_ASSERT(token.ny == ny, "create data array : ny mismatch");
			MESSAGE_ASSERT(token.nz == nz, "create data array : nz mismatch");
			return createArray<freqCPUArray>(token, nx,ny,nz, dat, keep_data);
		}
		else
		{
			MESSAGE_ASSERT(token.type == CPU, "create data array : type mismatch");
			MESSAGE_ASSERT(token.nx == nx, "create data array : nx mismatch");
			MESSAGE_ASSERT(token.ny == ny, "create data array : ny mismatch");
			MESSAGE_ASSERT(token.nz == nz, "create data array : nz mismatch");
			return createArray<CPUArray>(token, nx,ny,nz, dat, keep_data);
		}
	}

	void compute_semblance(int win, int nz, int ny, int nx, Token transpose, Token numerator, Token denominator)
	{
		MESSAGE_ASSERT(transpose.type == CPU, "semblance : transpose : type mismatch");
		MESSAGE_ASSERT(numerator.type == CPU, "semblance : numerator : type mismatch");
		MESSAGE_ASSERT(denominator.type == CPU, "semblance : denominator : type mismatch");
		float* dat_arr = (float*)get(transpose);
		float* num_arr = (float*)get(numerator);
		float* den_arr = (float*)get(denominator);
		semblance(win, nx, ny, nz, dat_arr, num_arr, den_arr);
	}

	void compute_transpose(int nx, int ny, int nz, Token input, Token transpose)
	{
		MESSAGE_ASSERT(input.type == CPU, "transpose : input : type mismatch");
		MESSAGE_ASSERT(transpose.type == CPU, "transpose : transpose : type mismatch");
		float* input_arr = (float*)get(input);
		float* transpose_arr = (float*)get(transpose);
		transpose_constY(nx
			, ny
			, nz
			, input_arr
			, transpose_arr
			);
	}

	void compute_convolution_rotation_kernel(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token rotation_kernel_token_freq
		, Token signal_token_freq
		, Token rotated_signal_token_freq
		)
	{
		std::cout << "cpu rotation not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_shear(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, Token rotated_freq
		, Token sheared_freq
		)
	{
		std::cout << "cpu compute shear not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_shear_time(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, Token rotated_time
		, Token sheared_time
		)
	{
		std::cout << "cpu compute shear time not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void inv_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token sheared_freq
		, Token sheared_time
		)
	{
		std::cout << "cpu inv fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void init_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		)
	{
		std::cout << "cpu init fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void destroy_fft()
	{
		std::cout << "cpu destroy fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_zsmooth(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token data
		)
	{
		std::cout << "cpu compute zsmooth not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_fault_likelihood(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token num
		, Token den
		, Token data
		)
	{
		std::cout << "cpu compute fault likelihood not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void update_maximum(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token fh
		, Token optimum_fh
		//, Token optimum_th
		//, Token optimum_phi
		)
	{
		std::cout << "cpu update maximum not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

};

#endif

