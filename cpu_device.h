#ifndef cpu_device_h
#define cpu_device_h

#include "cpu_array.h"

#include "freq_cpu_array.h"

class CPUDevice : public ComputeDevice
{
	void fft(int nz, int ny, int nx, std::string in_name, std::string out_name)
	{
		std::cout << "cpu fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	bool create(std::string name, int nx, int ny, int nz, float * dat = NULL, bool freq = false, bool keep_data = false)
	{
		return (freq) ? createArray<freqCPUArray>(name, nx*ny*nz, dat, keep_data)
			: createArray<CPUArray>(name, nx*ny*nz, dat, keep_data);
	}

	void compute_semblance(int win, int nz, int ny, int nx, std::string transpose, std::string numerator, std::string denominator)
	{
		std::cout << "cpu compute semblance not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_transpose(int nx, int ny, int nz, std::string input, std::string transpose)
	{
		std::cout << "cpu transpose not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_convolution_rotation_kernel(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string rotation_kernel_name_freq
		, std::string signal_name_freq
		, std::string rotated_signal_name_freq
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
		, std::string rotated_freq
		, std::string sheared_freq
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
		, std::string rotated_time
		, std::string sheared_time
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
		, std::string sheared_freq
		, std::string sheared_time
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
		, std::string data
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
		, std::string num
		, std::string den
		, std::string data
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
		, std::string fh
		, std::string optimum_fh
		//, std::string optimum_th
		//, std::string optimum_phi
		)
	{
		std::cout << "cpu update maximum not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

};

#endif

