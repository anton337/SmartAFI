#ifndef gpu_device_h
#define gpu_device_h

#include "cpu_device.h"

#include "gpu_array.h"

#include "freq_gpu_array.h"

class GPUDevice : public CPUDevice
{
	std::size_t ind;

	std::size_t totalGlobalMem;

	CUcontext cuContext;

	CUdevice cuDevice;
	CUmodule cuModule;

	char deviceName[100];

	cufftHandle plan;
	cufftHandle plan1;

public:
	GPUDevice(std::string _name,std::size_t _ind)
		: CPUDevice(_name)
	{
		ind = _ind;
		int major = 0, minor = 0;
		printf("device %d:\n", ind);
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		// get compute capabilities and the devicename
		_checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
		_checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
		printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
		_checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
		printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
		printf("  64-bit Memory Address:             %s\n", (totalGlobalMem >(unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");
		create_context();
		load_convolution_rotation_kernel();
		load_shear();
		load_shear_time();
		load_zsmooth();
		load_semblance();
		load_semblance_div();
		load_semblance_max();
		load_transpose();
		load_data_transfer_real_cplx();
		load_data_transfer_cplx_real();
	}
	~GPUDevice()
	{
		destroy_context();
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
			MESSAGE_ASSERT(token.type == FREQ_GPU, "create data array : type mismatch");
			MESSAGE_ASSERT(token.nx == nx, "create data array : nx mismatch");
			MESSAGE_ASSERT(token.ny == ny, "create data array : ny mismatch");
			MESSAGE_ASSERT(token.nz == nz, "create data array : nz mismatch");
			return createArray<freqGPUArray>(token, nx,ny,nz, dat, keep_data);
		}
		else
		{
			MESSAGE_ASSERT(token.type == GPU, "create data array : type mismatch");
			MESSAGE_ASSERT(token.nx == nx, "create data array : nx mismatch");
			MESSAGE_ASSERT(token.ny == ny, "create data array : ny mismatch");
			MESSAGE_ASSERT(token.nz == nz, "create data array : nz mismatch");
			return createArray<GPUArray>(token, nx,ny,nz, dat, keep_data);
		}
	}
private:
	CUfunction convolution_rotation_kernel;
	CUfunction shear;
	CUfunction shear_time;
	CUfunction zsmooth;
	CUfunction semblance;
	CUfunction semblance_div;
	CUfunction semblance_max;
	CUfunction transpose_constY;
	CUfunction data_transfer_real_cplx;
	CUfunction data_transfer_cplx_real;
	void create_context()
	{
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		_checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
	}
	void destroy_context()
	{
		_checkCudaErrors(cuCtxDestroy(cuContext));
	}
	void set_context()
	{
		_checkCudaErrors(cuCtxSetCurrent(cuContext));
	}
	void load_convolution_rotation_kernel()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_fft.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_fft.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&convolution_rotation_kernel, cuModule, "Hadamard_slice_kernel"));
	}
	void load_shear()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_shear.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_shear.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&shear, cuModule, "Shear"));
	}
	void load_shear_time()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_shear.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_shear.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&shear_time, cuModule, "ShearTimeDomain"));
	}
	void load_zsmooth()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_smooth.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_smooth.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&zsmooth, cuModule, "zSmooth"));
	}
	void load_semblance_div()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_semblance.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_semblance.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&semblance_div, cuModule, "SemblanceDiv"));
	}
	void load_semblance_max()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_semblance.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_semblance.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&semblance_max, cuModule, "SemblanceMax"));
	}
	void load_semblance()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_semblance.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_semblance.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&semblance, cuModule, "Semblance"));
	}
	void load_transpose()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_transpose.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_transpose.fatbin"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_transpose.fatbin"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_transpose.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&transpose_constY, cuModule, "transpose_constY"));
	}
	void load_data_transfer_real_cplx()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_data_transfer.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_data_transfer.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&data_transfer_real_cplx, cuModule, "data_transfer_real_cplx"));
	}
	void load_data_transfer_cplx_real()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_data_transfer.ptx"));
		//_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_data_transfer.fatbin"));
		_checkCudaErrors(cuModuleGetFunction(&data_transfer_cplx_real, cuModule, "data_transfer_cplx_real"));
	}

public:
	void compute_convolution_rotation_kernel(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token rotated_kernel_token_freq
		, Token signal_token_freq
		, Token rotated_signal_token_freq
		)
	{
		set_context();
		int sizex = nx;
		int sizey = ny;
		int sizez = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		cufftComplex * fk_input  = (cufftComplex*)get(signal_token_freq);
		cufftComplex * fk_kernel = (cufftComplex*)get(rotated_kernel_token_freq);
		cufftComplex * fk_output = (cufftComplex*)get(rotated_signal_token_freq);
		void *args[6] = { &sizex, &sizey, &sizez, &fk_input, &fk_kernel, &fk_output };
		{
			_checkCudaErrors(cuLaunchKernel(convolution_rotation_kernel,
				grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				NULL, args, NULL));
			_checkCudaErrors(cuCtxSynchronize());
		}
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
		set_context();
		int sizex = nx;
		int sizey = ny;
		int sizez = nz;
		float SHEAR_Y = shear_y;
		float SHEAR_X = shear_x;
		float CENTER_SHEAR_Y = 0;
		float CENTER_SHEAR_X = 0;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		cufftComplex * fk_rotated = (cufftComplex*)get(rotated_freq);
		cufftComplex * fk_sheared = (cufftComplex*)get(sheared_freq);
		void *args[9] = { &CENTER_SHEAR_Y, &CENTER_SHEAR_X, &SHEAR_Y, &SHEAR_X, &sizez, &sizey, &sizex, &fk_rotated, &fk_sheared };
		{
			_checkCudaErrors(cuLaunchKernel(shear,
				grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				NULL, args, NULL));
			_checkCudaErrors(cuCtxSynchronize());
		}
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
		set_context();
		int sizex = nx;
		int sizey = ny;
		int sizez = nz;
		float SHEAR_Y = shear_y;
		float SHEAR_X = shear_x;
		float CENTER_SHEAR_Y = 0;
		float CENTER_SHEAR_X = 0;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		float * rotated = (float*)get(rotated_time);
		float * sheared = (float*)get(sheared_time);
		void *args[9] = { &CENTER_SHEAR_Y, &CENTER_SHEAR_X, &SHEAR_Y, &SHEAR_X, &sizez, &sizey, &sizex, &rotated, &sheared };
		{
			_checkCudaErrors(cuLaunchKernel(shear_time,
				grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				NULL, args, NULL));
			_checkCudaErrors(cuCtxSynchronize());
		}
	}

	void init_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		)
	{
		std::cout << "init fft" << std::endl;
		{
			int n[2] = { ny, nx };
			cufftErrchk(cufftPlanMany(&plan, 2, n,
				NULL, 1, 0,
				NULL, 1, 0,
				CUFFT_C2C, nz)
				);
		}
		{
			int n[2] = { ny, nx };
			cufftErrchk(cufftPlanMany(&plan1, 2, n,
				NULL, 1, 0,
				NULL, 1, 0,
				CUFFT_C2C, 1)
				);
		}
		std::cout << "done init fft" << std::endl;
	}

	void destroy_fft()
	{
		cufftErrchk(cufftDestroy(plan));
		cufftErrchk(cufftDestroy(plan1));
	}

	void inv_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token sheared_freq
		, Token sheared_time
		)
	{
		set_context();
		
		cufftComplex * a_fft = (cufftComplex*)get(sheared_freq);
		Token out_cplx_token(sheared_freq.name + "_cplx",FREQ_GPU,nz,ny,nx);
		cufftComplex * a_cplx_out = (cufftComplex*)get(out_cplx_token);
		
		if (nz==1)
			cufftErrchk(cufftExecC2C(plan1, a_fft, a_cplx_out, CUFFT_INVERSE));
		else
			cufftErrchk(cufftExecC2C(plan, a_fft, a_cplx_out, CUFFT_INVERSE));

		float * a_out = (float*)get(sheared_time);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[5] = { &NX, &NY, &NZ, &a_cplx_out, &a_out };
		_checkCudaErrors(cuLaunchKernel(data_transfer_cplx_real,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
		
	}

	void compute_zsmooth(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token data
		)
	{
		set_context();
		float * a_data = (float*)get(data);
		float alpha = 0.90;
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[5] = { &NZ, &NY, &NX, &alpha, &a_data };
		_checkCudaErrors(cuLaunchKernel(zsmooth,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
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
		set_context();
		float * a_num = (float*)get(num);
		float * a_den = (float*)get(den);
		float * a_data = (float*)get(data);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[6] = { &NZ, &NY, &NX, &a_num, &a_den, &a_data };
		_checkCudaErrors(cuLaunchKernel(semblance_div,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
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
		set_context();
		float * a_data = (float*)get(fh);
		float * a_optimum = (float*)get(optimum_fh);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[6] = { &NZ, &NY, &NX, &a_data, &a_optimum };
		_checkCudaErrors(cuLaunchKernel(semblance_max,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
	}
	void compute_semblance(int win, int nz, int ny, int nx, Token transpose, Token numerator, Token denominator)
	{
		set_context();
		int WIN = win;
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		float * a_data = (float*)get(transpose);
		float * a_num = (float*)get(numerator);
		float * a_den = (float*)get(denominator);
		void *args[7] = { &WIN, &NX, &NY, &NZ, &a_data, &a_num, &a_den };
		_checkCudaErrors(cuLaunchKernel(semblance,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
		//{
		//	float * transpose_arr = new float[nx*ny*nz];
		//	get_output(transpose, transpose_arr);
		//	Token cpu_transpose(transpose.name + "_cpu", CPU, nz, ny, nx);
		//	CPUDevice::create(cpu_transpose, nz, ny, nx, transpose_arr);

		//	float * numerator_arr = new float[nx*ny*nz];
		//	get_output(numerator, numerator_arr);
		//	Token cpu_numerator(numerator.name + "_cpu", CPU, nz, ny, nx);
		//	CPUDevice::create(cpu_numerator, nz, ny, nx, numerator_arr);

		//	float * denominator_arr = new float[nx*ny*nz];
		//	get_output(denominator, denominator_arr);
		//	Token cpu_denominator(denominator.name + "_cpu", CPU, nz, ny, nx);
		//	CPUDevice::create(cpu_denominator, nz, ny, nx, denominator_arr);

		//	CPUDevice::compute_semblance(win, nz, ny, nx, cpu_transpose, cpu_numerator, cpu_denominator);
		//	set(numerator, numerator_arr);
		//	set(denominator, denominator_arr);
		//	remove(cpu_transpose);
		//	remove(cpu_numerator);
		//	remove(cpu_denominator);
		//}
    

		remove(transpose);
	}
	void compute_transpose(int nx,int ny,int nz,Token input,Token transpose)
	{
		set_context();
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		float * a_input = (float*)get(input);
		
		float * a_transpose = (float*)get(transpose);
		void *args[5] = { &NX, &NY, &NZ, &a_input, &a_transpose };
		_checkCudaErrors(cuLaunchKernel(transpose_constY,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
		//if (cuCtxSynchronize() != CUDA_SUCCESS)
		//{
		//	float * input_arr = new float[nx*ny*nz];
		//	get_output(input, input_arr);
		//	Token cpu_input(input.name + "_cpu", CPU, nz, ny, nx);
		//	CPUDevice::create(cpu_input, nz, ny, nx, input_arr);
		//	float * transpose_arr = new float[nx*ny*nz];
		//	get_output(transpose, transpose_arr);
		//	Token cpu_transpose(transpose.name + "_cpu", CPU, nz, ny, nx);
		//	CPUDevice::create(cpu_transpose, nz, ny, nx, transpose_arr);
		//	CPUDevice::compute_transpose(nx, ny, nz, cpu_input, cpu_transpose);
		//	set(transpose, transpose_arr);
		//	remove(cpu_transpose);
		//	remove(cpu_input);
		//}
	}
	void fft(int nz, int ny, int nx, Token in, Token out)
	{
		set_context();
		float * a_input = (float*)get(in);
		Token in_cplx_token(in.name + "_cplx",FREQ_GPU, nz, ny, nx);
		create(in_cplx_token, nz, ny, nx, NULL, true);
		create(out, nz, ny, nx, NULL, true);
		cufftComplex * in_cplx = (cufftComplex*)get(in_cplx_token);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[5] = { &NX, &NY, &NZ, &a_input, &in_cplx };
		_checkCudaErrors(cuLaunchKernel(data_transfer_real_cplx,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
		cufftComplex * a_fft = (cufftComplex*)get(out);
		
		if (nz == 1)
			cufftErrchk(cufftExecC2C(plan1, in_cplx, a_fft, CUFFT_FORWARD));
		else
			cufftErrchk(cufftExecC2C(plan, in_cplx, a_fft, CUFFT_FORWARD));

		//_checkCudaErrors(cuCtxSynchronize());
		remove(in_cplx_token);
		remove(in);
	}
	
};

#endif

