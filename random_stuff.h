#ifndef random_stuff_h
#define random_stuff_h

template<typename T>
void compute_shear(CUfunction fun,int ind,T * input_data, T * output_data, int nx, int ny, int nz, T shift_x, T shift_y)
{
	// shear each z slice by constant vector
	// the data order is {Z,Y,X}, where Z is the slowest dimension, and X is the fastest
	//
	
	cufftHandle plan;
	cufftComplex * input_mem = new cufftComplex[nx*ny*nz];
	cufftComplex * kernel_mem = new cufftComplex[nx*ny*nz];
	cufftComplex * output_mem = new cufftComplex[nx*ny*nz];
	float center_shift_x = 0;
	float center_shift_y = 0;
	T sz;
	for (int z = 0, k = 0; z<nz; z++)
	{
		sz = (T)(z) / (nz);
		for (int y = 0; y < ny; y++)
			for (int x = 0; x < nx; x++, k++)
			{
				kernel_mem[k].x =  cos((2 * M_PI*(x*center_shift_x + y*center_shift_y + x*sz*shift_x + y*sz*shift_y)) / ny);
				kernel_mem[k].y = -sin((2 * M_PI*(x*center_shift_x + y*center_shift_y + x*sz*shift_x + y*sz*shift_y)) / ny);
			}
	}
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				input_mem[k].x = input_data[k];
				input_mem[k].y = 0;
			}
	cufftComplex * input;
	cufftComplex * output;
	cufftComplex * fk_input;
	cufftComplex * fk_kernel;
	cufftComplex * fk_output;
	cudaMalloc((void**)&input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&output, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_kernel, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_output, sizeof(cufftComplex)*nx*ny*nz);
	int sizex = nx; 
	int sizey = ny;
	int sizez = nz;
	int n[2] = { ny, nx };
	cufftErrchk(cufftPlanMany(&plan, 2, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, nz)
		);
	int block_x = BLOCK_SIZE;
	int block_y = BLOCK_SIZE;
	int block_z = 1;
	dim3 block = dim3(block_x, block_y, block_z);
	//dim3 grid = dim3((nz + block_z - 1) / block_z, (ny + block_y - 1) / block_y, (nx + block_x - 1) / block_x);
	dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
	std::cout <<  grid.x << " " <<  grid.y << " " <<  grid.z << std::endl;
	std::cout << block.x << " " << block.y << " " << block.z << std::endl;
	void *args[6] = { &sizex, &sizey, &sizez, &fk_input, &fk_kernel, &fk_output };
	gpuErrchk(cudaMemcpy(input, input_mem, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(fk_kernel, kernel_mem, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyHostToDevice));
	//for (int k = 0; k < 1000; k++)
	{
		cufftErrchk(cufftExecC2C(plan, input, fk_input, CUFFT_FORWARD));
		//cudaProfilerStart();
		//Hadamard<<<grid,block>>>(nx,nz,ny,ny*nz,fk_input,fk_kernel,fk_output);
		//Hadamard << <(nx*ny*nz + 31) / 32, 32 >> >(nx*ny*nz, nx*ny*nz, fk_input, fk_kernel, fk_output);
		//cudaProfilerStop();
		// new CUDA 4.0 Driver API Kernel launch call
		//CUresult CUDAAPI cuLaunchKernel(
		//	CUfunction f,
		//	unsigned int gridDimX,
		//	unsigned int gridDimY,
		//	unsigned int gridDimZ,
		//	unsigned int blockDimX,
		//	unsigned int blockDimY,
		//	unsigned int blockDimZ,
		//	unsigned int sharedMemBytes,
		//	CUstream hStream,
		//	void **kernelParams,
		//	void **extra);

		_checkCudaErrors(cuLaunchKernel(fun, 
			//1,1,1,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));

		_checkCudaErrors(cuCtxSynchronize());

		cufftErrchk(cufftExecC2C(plan, fk_output, output, CUFFT_INVERSE));
		//cufftErrchk(cufftExecC2C(plan, fk_input, output, CUFFT_INVERSE));
	}
	gpuErrchk(cudaMemcpy(output_mem, output, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyDeviceToHost));
	verify("fk output:", nx, ny, nz, output_mem);
	verify_data("fk output:", nx, ny, nz, nx, output_mem,true);
	cufftDestroy(plan);
	T inv = 1.0f / (nx*ny);
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				output_data[k] = output_mem[k].x*inv;
			}
	cudaFree(input);
	cudaFree(output);
	cudaFree(fk_input);
	cudaFree(fk_kernel);
	cudaFree(fk_output);
	delete[] input_mem;
	delete[] kernel_mem;
	delete[] output_mem;
	
}

template<typename T>
void compute_rotation_convolution(CUfunction fun, int ind, T * input_data, T * output_data, int nx, int ny, int nz, T theta)
{
	// shear each z slice by constant vector
	// the data order is {Z,Y,X}, where Z is the slowest dimension, and X is the fastest
	//
	cufftHandle plan;
	cufftHandle plan1;
	cufftComplex * input_mem = new cufftComplex[nx*ny*nz];
	cufftComplex * kernel_mem = new cufftComplex[nx*ny];
	cufftComplex * output_mem = new cufftComplex[nx*ny*nz];
	/*
	float s = 2.0;
	for (float theta = -M_PI; theta <= 0; theta += M_PI / 32)
	{
		memset(rotation_kernel, 0, num_x*num_y);
		memset(rotation_kernel_display, 0, num_x*num_y);
		{
			float sigma_x_2 = 0.005f / (s*s);
			float sigma_y_2 = 0.000002f*10.0f;
			float a, b, c;
			float cos_theta = cos(theta);
			float cos_theta_2 = cos_theta*cos_theta;
			float sin_theta = sin(theta);
			float sin_theta_2 = sin_theta*sin_theta;
			float sin_2_theta = sin(2 * theta);
			float Z;
			float dx, dy;
			for (int y = 0, k = 0; y < num_y; y++)
			{
				for (int x = 0; x < num_x; x++, k++)
				{
					dx = (float)(x - (int)num_x / 2) / (float)num_x;
					dy = (float)(y - (int)num_y / 2) / (float)num_y;
					a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2);
					b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2);
					c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2);
					Z = exp(-(a*dx*dx - 2 * b*dx*dy + c*dy*dy));
					rotation_kernel[(((y + 2 * num_y - (int)num_y / 2) % num_y))*num_x + ((x + 2 * num_x - (int)num_x / 2) % num_x)] = Z;
					rotation_kernel_display[k] = Z;
				}
			}
		}
	}
	*/
	{
		float s = 2.0;
		float sigma_x_2 = 0.005f / (s*s);
		float sigma_y_2 = 0.000002f*10.0f;
		float a, b, c;
		float cos_theta = cos(theta);
		float cos_theta_2 = cos_theta*cos_theta;
		float sin_theta = sin(theta);
		float sin_theta_2 = sin_theta*sin_theta;
		float sin_2_theta = sin(2 * theta);
		float Z;
		float dx, dy;
		for (int y = 0, k = 0; y < ny; y++)
		for (int y = 0; y < ny; y++)
			for (int x = 0; x < nx; x++, k++)
			{
				dx = (float)(x - (int)nx / 2) / (float)nx;
				dy = (float)(y - (int)ny / 2) / (float)ny;
				a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2);
				b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2);
				c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2);
				Z = exp(-(a*dx*dx - 2 * b*dx*dy + c*dy*dy));
				kernel_mem[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].x = Z;
				kernel_mem[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].y = 0;
			}
	}
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				input_mem[k].x = input_data[k];
				input_mem[k].y = 0;
			}
	cufftComplex * input;
	cufftComplex * kernel;
	cufftComplex * output;
	cufftComplex * fk_input;
	cufftComplex * fk_kernel;
	cufftComplex * fk_output;
	cudaMalloc((void**)&input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&kernel, sizeof(cufftComplex)*nx*ny);
	cudaMalloc((void**)&output, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_kernel, sizeof(cufftComplex)*nx*ny);
	cudaMalloc((void**)&fk_output, sizeof(cufftComplex)*nx*ny*nz);
	int sizex = nx;
	int sizey = ny;
	int sizez = nz;
	int n[2] = { ny, nx };
	cufftErrchk(cufftPlanMany(&plan, 2, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, nz)
		);
	cufftErrchk(cufftPlanMany(&plan1, 2, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1)
		);
	int block_x = BLOCK_SIZE;
	int block_y = BLOCK_SIZE;
	int block_z = 1;
	dim3 block = dim3(block_x, block_y, block_z);
	//dim3 grid = dim3((nz + block_z - 1) / block_z, (ny + block_y - 1) / block_y, (nx + block_x - 1) / block_x);
	dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
	std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
	std::cout << block.x << " " << block.y << " " << block.z << std::endl;
	void *args[6] = { &sizex, &sizey, &sizez, &fk_input, &fk_kernel, &fk_output };
	gpuErrchk(cudaMemcpy(input, input_mem, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(kernel, kernel_mem, sizeof(cufftComplex)*nx*ny, cudaMemcpyHostToDevice));
	//for (int k = 0; k < 1000; k++)
	{
		cufftErrchk(cufftExecC2C(plan, input, fk_input, CUFFT_FORWARD));
		cufftErrchk(cufftExecC2C(plan1, kernel, fk_kernel, CUFFT_FORWARD));
		//cudaProfilerStart();
		//Hadamard<<<grid,block>>>(nx,nz,ny,ny*nz,fk_input,fk_kernel,fk_output);
		//Hadamard << <(nx*ny*nz + 31) / 32, 32 >> >(nx*ny*nz, nx*ny*nz, fk_input, fk_kernel, fk_output);
		//cudaProfilerStop();
		// new CUDA 4.0 Driver API Kernel launch call
		//CUresult CUDAAPI cuLaunchKernel(
		//	CUfunction f,
		//	unsigned int gridDimX,
		//	unsigned int gridDimY,
		//	unsigned int gridDimZ,
		//	unsigned int blockDimX,
		//	unsigned int blockDimY,
		//	unsigned int blockDimZ,
		//	unsigned int sharedMemBytes,
		//	CUstream hStream,
		//	void **kernelParams,
		//	void **extra);

		_checkCudaErrors(cuLaunchKernel(fun,
			//1,1,1,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));

		_checkCudaErrors(cuCtxSynchronize());

		cufftErrchk(cufftExecC2C(plan, fk_output, output, CUFFT_INVERSE));
		//cufftErrchk(cufftExecC2C(plan, fk_input, output, CUFFT_INVERSE));
	}
	gpuErrchk(cudaMemcpy(output_mem, output, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyDeviceToHost));
	verify("fk output:", nx, ny, nz, output_mem);
	verify_data("fk output:", nx, ny, nz, nx, output_mem, true);
	cufftDestroy(plan);
	cufftDestroy(plan1);
	T inv = 1.0f / (nx*ny);
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				output_data[k] = output_mem[k].x*inv;
			}
	cudaFree(input);
	cudaFree(output);
	cudaFree(fk_input);
	cudaFree(fk_kernel);
	cudaFree(fk_output);
	delete[] input_mem;
	delete[] kernel_mem;
	delete[] output_mem;
}

void test_shear(CUfunction fun,int ind)
{
	std::cout << "begin: test shear" << std::endl;
	int nx = 64;
	int ny = 64;
	int nz = 64;
	float * in = new float[nx*ny*nz];
	float *out = new float[nx*ny*nz];
	for (std::size_t x = 0, k = 0; x < nx; x++)
		for (std::size_t y = 0; y < ny; y++)
			for (std::size_t z = 0; z < nz; z++, k++)
			{
				out[k] = 0;
				in[k] = 0;
				if (	pow((int)x - (int)nx / 2, 2)
					+	pow((int)y - (int)ny / 2, 2)
					+	pow((int)z - (int)nz / 2, 2)
					>	pow(nx / 2, 2)
					)
				{
					in[k] += 5.0f;
				}
			}
	float shift_x = 0.0f;
	float shift_y = 0.0f;
	//d_local_update->update("in:", nx, ny, nz, in);
	compute_shear(fun, ind, in, out, nx, ny, nz, shift_x, shift_y);
	//d_local_update->update("out:", nx, ny, nz, out);
	std::cout << "done: test shear" << std::endl;
}

void test_convolution_kernel(CUfunction fun, int ind)
{
	std::cout << "begin: test convolution kernel" << std::endl;
	int nx = 64;
	int ny = 64;
	int nz = 64;
	float * in = new float[nx*ny*nz];
	float *out = new float[nx*ny*nz];
	for (std::size_t x = 0, k = 0; x < nx; x++)
		for (std::size_t y = 0; y < ny; y++)
			for (std::size_t z = 0; z < nz; z++, k++)
			{
				out[k] = 0;
				in[k] = 0;
				if (pow((int)x - (int)nx / 2, 2)
					+ pow((int)y - (int)ny / 2, 2)
					+ pow((int)z - (int)nz / 2, 2)
			>	pow(nx / 2, 2)
			)
				{
					in[k] += 5.0f;
				}
			}
	float theta = M_PI/3;
	//d_local_update->update("in:", nx, ny, nz, in);
	compute_rotation_convolution(fun, ind, in, out, nx, ny, nz, theta);
	//d_local_update->update("out:", nx, ny, nz, out);
	std::cout << "done: test shear" << std::endl;
}

void test_gpu()
{
	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	CUfunction cuFunction = 0;
	std::size_t totalGlobalMem;
	int major = 0, minor = 0;
	char deviceName[100];

	_checkCudaErrors(cuInit(0));

	int devices = 0;
	_checkCudaErrors(cuDeviceGetCount(&devices));
	printf("num devices:%d\n",devices);
	
	/*
	for (int ind = 0; ind < devices;ind++)
	{
		printf("device %d:\n",ind);
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		// get compute capabilities and the devicename
		_checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
		_checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
		printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
		_checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
		printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
		printf("  64-bit Memory Address:             %s\n", (totalGlobalMem > (unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");
	
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		_checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
		
		_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_fft.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&cuFunction, cuModule, "Hadamard"));
		
		test_shear(cuFunction,ind);

		_checkCudaErrors(cuCtxDestroy(cuContext));
		
	}
	*/
	
	for (int ind = 0; ind < devices; ind++)
	{
		printf("device %d:\n", ind);
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		// get compute capabilities and the devicename
		_checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
		_checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
		printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
		_checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
		printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
		printf("  64-bit Memory Address:             %s\n", (totalGlobalMem >(unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");

		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		_checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

		_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_fft.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&cuFunction, cuModule, "Hadamard_slice_kernel"));

		test_convolution_kernel(cuFunction, ind);

		_checkCudaErrors(cuCtxDestroy(cuContext));

	}

}

void quick_zsmooth_sanity_check()
{
	float dat[64];
	for (int i = 0; i < 64; i++)
	{
		dat[i] = 0;
	}
	int ind = 32;
	dat[ind] = 1;
	float a, b;
	b = 4;
	a = b - 1;
	float alpha = a/b;
	for (int i = 1; i < 64; i++)
	{
		dat[i] += dat[i - 1] * alpha;
	}
	float beta = a/b;
	for (int i = 64 - 2; i >= 0; i--)
	{
		dat[i] += dat[i + 1] * beta;
	}
	float factor = b*b / (a + b);
	int w = 5;
	for (int k = -w; k <= w; k++)
	{
		std::cout << ind+k << ") " << dat[ind + k]/factor << std::endl;
	}
}

void test_gpu_afi()
{
	
	int nx = 256;
	int ny = 256;
	int nz = 256;
	float * in = new float[nx*ny*nz];
	float *out = new float[nx*ny*nz];
	for (std::size_t x = 0, k = 0; x < nx; x++)
	{
		for (std::size_t y = 0; y < ny; y++)
		{
			for (std::size_t z = 0; z < nz; z++, k++)
			{
				out[k] = 0;
				in[k] = 0;
				if (pow((int)x - (int)nx / 2, 2)
					+ pow((int)y - (int)ny / 2, 2)
					+ pow((int)z - (int)nz / 2, 2)
					> pow(nx / 2, 2)
					)
				{
					in[k] = k%10;
				}
			}
		}
	}

	ComputeManager * c_manager = new ComputeManager();

	int num_cpu_copies = 0;
	int num_gpu_copies = 1;
	std::vector<ComputeDevice*> device = c_manager->create_compute_devices(num_cpu_copies,num_gpu_copies);

	std::string input_name = "input";            // note to self: could potentially delete this after getting transpose, if memory is an issue
	std::string transpose_name = "transpose";    // can also get rid of this guy, once the semblance volumes have been calculated
	std::string numerator_name = "numerator";
	std::string denominator_name = "denominator";
	std::string numerator_name_freq = "numerator_freq";
	std::string denominator_name_freq = "denominator_freq";
	std::string rotation_kernel_name = "rotation_kernel";
	std::string rotated_numerator_name = "rotated_numerator";
	std::string rotated_denominator_name = "rotated_denominator";
	std::string rotated_numerator_name_freq = "rotated_numerator_freq";
	std::string rotated_denominator_name_freq = "rotated_denominator_freq";
	std::string rotated_numerator_name_time = "rotated_numerator_time";
	std::string rotated_denominator_name_time = "rotated_denominator_time";
	std::string sheared_numerator_name_freq = "sheared_numerator_freq";
	std::string sheared_denominator_name_freq = "sheared_denominator_freq"; // this also serves as the semblance output, but it is repopulated with fresh data with each shear iteration, so it's ok
	std::string sheared_numerator_name_time = "sheared_numerator_time";
	std::string sheared_denominator_name_time = "sheared_denominator_time";
	std::string fault_likelihood_name = "fault_likelihood";
	std::string optimal_fault_likelihood_name = "optimal_fault_likelihood";
	std::string output_fault_likelihood_name = "output_fault_likelihood";

	for (int ind = 0; ind < device.size(); ind++)
	{
		std::cout << "device-" << ind << std::endl;
		ComputeDevice * d = device[ind];

		// input = {X,Y,Z}
		//tile_display_update->update("input", num_x, num_y, num_z, input);
		d->create(input_name, nx, ny, nz, in, false, true); // allocate input array (time domain) {X,Y,Z}

		// transpose: {X,Y,Z} -> {Z,Y,X}
		//float * data_zyx = new float[num_x*num_y*num_z]; // {Z,Y,X}
		d->create(transpose_name, nz, ny, nx); // create transpose array (time domain) {Z,Y,X}
		d->compute_transpose(nx, ny, nz, input_name, transpose_name);

		//transpose_constY(num_x, num_y, num_z, input, data_zyx);
		//tile_display_update->update1("zyx", num_z, num_y, num_x, data_zyx);
		// semblance on: {Z,Y,X}
		//float * num = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//float * den = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//d->create(numerator_name, nz, ny, nx); // create numerator array (time domain) {X,Y,Z}
		//d->create(denominator_name, nz, ny, nx); // create denominator array (time domain) {X,Y,Z}

		//int win = 1;
		//semblance_structure_oriented(win, num_z, num_y, num_x, data_zyx, num, den);
		//int win = 1;
		//d->compute_semblance(win, nz, ny, nx, transpose_name, numerator_name, denominator_name);
		//if (tile_display_update->comprehensive)tile_display_update->update("numerator", num_z, num_y, num_x, num);
		//if (tile_display_update->comprehensive)tile_display_update->update("denominator", num_z, num_y, num_x, den);

		d->init_fft(nz, ny, nx);

		d->initialize_semblance(nz,ny,nx,transpose_name,numerator_name,denominator_name,numerator_name_freq,denominator_name_freq);

		//std::size_t num_sheared_y = num_y + 2 * 64;
		//std::size_t num_sheared_x = num_x + 2 * 64;
		//std::size_t num_sheared_y = ny + 2 * 64;
		//std::size_t num_sheared_x = nx + 2 * 64;

		//float * rotation_kernel = new float[num_x*num_y*1]; // {1,Y,X}
		//float * rotation_kernel_display = new float[num_x*num_y*1]; // {1,Y,X}
		//float * num_rotation = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//float * den_rotation = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//float * num_shear = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		//float * den_shear = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		//float * shear_semblance = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		//float * semblance = new float[num_x * num_y * num_z]; // {Z,Y,X}
		//float * semblance_optimum = new float[num_x * num_y * num_z]; // {Z,Y,X}
		//float * fault_likelihood_optimum = new float[num_x * num_y * num_z]; // {Z,Y,X}

		//for (int k = 0, size = num_x*num_y*num_z; k < size; k++)
		//{
		//	semblance_optimum[k] = 1.0f;
		//	fault_likelihood_optimum[k] = 0.0f;
		//}

		d->create(optimal_fault_likelihood_name, nz, ny, nx);
		d->create(output_fault_likelihood_name, nx, ny, nz);

		d->create(rotated_numerator_name_freq, nz, ny, nx, NULL, true);
		d->create(rotated_denominator_name_freq, nz, ny, nx, NULL, true);

		d->create(rotated_numerator_name_time, nz, ny, nx, NULL, true);
		d->create(rotated_denominator_name_time, nz, ny, nx, NULL, true);

		d->create(sheared_numerator_name_freq, nz, ny, nx, NULL, true);
		d->create(sheared_denominator_name_freq, nz, ny, nx, NULL, true);

		d->create(sheared_numerator_name_time, nz, ny, nx, NULL, true);
		d->create(sheared_denominator_name_time, nz, ny, nx, NULL, true);

		d->init_shear(nz, ny, nx, fault_likelihood_name, sheared_numerator_name_freq, sheared_denominator_name_freq);

		//float s = 2.0;
		int theta_ind = 0;
		for (float theta = -M_PI; theta <= 0; theta += M_PI / 64, theta_ind++)
		{
			std::stringstream ss_rotation_kernel_name_freq;
			ss_rotation_kernel_name_freq << rotation_kernel_name << "-" << theta_ind << "-freq";
			if (d->create(ss_rotation_kernel_name_freq.str(), 1, ny, nx, NULL, true)) // create frequency domain array for kernel
			{
				std::stringstream ss_rotation_kernel_name_time;
				ss_rotation_kernel_name_time << rotation_kernel_name << "-" << theta_ind << "-time";
				d->initialize_rotation_kernel(nx, ny, theta, ss_rotation_kernel_name_time.str(), ss_rotation_kernel_name_freq.str());
			}
		//	memset(rotation_kernel, 0, num_x*num_y);
		//	memset(rotation_kernel_display, 0, num_x*num_y);
		//	{
		//		float sigma_x_2 = 0.005f / (s*s);
		//		float sigma_y_2 = 0.000002f*10.0f;
		//		float a, b, c;
		//		float cos_theta = cos(theta);
		//		float cos_theta_2 = cos_theta*cos_theta;
		//		float sin_theta = sin(theta);
		//		float sin_theta_2 = sin_theta*sin_theta;
		//		float sin_2_theta = sin(2 * theta);
		//		float Z;
		//		float dx, dy;
		//		for (int y = 0, k = 0; y < num_y; y++)
		//		{
		//			for (int x = 0; x < num_x; x++, k++)
		//			{
		//				dx = (float)(x - (int)num_x / 2) / (float)num_x;
		//				dy = (float)(y - (int)num_y / 2) / (float)num_y;
		//				a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2);
		//				b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2);
		//				c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2);
		//				Z = exp(-(a*dx*dx - 2 * b*dx*dy + c*dy*dy));
		//				rotation_kernel[(((y + 2 * num_y - (int)num_y / 2) % num_y))*num_x + ((x + 2 * num_x - (int)num_x / 2) % num_x)] = Z;
		//				rotation_kernel_display[k] = Z;
		//			}
		//		}
		//	}
		//	tile_display_update->update2("rotation kernel", 1, num_y, num_x, rotation_kernel_display);
		// convolve: rotation_kernel * num
			
			d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), numerator_name_freq, rotated_numerator_name_freq);
			//d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), numerator_name_freq, rotated_numerator_name_freq);
			//d->inv_fft(nz, ny, nx, rotated_numerator_name_freq, rotated_numerator_name_time);

		//	compute_convolution_2d_slices_fast_b_c2c(num_z, num_y, num_x, num, rotation_kernel, num_rotation);
		//	if (tile_display_update->comprehensive)tile_display_update->update("numerator rotated", num_z, num_y, num_x, num_rotation);
			
		// convolve: rotation_kernel * den
			
			d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), denominator_name_freq, rotated_denominator_name_freq);
			//d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), denominator_name_freq, rotated_denominator_name_freq);
			//d->inv_fft(nz, ny, nx, rotated_denominator_name_freq, rotated_denominator_name_time);

		//	compute_convolution_2d_slices_fast_b_c2c(num_z, num_y, num_x, den, rotation_kernel, den_rotation);
		//	if (tile_display_update->comprehensive)tile_display_update->update("denominator rotated", num_z, num_y, num_x, den_rotation);

			float shear_extend = 0.1f;
			for (float shear = -shear_extend; shear <= shear_extend; shear += shear_extend/32.0f)
		//	float shear = 0.0f;
			{
				//std::cout << "theta:" << theta << "   shear:" << shear << std::endl;

				float shear_y = shear*cos(theta);
				float shear_x = -shear*sin(theta);

				d->compute_fault_likelihood(
					nz
					, ny
					, nx
					, shear_y
					, shear_x
					, rotated_numerator_name_freq
					, rotated_denominator_name_freq
					, sheared_numerator_name_freq
					, sheared_denominator_name_freq/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
					, fault_likelihood_name/*time domain {Z,Y,X}*/
					, optimal_fault_likelihood_name
					//, optimal_theta_name
					//, optimal_phi_name
					);

				//d->compute_fault_likelihood_time(
				//	nz
				//	, ny
				//	, nx
				//	, shear_y
				//	, shear_x
				//	, rotated_numerator_name_time
				//	, rotated_denominator_name_time
				//	, sheared_numerator_name_time
				//	, sheared_denominator_name_time/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
				//	, fault_likelihood_name/*time domain {Z,Y,X}*/
				//	, optimal_fault_likelihood_name
				//	//, optimal_theta_name
				//	//, optimal_phi_name
				//	);


		//		shear_2d(FORWARD, LINEAR/*FFT*/, 1, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, num_rotation, num_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("numerator sheared", num_z, num_sheared_y, num_sheared_x, num_shear);
		//		shear_2d(FORWARD, LINEAR/*FFT*/, 1, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, den_rotation, den_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("denominator sheared", num_z, num_sheared_y, num_sheared_x, den_shear);

		//		float scalar = 0.9f;
		//		zsmooth(scalar, num_z, num_sheared_y, num_sheared_x, num_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("numerator zsmooth", num_z, num_sheared_y, num_sheared_x, num_shear);
		//		zsmooth(scalar, num_z, num_sheared_y, num_sheared_x, den_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("denomiantor zsmooth", num_z, num_sheared_y, num_sheared_x, num_shear);

		//		semblance_div(num_z, num_sheared_y, num_sheared_x, shear_semblance, num_shear, den_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("semblance div", num_z, num_sheared_y, num_sheared_x, shear_semblance);

		//		shear_2d(BACKWARD, LINEAR/*FFT*/, 0, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, semblance, shear_semblance);
		//		if (tile_display_update->comprehensive)tile_display_update->update("semblance", num_z, num_y, num_x, semblance);

		//for (int k = 0, size = num_x*num_y*num_z; k < size; k++)
		//{
		//	semblance_optimum[k] = (semblance[k] < semblance_optimum[k]) ? semblance[k] : semblance_optimum[k];
		//}

		//		float val;
		//		float fh;
		//		for (int z = 0, k = 0; z < num_z; z++)
		//			for (int y = 0; y < num_y; y++)
		//				for (int x = 0; x < num_x; x++, k++)
		//if (x >= pad && x + pad < num_x)
		//if (y >= pad && y + pad < num_y)
		//if (z >= pad && z + pad < num_z)
		//				{
		//					val = semblance[k];
		//					val *= val;
		//					val *= val;
		//					val *= val;
		//					fh = 1.0f - val;
		//					fault_likelihood_optimum[k] = (fh>fault_likelihood_optimum[k]) ? fh : fault_likelihood_optimum[k];
		//				}

		//		tile_display_update->update("fault_likelihood_optimum", num_z, num_y, num_x, fault_likelihood_optimum);

			}

		}

		d->destroy_fft();

		//std::cout << "p1" << std::endl;
		d->compute_transpose(nz, ny, nx, optimal_fault_likelihood_name, output_fault_likelihood_name);
		//std::cout << "p2" << std::endl;
		//transpose_constY(num_z, num_y, num_x, fault_likelihood_optimum, input);

		//tile_display_update->clear();
		//tile_display_update->clear1();
		//tile_display_update->clear2();

		//delete[] data_zyx;
		//delete[] semblance;
		//delete[] semblance_optimum;
		//delete[] fault_likelihood_optimum;
		//delete[] shear_semblance;
		//delete[] num_shear;
		//delete[] den_shear;
		//delete[] num_rotation;
		//delete[] den_rotation;
		//delete[] rotation_kernel;
		//delete[] rotation_kernel_display;
		//delete[] num;
		//delete[] den;
	}
	//std::cout << "p3" << std::endl;
	c_manager->destroy();
	//std::cout << "p4" << std::endl;
	delete c_manager;
	//std::cout << "p5" << std::endl;
}

#endif

