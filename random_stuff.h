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

#endif

