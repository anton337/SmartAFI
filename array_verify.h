#ifndef array_verify_h
#define array_verify_h

void verify(std::string message, std::size_t nx, std::size_t ny, std::size_t nz, float * arr,bool pause=false)
{
	float min_val = 1000000;
	float max_val = -1000000;
	std::size_t size = nx*ny*nz;
	for (std::size_t k = 0; k < size; k++)
	{
		max_val = (arr[k] > max_val) ? arr[k] : max_val;
		min_val = (arr[k] < min_val) ? arr[k] : min_val;
	}
	std::cout << message << ": " << nx << "x" << ny << "x" << nz << " === " << min_val << " --- " << max_val << std::endl;
	if (pause)
	{
		std::cout << "press any key to continue..." << std::endl;
		char ch;
		std::cin >> ch;
	}
}

void verify(std::string message, std::size_t nx, std::size_t ny, std::size_t nz, cufftComplex * arr, bool pause = false)
{
	float min_val = 1000000;
	float max_val = -1000000;
	float val;
	std::size_t size = nx*ny*nz;
	for (std::size_t k = 0; k < size; k++)
	{
		val = sqrt(arr[k].x*arr[k].x + arr[k].y*arr[k].y);
		max_val = (val > max_val) ? val : max_val;
		min_val = (val < min_val) ? val : min_val;
	}
	std::cout << message << ": " << nx << "x" << ny << "x" << nz << " === " << min_val << " --- " << max_val << std::endl;
	if (pause)
	{
		std::cout << "press any key to continue..." << std::endl;
		char ch;
		std::cin >> ch;
	}
}

void verifyGPU(std::string message, std::size_t nx, std::size_t ny, std::size_t nz, float * arr, bool pause = false)
{
	
	float min_val = 1000000;
	float max_val = -1000000;
	float val;
	std::size_t size = nx*ny*nz;
	float * tmp = new float[size];
	cudaMemcpy(tmp, arr, sizeof(float)*size, cudaMemcpyDeviceToHost);
	for (std::size_t k = 0; k < size; k++)
	{
		val = tmp[k];
		max_val = (val > max_val) ? val : max_val;
		min_val = (val < min_val) ? val : min_val;
	}
	delete[] tmp;
	
	std::cout << message << ": " << nx << "x" << ny << "x" << nz << " === " << min_val << " --- " << max_val << std::endl;
	if (pause)
	{
		std::cout << "press any key to continue..." << std::endl;
		char ch;
		std::cin >> ch;
	}
}

void verifyGPU(std::string message, std::size_t nx, std::size_t ny, std::size_t nz, cufftComplex * arr, bool pause = false)
{
	float min_val = 1000000;
	float max_val = -1000000;
	float val;
	std::size_t size = nx*ny*nz;
	cufftComplex * tmp = new cufftComplex[size];
	cudaMemcpy(tmp, arr, sizeof(cufftComplex)*size, cudaMemcpyDeviceToHost);
	for (std::size_t k = 0; k < size; k++)
	{
		val = sqrtf(tmp[k].x*tmp[k].x + tmp[k].y*tmp[k].y);
		max_val = (val > max_val) ? val : max_val;
		min_val = (val < min_val) ? val : min_val;
	}
	delete[] tmp;
	std::cout << message << ": " << nx << "x" << ny << "x" << nz << " === " << min_val << " --- " << max_val << std::endl;
	if (pause)
	{
		std::cout << "press any key to continue..." << std::endl;
		char ch;
		std::cin >> ch;
	}
}

void verify_data(std::string message, std::size_t nx, std::size_t ny, std::size_t nz, std::size_t step, cufftComplex * arr, bool pause = false)
{
	std::size_t size = nx*ny*nz;
	std::cout << message << ": " << nx << "x" << ny << "x" << nz << std::endl;
	for (std::size_t k = 0; k < size; k+=step)
	{
		std::cout << k << " === " << k + step << std::endl;
		for (std::size_t i = 0; i < step && k+i<size; i++)
		{
			std::cout << arr[k+i].x << "+" << arr[k+i].y << "i" << "\t";
		}
		std::cout << std::endl;
		if (pause)
			{
				std::cout << "press any key to continue, 'c' to terminate" << std::endl;
				char ch;
				std::cin >> ch;
				std::cout << "\"" << ch << "\"" << std::endl;
				if (ch == 'c')
				{
					std::cout << "terminating" << std::endl;
					return;
				}
				std::cout << "continuing" << std::endl;
			}
	}
	
}

#endif
