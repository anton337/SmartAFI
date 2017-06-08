#ifndef freq_gpu_array_h
#define freq_gpu_array_h

class freqGPUArray : public DataArray
{
	cufftComplex * data;
public:
	freqGPUArray(std::string name
				, std::size_t size)
		: DataArray(name, size,FREQ_GPU)
	{

	}
	void put(void * _data)
	{
		std::cout << "freq GPU put not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}
	void * get()
	{
		return (void*)data;
	}
  void get_output(void * ptr)
  {
		std::cout << "freq GPU get_output not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
  }
	void allocate(float * arr = NULL, bool keep_data = false)
	{
		//std::cout << "cudaMalloc:" << get_name() << std::endl;
		gpuErrchk(cudaMalloc((void**)&data, sizeof(cufftComplex)*get_size()));
		if (arr)
		{
			cufftComplex * tmp = new cufftComplex[get_size()];
			for (int i = 0; i < get_size(); i++)
			{
				tmp[i].x = arr[i];
				tmp[i].y = 0;
			}
			if (!keep_data)
			{
				delete[] arr; // don't forget
			}
			gpuErrchk(cudaMemcpy(data, tmp, sizeof(cufftComplex)*get_size(), cudaMemcpyHostToDevice));
			delete[] tmp;
		}
	}
	void set(float * arr)
	{
		std::cout << "freq GPU set not implemented yet." << std::endl;
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void fill(float dat)
	{
		// can't figure it out, just leave it empty
		// cufftComplex data is randomly initialized
	}
	void destroy()
	{
		gpuErrchk(cudaFree(data));
	}
};

#endif

