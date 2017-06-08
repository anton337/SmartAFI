#ifndef gpu_array_h
#define gpu_array_h

class GPUArray : public DataArray
{
	float * data;
public:
	GPUArray(std::string name
			,std::size_t size)
		: DataArray(name, size,GPU)
	{

	}
	void put(void * _data)
	{
		std::cout << "GPU put not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}
	void * get()
	{
		return (void*)data;
	}
  void get_output(void * ptr)
  {
	  gpuErrchk(cudaMemcpy(ptr, data, sizeof(float)*get_size(), cudaMemcpyDeviceToHost));
  }
	void allocate(float * arr = NULL, bool keep_data = false)
	{
		//std::cout << "cudaMalloc:" << get_name() << std::endl;
		gpuErrchk(cudaMalloc((void**)&data, sizeof(float)*get_size()));
		if (arr)
		{
			gpuErrchk(cudaMemcpy(data, arr, sizeof(float)*get_size(), cudaMemcpyHostToDevice));
			if (!keep_data)
			{
				delete[] arr;
			}
		}
		else
		{
			gpuErrchk(cudaMemset(data,0,sizeof(float)*get_size()));
		}
	}
	void set(float * arr)
	{
		if (arr)
		{
      std::cout << data << std::endl;
      std::cout << arr << std::endl;
      std::cout << get_size() << std::endl;
			gpuErrchk(cudaMemcpy(data, arr, sizeof(float)*get_size(), cudaMemcpyHostToDevice));
    }
	}
	void fill(float arr)
	{
		//gpuErrchk(cudaMemset(data, 0, sizeof(float)*get_size()));
	}
	void destroy()
	{
		cudaFree(data);
	}
};

#endif

