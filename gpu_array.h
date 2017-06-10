#ifndef gpu_array_h
#define gpu_array_h

class GPUArray : public DataArray
{
	float * data;
public:
	GPUArray(Token token
			,std::size_t nx,std::size_t ny,std::size_t nz)
		: DataArray(token,nx,ny,nz,GPU)
	{
		MESSAGE_ASSERT(token.type==GPU,"type mismatch");
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
	std::string get_validation()
	{
		{
			float min_val = 1000000;
			float max_val = -1000000;
			float val;
			std::size_t size = get_size();
			float * tmp = new float[size];
			cudaMemcpy(tmp, data, sizeof(float)*size, cudaMemcpyDeviceToHost);
			for (std::size_t k = 0; k < size; k++)
			{
				val = tmp[k];
				max_val = (val > max_val) ? val : max_val;
				min_val = (val < min_val) ? val : min_val;
			}
			delete[] tmp;
			std::stringstream ss;
			ss << "\t" << get_nx() << "x" << get_ny() << "x" << get_nz() << " === " << min_val << " --- " << max_val << std::endl;
			return ss.str();
		}
	}
};

#endif

