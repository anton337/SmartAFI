#ifndef freq_gpu_array_h
#define freq_gpu_array_h

class freqGPUArray : public DataArray
{
	cufftComplex * data;
public:
	freqGPUArray( Token token
				, std::size_t nx,std::size_t ny,std::size_t nz)
		: DataArray(token,nx,ny,nz,FREQ_GPU)
	{
		MESSAGE_ASSERT(token.type == FREQ_GPU, "type mismatch");
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
	std::string get_validation()
	{
		{
			float min_val = 1000000;
			float max_val = -1000000;
			float val;
			std::size_t size = get_size();
			cufftComplex * tmp = new cufftComplex[size];
			cudaMemcpy(tmp, data, sizeof(cufftComplex)*size, cudaMemcpyDeviceToHost);
			for (std::size_t k = 0; k < size; k++)
			{
				val = sqrtf(tmp[k].x*tmp[k].x + tmp[k].y*tmp[k].y);
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

