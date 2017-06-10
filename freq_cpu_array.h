#ifndef freq_cpu_array_h
#define freq_cpu_array_h

class freqCPUArray : public DataArray
{
	fftwf_complex * data;
public:
	freqCPUArray(Token token
			,std::size_t nx,std::size_t ny,std::size_t nz)
		: DataArray(token,nx,ny,nz,FREQ_CPU)
	{
		MESSAGE_ASSERT(token.type == FREQ_CPU, "type mismatch");
	}
	void put(void * _data)
	{
		data = (fftwf_complex*)_data;
	}
	void * get()
	{
		return (void*)data;
	}	
  void get_output(void * ptr)
  {
		std::cout << "freq CPU get_output not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
  }
	void allocate(float * arr = NULL, bool keep_data = false)
	{
		std::cout << "freq CPU allocate not implemented yet." << std::endl;
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void set(float * arr)
	{
		std::cout << "freq CPU allocate not implemented yet." << std::endl;
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void fill(float arr)
	{
		std::cout << "freq CPU fill not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}
	void destroy()
	{
		delete[] data;
	}
	std::string get_validation()
	{
		{
			float min_val = 1000000;
			float max_val = -1000000;
			float val;
			std::size_t size = get_size();
			for (std::size_t k = 0; k < size; k++)
			{
				val = sqrt(data[k][0]*data[k][0] + data[k][1]*data[k][1]);
				max_val = (val > max_val) ? val : max_val;
				min_val = (val < min_val) ? val : min_val;
			}
			std::stringstream ss;
			ss << "\t" << get_nx() << "x" << get_ny() << "x" << get_nz() << " === " << min_val << " --- " << max_val << std::endl;
			return ss.str();
		}
	}
};

#endif

