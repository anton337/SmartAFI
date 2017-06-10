#ifndef cpu_array_h
#define cpu_array_h

class CPUArray : public DataArray
{
	float * data;
public:
	CPUArray(Token token
			,std::size_t nx,std::size_t ny,std::size_t nz)
		: DataArray(token,nx,ny,nz,CPU)
	{
		MESSAGE_ASSERT(token.type == CPU, "type mismatch");
	}
	void put(void * _data)
	{
		data = (float*)_data;
	}
	void * get()
	{
		return (void*)data;
	}
  void get_output(void * ptr)
  {
    ptr = data;
  }
	void allocate(float * arr = NULL,bool keep_data = false)
	{
		if (arr)
		{
			data = arr;
		}
		else
		{
			data = new float[get_size()];
		}
	}
	void set(float * arr)
	{
		for (int i = 0, size = get_size(); i < size; i++)
		{
			data[i] = arr[i];
		}
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void fill(float arr)
	{
		std::cout << "CPU fill not implemented yet." << std::endl;
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
			std::size_t size = get_size();
			for (std::size_t k = 0; k < size; k++)
			{
				max_val = (data[k] > max_val) ? data[k] : max_val;
				min_val = (data[k] < min_val) ? data[k] : min_val;
			}
			std::stringstream ss;
			ss << "\t" << get_nx() << "x" << get_ny() << "x" << get_nz() << " === " << min_val << " --- " << max_val << std::endl;
			return ss.str();
		}
	}
};

#endif

