#ifndef freq_cpu_array_h
#define freq_cpu_array_h

class freqCPUArray : public DataArray
{
	fftwf_complex * data;
public:
	freqCPUArray(std::string name
			,std::size_t size)
		: DataArray(name,size,FREQ_CPU)
	{

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
};

#endif

