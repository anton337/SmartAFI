#ifndef cpu_array_h
#define cpu_array_h

class CPUArray : public DataArray
{
	float * data;
public:
	CPUArray(std::string name
			,std::size_t size)
		: DataArray(name,size,CPU)
	{

	}
	void put(void * _data)
	{
		data = (float*)_data;
	}
	void * get()
	{
		return (void*)data;
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
};

#endif

