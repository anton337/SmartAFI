#ifndef data_array_h
#define data_array_h

enum TYPE {FREQ_CPU,CPU,FREQ_GPU,GPU};

class DataArray
{
	std::string name;
	std::size_t size;
	TYPE type;
public:
	DataArray(std::string _name,std::size_t _size,TYPE _type)
		: name(_name)
		, size(_size)
		, type(_type)
	{

	}
	std::string get_name()
	{
		return name;
	}
	std::size_t get_size()
	{
		return size;
	}
	TYPE get_type()
	{
		return type;
	}
	virtual void allocate(float * arr = NULL, bool keep_data = false) = 0;
	virtual void put(void *) = 0;
	virtual void set(float *) = 0;
	virtual void fill(float) = 0;
	virtual void * get() = 0;
	virtual void destroy() = 0;
};

#endif

