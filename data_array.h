#ifndef data_array_h
#define data_array_h

#include "token.h"

class DataArray
{
	const Token token;
	const std::size_t nx,ny,nz;
	const TYPE type;
public:
	DataArray(Token _token,std::size_t _nx,std::size_t _ny,std::size_t _nz,TYPE _type)
		: token(_token)
		, nx(_nx)
		, ny(_ny)
		, nz(_nz)
		, type(_type)
	{

	}
	std::string get_name()
	{
		return token.name;
	}
	std::size_t get_size()
	{
		return nx*ny*nz;
	}
	std::size_t get_nx()
	{
		return nx;
	}
	std::size_t get_ny()
	{
		return ny;
	}
	std::size_t get_nz()
	{
		return nz;
	}
	TYPE get_type()
	{
		return type;
	}
	std::string get_pretty_type()
	{
		switch (type)
		{
		case GPU:return "GPU";
		case CPU:return "CPU";
		case FREQ_GPU:return "FREQ_GPU";
		case FREQ_CPU:return "FREQ_CPU";
		}
	}
	virtual std::string get_validation() = 0;
	virtual void allocate(float * arr = NULL, bool keep_data = false) = 0;
	virtual void put(void *) = 0;
	virtual void set(float *) = 0;
	virtual void fill(float) = 0;
	virtual void * get() = 0;
    virtual void get_output(void *) = 0;
	virtual void destroy() = 0;
};

#endif

