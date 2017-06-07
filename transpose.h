#ifndef transpose_h
#define transpose_h

#include <boost/thread.hpp>

// traditional order is {X,Y,Z} where Z is fastest dimension, X is the slowest

void transpose_constX_worker(std::size_t start_x, std::size_t end_x, std::size_t num_y, std::size_t num_z, float * in, float * out)
{
	for (std::size_t x = start_x; x < end_x; x++)
	{
		//  input is ordered by {X,Y,Z}
		// output is ordered by {X,Z,Y}
		// out[(x*num_z*num_y) + z*num_y + y] := in[(x*num_y*num_z) + y*num_z + z]
		float *  in_x = &in[x*num_y*num_z];
		float * out_x = &out[x*num_y*num_z];
		for (std::size_t y = 0, k = 0; y < num_y; y++)
		{
			for (std::size_t z = 0; z < num_z; z++, k++)
			{
				out_x[z*num_y + y] = in_x[k];
			}
		}
	}
}

// for each X slice, we swap Y <---> Z
void transpose_constX(std::size_t num_x, std::size_t num_y, std::size_t num_z, float * in, float * out)
{
	std::cout << "{X,Y,Z} ---> {X,Z,Y}" << std::endl;
	int nthreads = boost::thread::hardware_concurrency();
	std::size_t dx = num_x / nthreads;
	std::vector<boost::thread*> threads;
	// can parallelize x
	for (std::size_t x = 0; x < num_x; x += dx)
	{
		threads.push_back(new boost::thread(transpose_constX_worker
			, x
			, (x + dx < num_x) ? x + dx : num_x
			, num_y
			, num_z
			, in
			, out
			)
			);
	}
	for (std::size_t i = 0; i < threads.size(); i++)
	{
		threads[i]->join();
		delete threads[i];
	}
	threads.clear();
}

void transpose_constY_worker(std::size_t num_x, std::size_t start_y, std::size_t end_y, std::size_t num_y, std::size_t num_z, float * in, float * out)
{
	for (std::size_t y = start_y; y < end_y; y++)
	{
		//  input is ordered by {X,Y,Z}
		// output is ordered by {Z,Y,X}
		// out[z*num_y*num_x + y*num_x + x] := in[x*num_y*num_z + y*num_z + z]
		for (std::size_t x = 0; x < num_x; x++)
		{
			float *  in_xy = &in[x*num_y*num_z + y*num_z];
			for (std::size_t z = 0; z < num_z; z++)
			{
				out[(z*num_y + y)*num_x + x] = in_xy[z];
			}
		}
	}
}

void transpose_constY(std::size_t num_x, std::size_t num_y, std::size_t num_z, float * in, float * out)
{
	std::cout << "{X,Y,Z} ---> {Z,Y,X}" << std::endl;
	int nthreads = boost::thread::hardware_concurrency();
	std::size_t dy = num_y / nthreads;
	std::vector<boost::thread*> threads;
	// can parallelize y
	for (std::size_t y = 0; y < num_y; y += dy)
	{
		threads.push_back(new boost::thread(transpose_constY_worker
			, num_x
			, y
			, (y + dy < num_y) ? y + dy : num_y
			, num_y
			, num_z
			, in
			, out
			)
			);
	}
	for (std::size_t i = 0; i < threads.size(); i++)
	{
		threads[i]->join();
		delete threads[i];
	}
	threads.clear();
}

void transpose_constZ_worker(std::size_t num_x, std::size_t num_y, std::size_t start_z, std::size_t end_z, std::size_t num_z, float * in, float * out)
{
	for (std::size_t z = start_z; z < end_z; z++)
	{
		//  input is ordered by {X,Y,Z}
		// output is ordered by {Y,X,Z}
		// out[y*num_x*num_z + x*num_z + z] := in[x*num_y*num_z + y*num_z + z]
		for (std::size_t x = 0; x < num_x; x++)
		{
			for (std::size_t y = 0; y < num_y; y++)
			{
				out[(y*num_x + x)*num_z + z] = in[(x*num_y + y)*num_z + z];
			}
		}
	}
}


// AVOID THIS FUNCTION AT ALL COSTS!!!! IT IS SUPER DUPER SLOW!!!!
void transpose_constZ(std::size_t num_x, std::size_t num_y, std::size_t num_z, float * in, float * out)
{
	std::cout << "{X,Y,Z} ---> {Y,X,Z}" << std::endl;
	std::cout << "AVOID THIS FUNCTION AT ALL COSTS!!!! IT IS SUPER DUPER SLOW!!!!" << std::endl;
	int nthreads = boost::thread::hardware_concurrency();
	std::size_t dz = num_z / nthreads;
	std::vector<boost::thread*> threads;
	// can parallelize z
	for (std::size_t z = 0; z < num_z; z += dz)
	{
		threads.push_back(new boost::thread(transpose_constZ_worker
			, num_x
			, num_y
			, z
			, (z + dz < num_z) ? z + dz : num_z
			, num_z
			, in
			, out
			)
			);
	}
	for (std::size_t i = 0; i < threads.size(); i++)
	{
		threads[i]->join();
		delete threads[i];
	}
	threads.clear();
}

#endif transpose_h