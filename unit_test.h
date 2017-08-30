#ifndef unit_test_h
#define unit_test_h

#include <iostream>

#include "AFIfunctor.h"

#include "pad.h"

#include "tile.h"

#include "fft.h"

#include "shear.h"

#include "transpose.h"

#include "semblance.h"


void run_some_tests	( DisplayUpdate * d_global_update
					          , DisplayUpdate * d_local_update
					          , std::size_t nx
					          , std::size_t ny
					          , std::size_t nz
					          , float * arr
					          )
{

	{
		//n1 = 876 (number of samples per trace)
		//n2 = 1701 (number of traces in inline direction)
		//n3 = 681 (number of traces in crossline direction)

		std::size_t mpad = 48;//32;
		std::size_t scale = 8;

		std::size_t tile_size_x = scale * 128 - 2 * scale * mpad;
		std::size_t tile_size_y = scale * 128 - 2 * scale * mpad;
		std::size_t tile_size_z = scale * 128 - 2 * scale * mpad;

		std::size_t orig_size_x = nx + 2 * mpad;
		std::size_t orig_size_y = ny + 2 * mpad;
		std::size_t orig_size_z = nz + 2 * mpad;

		std::size_t orig_size_multiple_x = tile_size_x*(orig_size_x / tile_size_x + 1);
		std::size_t orig_size_multiple_y = tile_size_y*(orig_size_y / tile_size_y + 1);
		std::size_t orig_size_multiple_z = tile_size_z*(orig_size_z / tile_size_z + 1);

		std::size_t padded_size_x = orig_size_multiple_x + 2 * mpad;
		std::size_t padded_size_y = orig_size_multiple_y + 2 * mpad;
		std::size_t padded_size_z = orig_size_multiple_z + 2 * mpad;

		std::cout << nx << " " << ny << " " << nz << std::endl;
		std::cout << orig_size_multiple_x << " " << orig_size_multiple_y << " " << orig_size_multiple_z << std::endl;

		float * orig_input = new float[padded_size_x*padded_size_y*padded_size_z];

		memset(orig_input, 0, padded_size_x*padded_size_y*padded_size_z);

		float * output = new float[padded_size_x*padded_size_y*padded_size_z];

		std::cout << "populating data" << std::endl;
		int X, Y, Z;
		for (std::size_t x = 0,k=0; x < padded_size_x; x++)
		{
			for (std::size_t y = 0; y < padded_size_y; y++)
			{
				for (std::size_t z = 0; z < padded_size_z; z++,k++)
				{
					X = x - mpad;
					if (X < 0)X = 0;
					if (X >= nx)X = nx - 1;
					Y = y - mpad;
					if (Y < 0)Y = 0;
					if (Y >= ny)Y = ny - 1;
					Z = z - mpad;
					if (Z < 0)Z = 0;
					if (Z >= nz)Z = nz - 1;
					orig_input[k] = arr[(X*ny+Y)*nz+Z];
				}
			}
		}

		delete[] arr;

	  float max_val = -10000000;
	  float min_val = 10000000;
	  for (std::size_t k(0), len(padded_size_x*padded_size_y*padded_size_z); k < len; k++)
	  {
	  	max_val = (orig_input[k]>max_val) ? orig_input[k] : max_val;
	  	min_val = (orig_input[k]<min_val) ? orig_input[k] : min_val;
	  }
	  float fact = 1.0f / (max_val-min_val);
	  for (std::size_t k(0), len(padded_size_x*padded_size_y*padded_size_z); k < len; k++)
	  {
	  	orig_input[k] *= fact;
	  }

		std::cout << "update input" << std::endl;
		d_global_update->update ( "original input:"
                            , padded_size_x , mpad , nx
                            , padded_size_y , mpad , ny
                            , padded_size_z , mpad , nz
                            , orig_input
                            );

		std::cout << "process" << std::endl;
		process<AFIfunctor>(nx, ny, nz, padded_size_x, padded_size_y, padded_size_z, tile_size_x, tile_size_y, tile_size_z, mpad, orig_input, output, d_global_update, d_local_update);

		std::cout << "update output" << std::endl;
		d_global_update->update1 ( "output:"
                             , padded_size_x , mpad , nx
                             , padded_size_y , mpad , ny
                             , padded_size_z , mpad , nz
                             , output
                             );

		std::cout << "done" << std::endl;

		//delete[] orig_input;
		//delete[] output;
	}

	
}

class UnitTest
{
public:
	void operator() (DisplayUpdate * d_global_update,DisplayUpdate * d_local_update,std::size_t nx,std::size_t ny,std::size_t nz,float * arr)
	{
		boost::thread * test_thread = new boost::thread(run_some_tests,d_global_update,d_local_update,nx,ny,nz,arr);

	}
};

#endif
