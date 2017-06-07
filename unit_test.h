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
	std::cout << "Running some unit tests" << std::endl;
	/*
	{
		std::cout << "Testing transpose functionality:" << std::endl;

		std::size_t num_x = 128;
		std::size_t num_y = 128;
		std::size_t num_z = 128;

		float * data_xyz = new float[num_x*num_y*num_z]; // {X,Y,Z}
		for (std::size_t x = 0, k = 0; x < num_x; x++)
			for (std::size_t y = 0; y < num_y; y++)
				for (std::size_t z = 0; z < num_z; z++, k++)
				{
					data_xyz[k] = (float)k;
				}

		std::cout << data_xyz[55] << " " << data_xyz[55 + num_z] << std::endl;
		float * data_xzy = new float[num_x*num_y*num_z];
		transpose_constX(num_x, num_y, num_z, data_xyz, data_xzy); // {X,Y,Z} --> {X,Z,Y}
		delete[] data_xyz;

		std::cout << data_xzy[55] << " " << data_xzy[55 + num_z] << std::endl;
		float * data_yzx = new float[num_x*num_y*num_z];
		transpose_constY(num_x, num_z, num_y, data_xzy, data_yzx); // {X,Z,Y} --> {Y,Z,X}
		delete[] data_xzy;

		std::cout << data_yzx[55] << " " << data_yzx[55 + num_z] << std::endl;
		float * data_yxz = new float[num_x*num_y*num_z];
		transpose_constX(num_y, num_z, num_x, data_yzx, data_yxz); // {Y,Z,X} --> {Y,X,Z}
		delete[] data_yzx;

		std::cout << data_yxz[55] << " " << data_yxz[55 + num_z] << std::endl;
		float * data_xyz_final = new float[num_x*num_y*num_z];
		transpose_constZ(num_y, num_x, num_z, data_yxz, data_xyz_final); // {Y,X,Z} --> {X,Y,Z}
		delete[] data_yxz;

		std::cout << data_xyz_final[55] << " " << data_xyz_final[55 + num_z] << std::endl;
		delete[] data_xyz_final;
	}

	{
		std::cout << "Testing fft functionality:" << std::endl;

		std::size_t num_x = 128;
		std::size_t num_y = 128;
		std::size_t num_z = 128;

		float * data_A = new float[num_x*num_y*num_z]; // A
		for (std::size_t x = 0, k = 0; x < num_x; x++)
			for (std::size_t y = 0; y < num_y; y++)
				for (std::size_t z = 0; z < num_z; z++, k++)
				{
					data_A[k] = (float)k;
				}

		float * data_B = new float[num_x*num_y*num_z]; // B
		for (std::size_t x = 0, k = 0; x < num_x; x++)
			for (std::size_t y = 0; y < num_y; y++)
				for (std::size_t z = 0; z < num_z; z++, k++)
				{
					data_B[k] = (float)k;
				}

		float * data_AB = new float[num_x*num_y*num_z]; // A * B 

		compute_convolution_2d_slices(num_x, num_y, num_z, data_A, data_B, data_AB);

		std::cout << data_A[10] << " * " << data_B[10] << " === " << data_AB[10] << std::endl;

		delete[] data_A;
		delete[] data_B;
		delete[] data_AB;
	}

	{
		std::cout << "Testing semblance functionality:" << std::endl;

		std::size_t num_x = 128;
		std::size_t num_y = 128;
		std::size_t num_z = 128;

		float * data = new float[num_x*num_y*num_z];
		float dat[5];
		dat[0] = 1.0f;
		dat[1] = 1.05f;
		dat[2] = 0.95f;
		dat[3] = 1.1f;
		dat[4] = 1.03f;
		for (std::size_t x = 0, k = 0; x < num_x; x++)
			for (std::size_t y = 0; y < num_y; y++)
				for (std::size_t z = 0; z < num_z; z++, k++)
				{
					data[k] = dat[k % 5];
				}

		float * num = new float[num_x*num_y*num_z];
		float * den = new float[num_x*num_y*num_z];

		int win = 1;

		semblance(win, num_x, num_y, num_z, data, num, den);

		std::cout << data[54355] << " " << num[54355] << " " << den[54355] << " semb:" << (num[54355] + 1e-5) / (den[54355] + 1e-5) << std::endl;

		delete[] data;
		delete[] num;
		delete[] den;
	}

	{
		std::cout << "Testing shear functionality:" << std::endl;

		std::size_t num_x = 128;
		std::size_t num_y = 128;
		std::size_t num_z = 128;

		float * data_orig = new float[num_x*num_y*num_z];
		float dat[5];
		dat[0] = 1.0f;
		dat[1] = 1.05f;
		dat[2] = 0.95f;
		dat[3] = 1.1f;
		dat[4] = 1.03f;
		for (std::size_t x = 0, k = 0; x < num_x; x++)
			for (std::size_t y = 0; y < num_y; y++)
				for (std::size_t z = 0; z < num_z; z++, k++)
				{
					data_orig[k] = dat[k % 5];
				}

		std::size_t num_sheared_y = num_y * 2;
		std::size_t num_sheared_z = num_z * 2;

		float * data_sheared = new float[num_x*num_sheared_y*num_sheared_z];

		float shear_y = 0.05f;
		float shear_z = 0.025f;

		shear_2d(FORWARD, LINEAR, num_x, num_y, num_z, num_sheared_y, num_sheared_z, shear_y, shear_z, data_orig, data_sheared);

		// do some processing in sheared domain, presumably

		float * data_unsheared = new float[num_x*num_y*num_z];
		shear_2d(BACKWARD, LINEAR, num_x, num_y, num_z, num_sheared_y, num_sheared_z, shear_y, shear_z, data_unsheared, data_sheared);

		std::cout << data_orig[54353] << " === " << data_unsheared[54353] << std::endl;

		delete[] data_orig;
		delete[] data_sheared;
		delete[] data_unsheared;
	}
	*/
	/*
	{
		std::cout << "Test tile functionality:" << std::endl;

		std::size_t total_size_x = 1024;
		std::size_t total_size_y = 1024;
		std::size_t total_size_z = 1024;

		float * total_input = new float[total_size_x*total_size_y*total_size_z];
		float dat[5];
		dat[0] = 1.0f;
		dat[1] = 1.05f;
		dat[2] = 0.95f;
		dat[3] = 1.1f;
		dat[4] = 1.03f;
		for (std::size_t x = 0, k = 0; x < total_size_x; x++)
			for (std::size_t y = 0; y < total_size_y; y++)
				for (std::size_t z = 0; z < total_size_z; z++, k++)
				{
					total_input[k] = dat[k % 5];
				}
		d_local_update->update(total_size_x, total_size_y, total_size_z, total_input);
		float * tile = NULL;

		std::size_t start_write_x = 300;
		std::size_t start_write_y = 400;
		std::size_t start_write_z = 500;
		std::size_t size_write_x = 256;
		std::size_t size_write_y = 256;
		std::size_t size_write_z = 256;
		std::size_t pad = 90;
		std::size_t size_read_x = size_write_x + 2 * pad;
		std::size_t size_read_y = size_write_y + 2 * pad;
		std::size_t size_read_z = size_write_z + 2 * pad;

		tile = new float[size_read_x*size_read_y*size_read_z];

		get_tile(total_size_x, total_size_y, total_size_z, start_write_x, start_write_y, start_write_z, size_write_x, size_write_y, size_write_z, pad, total_input, tile);
		d_local_update->update(size_read_x, size_read_y, size_read_z, tile);
		// do some processing on tile data, presumably

		put_tile(total_size_x, total_size_y, total_size_z, start_write_x, start_write_y, start_write_z, size_write_x, size_write_y, size_write_z, pad, total_input, tile);

		delete[] tile;
		delete[] total_input;
	}
	*/
	/*
	{
		std::cout << "Test pad functionality:" << std::endl;

		std::size_t orig_size_x = 1024;
		std::size_t orig_size_y = 1024;
		std::size_t orig_size_z = 1024;
		std::size_t mpad = 60;

		float * orig_input = new float[orig_size_x*orig_size_y*orig_size_z];
		float dat[5];
		dat[0] = 1.0f;
		dat[1] = 1.05f;
		dat[2] = 0.95f;
		dat[3] = 1.1f;
		dat[4] = 1.03f;
		for (std::size_t x = 0, k = 0; x < orig_size_x; x++)
			for (std::size_t y = 0; y < orig_size_y; y++)
				for (std::size_t z = 0; z < orig_size_z; z++, k++)
				{
					orig_input[k] = dat[k % 5];
				}

		std::size_t padded_size_x = orig_size_x + 2 * mpad;
		std::size_t padded_size_y = orig_size_y + 2 * mpad;
		std::size_t padded_size_z = orig_size_z + 2 * mpad;

		float * padded_output = new float[padded_size_x*padded_size_y*padded_size_z];

		array_pad(orig_size_x, orig_size_y, orig_size_z, padded_size_x, padded_size_y, padded_size_z, mpad, orig_input, padded_output);

		delete[] orig_input;
		delete[] padded_output;
	}
	*/
	/*
	{
		std::cout << "Test processing:" << std::endl;
		std::size_t mpad = 32;
		std::size_t scale = 2;
		std::size_t tile_size_x = scale*128 - 2 * mpad;
		std::size_t tile_size_y = scale*128 - 2 * mpad;
		std::size_t tile_size_z = scale*128 - 2 * mpad;
		std::size_t padded_size_x = orig_size_x + 2 * mpad;
		std::size_t padded_size_y = orig_size_y + 2 * mpad;
		std::size_t padded_size_z = orig_size_z + 2 * mpad;

		std::size_t orig_size_x = 3*tile_size_x;
		std::size_t orig_size_y = 3*tile_size_y;
		std::size_t orig_size_z = 3*tile_size_z;

		float * orig_input = new float[padded_size_x*padded_size_y*padded_size_z];
		float * output = new float[padded_size_x*padded_size_y*padded_size_z];
		float dat[7];
		dat[0] = 1.0f;
		dat[1] = 1.05f;
		dat[2] = 0.95f;
		dat[3] = 1.1f;
		dat[4] = 1.03f;
		dat[5] = 0.93f;
		dat[6] = 0.97f;
		for (std::size_t x = 0, k = 0; x < padded_size_x; x++)
			for (std::size_t y = 0; y < padded_size_y; y++)
				for (std::size_t z = 0; z < padded_size_z; z++, k++)
				{
					output[k] = 0;
					orig_input[k] = dat[k % 7];
					if ( pow((int)x - (int)padded_size_x / 2, 2) 
					   + pow((int)y - (int)padded_size_y / 2, 2) 
					   + pow((int)z - (int)padded_size_z / 2, 2) 
				       > pow(0.5*padded_size_x/2, 2)
					   )
					{
						orig_input[k] += 5.0f;
					}
					if (pow((int)x - (int)padded_size_x / 2, 2)
						+ pow((int)y - (int)padded_size_y / 2, 2)
						+ pow((int)z - (int)padded_size_z / 2, 2)
						> pow(0.75*padded_size_x / 2, 2)
						)
					{
						orig_input[k] += 5.0f;
					}
					if (pow((int)x - (int)padded_size_x / 2, 2)
						+ pow((int)y - (int)padded_size_y / 2, 2)
						+ pow((int)z - (int)padded_size_z / 2, 2)
						> pow(1.00*padded_size_x / 2, 2)
						)
					{
						orig_input[k] += 5.0f;
					}
				}
		d_local_update->update("original input:",padded_size_x, padded_size_y, padded_size_z, orig_input);

		AFIfunctor * functor = new AFIfunctor(d_local_update);

		process(padded_size_x, padded_size_y, padded_size_z, tile_size_x, tile_size_y, tile_size_z, mpad, orig_input, output, functor);

		d_local_update->update("output:", padded_size_x, padded_size_y, padded_size_z, output);

		delete[] orig_input;
		//delete[] output;
	}
	*/


	{
		std::cout << "Test real data processing:" << std::endl;

		//n1 = 876 (number of samples per trace)
		//n2 = 1701 (number of traces in inline direction)
		//n3 = 681 (number of traces in crossline direction)

		std::size_t mpad = 32;
		std::size_t scale = 4;

		std::size_t tile_size_x = scale * 128 - 2 * mpad;
		std::size_t tile_size_y = scale * 128 - 2 * mpad;
		std::size_t tile_size_z = scale * 128 - 2 * mpad;

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
					//std::cout << "x:" << x << " y:" << y << " z:" << z << "   k:" << k << "  total:" << (nx*ny*nz) << " " << (((z + mpad)*padded_size_y + (y + mpad))*padded_size_z + z + mpad) << std::endl;
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

		/*
		float dat[7];
		dat[0] = 1.0f;
		dat[1] = 1.05f;
		dat[2] = 0.95f;
		dat[3] = 1.1f;
		dat[4] = 1.03f;
		dat[5] = 0.93f;
		dat[6] = 0.97f;
		for (std::size_t x = 0, k = 0; x < padded_size_x; x++)
			for (std::size_t y = 0; y < padded_size_y; y++)
				for (std::size_t z = 0; z < padded_size_z; z++, k++)
				{
					output[k] = 0;
					orig_input[k] = dat[k % 7];
					if (pow((int)x - (int)padded_size_x / 2, 2)
						+ pow((int)y - (int)padded_size_y / 2, 2)
						+ pow((int)z - (int)padded_size_z / 2, 2)
						> pow(0.5*padded_size_x / 2, 2)
						)
					{
						orig_input[k] += 5.0f;
					}
					if (pow((int)x - (int)padded_size_x / 2, 2)
						+ pow((int)y - (int)padded_size_y / 2, 2)
						+ pow((int)z - (int)padded_size_z / 2, 2)
						> pow(0.75*padded_size_x / 2, 2)
						)
					{
						orig_input[k] += 5.0f;
					}
					if (pow((int)x - (int)padded_size_x / 2, 2)
						+ pow((int)y - (int)padded_size_y / 2, 2)
						+ pow((int)z - (int)padded_size_z / 2, 2)
						> pow(1.00*padded_size_x / 2, 2)
						)
					{
						orig_input[k] += 5.0f;
					}
				}
				*/

		std::cout << "update input" << std::endl;
		d_global_update->update("original input:", padded_size_x, padded_size_y, padded_size_z, orig_input);

		AFIfunctor * functor = new AFIfunctor(d_global_update,d_local_update);

		std::cout << "process" << std::endl;
		process(padded_size_x, padded_size_y, padded_size_z, tile_size_x, tile_size_y, tile_size_z, mpad, orig_input, output, functor);

		std::cout << "update output" << std::endl;
		d_global_update->update1("output:", padded_size_x, padded_size_y, padded_size_z, output);

		std::cout << "done" << std::endl;

		delete[] orig_input;
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
