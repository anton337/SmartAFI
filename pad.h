#ifndef pad_h
#define pad_h

void array_pad(
	std::size_t num_x
	, std::size_t num_y
	, std::size_t num_z
	, std::size_t total_num_x
	, std::size_t total_num_y
	, std::size_t total_num_z
	, std::size_t pad
	, float * in
	, float * out
	)
{
	std::size_t padded_size_x = num_x + 2 * pad;
	std::size_t padded_size_y = num_y + 2 * pad;
	std::size_t padded_size_z = num_z + 2 * pad;
	if (padded_size_x > total_num_x || padded_size_y > total_num_y || padded_size_z > total_num_z)
	{
		std::cout << "yout have to allocate an output array with sufficient space for data and pad." << std::endl;
		return;
	}
	// float * padded_output = new float[total_num_x*total_num_y*total_num_z];
	int X, Y, Z;
	int Pad = (int)pad;
	std::size_t _X, _Y, _Z;
	for (std::size_t x = 0, k = 0; x < padded_size_x; x++)
	{
		for (std::size_t y = 0; y < padded_size_y; y++)
		{
			for (std::size_t z = 0; z < padded_size_z; z++, k++)
			{
				X = (int)x - Pad;
				if (X < 0)X = 0;
				if (X >= num_x) X = (int)num_x - 1;
				_X = (std::size_t)X;
				Y = (int)y - Pad;
				if (Y < 0)Y = 0;
				if (Y >= num_y) Y = (int)num_y - 1;
				_Y = (std::size_t)Y;
				Z = (int)z - Pad;
				if (Z < 0)Z = 0;
				if (Z >= num_z) Z = (int)num_z - 1;
				_Z = (std::size_t)Z;
				out[k] = in[(_X*num_y + _Y)*num_z + _Z];
			}
		}
	}
}

#endif
