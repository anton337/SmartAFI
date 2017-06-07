#ifndef tile_h
#define tile_h

void get_tile(
	std::size_t num_x
	, std::size_t num_y
	, std::size_t num_z
	, std::size_t start_write_x
	, std::size_t start_write_y
	, std::size_t start_write_z
	, std::size_t size_write_x
	, std::size_t size_write_y
	, std::size_t size_write_z
	, std::size_t pad
	, float * total_input
	, float * tile
	)
{
	std::size_t start_read_x = start_write_x - pad;
	std::size_t start_read_y = start_write_y - pad;
	std::size_t start_read_z = start_write_z - pad;
	if (start_read_x >= num_x || start_read_y >= num_y || start_read_z >= num_z)
	{
		std::cout << "tile boundary collision." << std::endl;
		std::cout << start_read_x << " " << num_x << std::endl;
		std::cout << start_read_y << " " << num_y << std::endl;
		std::cout << start_read_z << " " << num_z << std::endl;
		return;
	}
	std::size_t size_read_x = size_write_x + 2 * pad;
	std::size_t size_read_y = size_write_y + 2 * pad;
	std::size_t size_read_z = size_write_z + 2 * pad;
	if (start_read_x + size_read_x > num_x || start_read_y + size_read_y > num_y || start_read_z + size_read_z > num_z)
	{
		std::cout << "tile boundary collision." << std::endl;
		std::cout << start_read_x << " " << size_read_x << " " << num_x << std::endl;
		std::cout << start_read_y << " " << size_read_y << " " << num_y << std::endl;
		std::cout << start_read_z << " " << size_read_z << " " << num_z << std::endl;
		return;
	}
	// tile = new float[size_read_x*size_read_y*size_read_z];
	for (std::size_t X = start_read_x, x = 0; x < size_read_x; x++, X++)
	{
		for (std::size_t Y = start_read_y, y = 0; y < size_read_y; y++, Y++)
		{
			float * src = &total_input[(X*num_y + Y)*num_z + start_read_z];
			float * dst = &tile[(x*size_read_y + y)*size_read_z];
			//memcpy(dst, src, size_read_z);
			for (std::size_t z = 0; z < size_read_z; z++)
			{
				dst[z] = src[z];
			}
		}
	}
}

// do some processing on tile data, presumably

void put_tile(
	std::size_t num_x
	, std::size_t num_y
	, std::size_t num_z
	, std::size_t start_write_x
	, std::size_t start_write_y
	, std::size_t start_write_z
	, std::size_t size_write_x
	, std::size_t size_write_y
	, std::size_t size_write_z
	, std::size_t pad
	, float * total_input
	, float * tile
	)
{
	std::size_t size_read_x = size_write_x + 2 * pad;
	std::size_t size_read_y = size_write_y + 2 * pad;
	std::size_t size_read_z = size_write_z + 2 * pad;
	// tile = new float[size_read_x*size_read_y*size_read_z];
	for (std::size_t X = start_write_x, x = 0; x < size_write_x; x++, X++)
	{
		for (std::size_t Y = start_write_y, y = 0; y < size_write_y; y++, Y++)
		{
			float * src = &tile[((x+pad)*size_read_y + y+pad)*size_read_z+pad];
			float * dst = &total_input[(X*num_y + Y)*num_z + start_write_z];
			//memcpy(dst, src, size_write_z);
			for (std::size_t z = 0; z < size_write_z; z++)
			{
				dst[z] = src[z];
			}
		}
	}
}

template<typename Functor>
void process(
	std::size_t num_x
	, std::size_t num_y
	, std::size_t num_z
	, std::size_t size_write_x
	, std::size_t size_write_y
	, std::size_t size_write_z
	, std::size_t pad
	, float * padded_input // num_x * num_y * num_z
	, float * padded_output // num_x * num_y * num_z
	, Functor * functor
	)
{
	std::size_t tile_write_num_x = num_x - 2 * pad;
	std::size_t tile_write_num_y = num_y - 2 * pad;
	std::size_t tile_write_num_z = num_z - 2 * pad;
	std::size_t num_tiles_x = tile_write_num_x / size_write_x;
	std::size_t num_tiles_y = tile_write_num_y / size_write_y;
	std::size_t num_tiles_z = tile_write_num_z / size_write_z;
	if (num_tiles_x <= 0 || num_tiles_y <= 0 || num_tiles_z <= 0)
	{
		std::cout << "zero tiles fit in this array." << std::endl;
		return;
	}
	std::size_t size_read_x = size_write_x + 2 * pad;
	std::size_t size_read_y = size_write_y + 2 * pad;
	std::size_t size_read_z = size_write_z + 2 * pad;
	float * tile = new float[size_read_x*size_read_y*size_read_z];

	for (int x = 0, X=0; x < num_tiles_x; x++,X+=size_write_x)
	{
		for (int y = 0, Y=0; y < num_tiles_y; y++,Y += size_write_y)
		{
			for (int z = 0, Z=0; z < num_tiles_z; z++,Z += size_write_z)
			{
				get_tile(
					num_x
					, num_y
					, num_z
					, X + pad
					, Y + pad
					, Z + pad
					, size_write_x
					, size_write_y
					, size_write_z
					, pad
					, padded_input
					, tile
					);
				// do some processing on the tile
				functor -> operator() (
					size_read_x
					, size_read_y
					, size_read_z
					, pad
					, tile
					);
				put_tile(
					num_x
					, num_y
					, num_z
					, X + pad
					, Y + pad
					, Z + pad
					, size_write_x
					, size_write_y
					, size_write_z
					, pad
					, padded_output
					, tile
					);
			}
		}
	}
	delete[] tile;
}

#endif tile_h
