#ifndef tile_h
#define tile_h

#include "producer_consumer_queue.h"

#include "compute_manager.h"

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 100000
#endif

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

struct tile_params
{
  std::size_t nx;
  std::size_t ny;
  std::size_t nz;
	std::size_t num_x;
	std::size_t num_y;
	std::size_t num_z;
	std::size_t start_write_x;
	std::size_t start_write_y;
	std::size_t start_write_z;
	std::size_t size_write_x;
	std::size_t size_write_y;
	std::size_t size_write_z;
	std::size_t size_read_x;
	std::size_t size_read_y;
	std::size_t size_read_z;
  std::size_t pad;
  float * padded_input;
  float * padded_output;
  float * tile;
  ComputeDevice * device;
  ProducerConsumerQueue<ComputeDevice> * device_queue;
  tile_params(
    std::size_t _nx
    , std::size_t _ny
    , std::size_t _nz
	  , std::size_t _num_x
	  , std::size_t _num_y
	  , std::size_t _num_z
	  , std::size_t _start_write_x
	  , std::size_t _start_write_y
	  , std::size_t _start_write_z
	  , std::size_t _size_write_x
	  , std::size_t _size_write_y
	  , std::size_t _size_write_z
	  , std::size_t _size_read_x
	  , std::size_t _size_read_y
	  , std::size_t _size_read_z
    , std::size_t _pad
    , float * _padded_input
    , float * _padded_output
    , float * _tile
    , ComputeDevice * _device
    , ProducerConsumerQueue<ComputeDevice> * _device_queue
    )
    : nx(_nx)
    , ny(_ny)
    , nz(_nz)
    , num_x(_num_x)
	  , num_y(_num_y)
	  , num_z(_num_z)
	  , start_write_x(_start_write_x)
	  , start_write_y(_start_write_y)
	  , start_write_z(_start_write_z)
	  , size_write_x(_size_write_x)
	  , size_write_y(_size_write_y)
	  , size_write_z(_size_write_z)
	  , size_read_x(_size_read_x)
	  , size_read_y(_size_read_y)
	  , size_read_z(_size_read_z)
    , pad(_pad)
    , padded_input(_padded_input)
    , padded_output(_padded_output)
    , tile(_tile)
    , device(_device)
    , device_queue(_device_queue)
  {

  }
};

template<typename Functor>
void process_tile ( tile_params params
	                , DisplayUpdate * d_global_update
                  , DisplayUpdate * d_local_update
                  )
{
				get_tile(
					  params.num_x
					, params.num_y
					, params.num_z
					, params.start_write_x
					, params.start_write_y
					, params.start_write_z
					, params.size_write_x
					, params.size_write_y
					, params.size_write_z
					, params.pad
					, params.padded_input
					, params.tile
					);
				// do some processing on the tile
        Functor functor(d_global_update,d_local_update);
				functor(
					  params.size_read_x
					, params.size_read_y
					, params.size_read_z
					, params.pad
					, params.tile
          , params.device
					);
				put_tile(
					  params.num_x
					, params.num_y
					, params.num_z
					, params.start_write_x
					, params.start_write_y
					, params.start_write_z
					, params.size_write_x
					, params.size_write_y
					, params.size_write_z
					, params.pad
					, params.padded_output
					, params.tile
					);
		    std::cout << "update output" << std::endl;
		    d_global_update->update1  ( "output:"
                                  , params.num_x , params.pad , params.nx
                                  , params.num_y , params.pad , params.ny
                                  , params.num_z , params.pad , params.nz
                                  , params.padded_output
                                  );
        params.device_queue->put(params.device);
}

template<typename Functor>
void process(
  std::size_t nx
  , std::size_t ny
  , std::size_t nz
	, std::size_t num_x
	, std::size_t num_y
	, std::size_t num_z
	, std::size_t size_write_x
	, std::size_t size_write_y
	, std::size_t size_write_z
	, std::size_t pad
	, float * padded_input // num_x * num_y * num_z
	, float * padded_output // num_x * num_y * num_z
	, DisplayUpdate * d_global_update
  , DisplayUpdate * d_local_update
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

	ComputeManager * c_manager = new ComputeManager();

	int num_cpu_copies = 0;
	int num_gpu_copies = 1;
	std::vector<ComputeDevice*> device = c_manager->create_compute_devices(num_cpu_copies,num_gpu_copies);

  ProducerConsumerQueue<ComputeDevice> device_queue(BUFFER_SIZE);

  for(int k=0;k<device.size();k++)
  {
    device_queue.put(device[k]);
  }

	for (int x = 0, X=0; x < num_tiles_x; x++,X+=size_write_x)
	{
		for (int y = 0, Y=0; y < num_tiles_y; y++,Y += size_write_y)
		{
			for (int z = 0, Z=0; z < num_tiles_z; z++,Z += size_write_z)
			{
        ComputeDevice * d = device_queue.get();
        tile_params params
                    ( nx
                    , ny
                    , nz
	                  , num_x
	                  , num_y
	                  , num_z
	                  , X+pad
	                  , Y+pad
	                  , Z+pad
	                  , size_write_x
	                  , size_write_y
	                  , size_write_z
	                  , size_read_x
	                  , size_read_y
	                  , size_read_z
                    , pad
                    , padded_input
                    , padded_output
                    , tile
                    , d
                    , &device_queue
                    );
        boost::thread t(process_tile<Functor>,params,d_global_update,d_local_update);
			}
		}
	}

	c_manager->destroy();
	delete c_manager;
	delete[] tile;
}

#endif tile_h
