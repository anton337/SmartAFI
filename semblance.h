#ifndef semblance_h
#define semblance_h

// traditional order is {X,Y,Z} where Z is fastest dimension, X is the slowest

void semblance_worker(int win, std::size_t start_x, std::size_t end_x, std::size_t num_y, std::size_t num_z, float * dat, float * num, float * den)
{
	for (std::size_t x = start_x; x < end_x; x++)
	{
		// dat[(x*num_y*num_z) + y*num_z + z]
		float * dat_X = &dat[x*num_y*num_z];
		float * num_X = &num[x*num_y*num_z];
		float * den_X = &den[x*num_y*num_z];
		memset(num_X, 0, num_y*num_z);
		memset(den_X, 0, num_y*num_z);
		std::size_t ind;
		float val;
		int factor = 2 * win + 1;
		factor *= factor;
		for (std::size_t y = win; y + 1 + win < num_y; y++)
		{
			for (std::size_t z = win; z + 1 + win < num_z; z++)
			{
				ind = y*num_z + z;
				for (int dy = -win; dy <= win; dy++)
				{
					for (int dz = -win; dz <= win; dz++)
					{
						val = dat_X[ind + dy*num_z + dz];
						num_X[ind] += val;
						den_X[ind] += val*val;
					}
				}
				num_X[ind] *= num_X[ind];
				den_X[ind] *= factor;
			}
		}
	}
}

// this is simple (non structure oriented) semblance
// we just calculate it along each {Y,Z} slice,
// so we can parallelize along X direction
void semblance(int win, std::size_t num_x, std::size_t num_y, std::size_t num_z, float * dat, float * num, float * den)
{
	int nthreads = boost::thread::hardware_concurrency();
	std::size_t dx = num_x / nthreads;
	std::vector<boost::thread*> threads;
	// can parallelize x
	for (std::size_t x = 0; x < num_x; x += dx)
	{
		threads.push_back(new boost::thread(semblance_worker
			, win
			, x
			, (x + dx < num_x) ? x + dx : num_x
			, num_y
			, num_z
			, dat
			, num
			, den
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

void semblance_worker_structure_oriented(int win, std::size_t start_x, std::size_t end_x, std::size_t num_y, std::size_t num_z, float * dat, float * num, float * den)
{
	for (std::size_t x = start_x; x < end_x; x++)
	{
		// dat[(x*num_y*num_z) + y*num_z + z]
		float * dat_X = &dat[x*num_y*num_z];
		float * num_X = &num[x*num_y*num_z];
		float * den_X = &den[x*num_y*num_z];
		float * dat_pX = &dat[(x + 1)*num_y*num_z];
		float * num_pX = &num[(x + 1)*num_y*num_z];
		float * den_pX = &den[(x + 1)*num_y*num_z];
		float * dat_mX = &dat[(x - 1)*num_y*num_z];
		float * num_mX = &num[(x - 1)*num_y*num_z];
		float * den_mX = &den[(x - 1)*num_y*num_z];
		memset(num_X, 0, num_y*num_z);
		memset(den_X, 0, num_y*num_z);
		std::size_t ind;
		float val;
		float val_compare;
		float val_final;
		float diff;
		float mdiff;
		int factor = 2 * win + 1;
		factor *= factor;
		for (std::size_t y = win; y + 1 + win < num_y; y++)
		{
			for (std::size_t z = win; z + 1 + win < num_z; z++)
			{
				ind = y*num_z + z;
				val_compare = dat_X[ind];
				for (int dy = -win; dy <= win; dy++)
				{
					for (int dz = -win; dz <= win; dz++)
					{
						mdiff = 10000;
						val = dat_X[ind + dy*num_z + dz];
						diff = fabs(val - val_compare);
						if (diff < mdiff){ mdiff = diff; val_final = val; }
						val = dat_pX[ind + dy*num_z + dz];
						diff = fabs(val - val_compare);
						if (diff < mdiff){ mdiff = diff; val_final = val; }
						val = dat_mX[ind + dy*num_z + dz];
						diff = fabs(val - val_compare);
						if (diff < mdiff){ mdiff = diff; val_final = val; }
						num_X[ind] += val_final;
						den_X[ind] += val_final*val_final;
					}
				}
				num_X[ind] *= num_X[ind];
				den_X[ind] *= factor;
			}
		}
	}
}

// this is simple (non structure oriented) semblance
// we just calculate it along each {Y,Z} slice,
// so we can parallelize along X direction
void semblance_structure_oriented(int win, std::size_t num_x, std::size_t num_y, std::size_t num_z, float * dat, float * num, float * den)
{
	int nthreads = boost::thread::hardware_concurrency();
	std::size_t dx = num_x / nthreads;
	std::vector<boost::thread*> threads;
	// can parallelize x
	for (std::size_t x = 1; x+1 < num_x; x += dx)
	{
		threads.push_back(new boost::thread(semblance_worker_structure_oriented
			, win
			, x
			, (x + dx < num_x-1) ? x + dx : num_x-1
			, num_y
			, num_z
			, dat
			, num
			, den
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

void semblance_div_worker(std::size_t start_x, std::size_t end_x, std::size_t num_y, std::size_t num_z, float * dat, float * num, float * den)
{
	for (std::size_t x = start_x; x < end_x; x++)
	{
		// dat[(x*num_y*num_z) + y*num_z + z]
		float * dat_X = &dat[x*num_y*num_z];
		float * num_X = &num[x*num_y*num_z];
		float * den_X = &den[x*num_y*num_z];
		for (std::size_t y = 0,ind=0; y < num_y; y++)
		{
			for (std::size_t z = 0; z < num_z; z++,ind++)
			{	
				dat_X[ind] = (num_X[ind] + 1e-5)/(den_X[ind] + 1e-5);
				dat_X[ind] = (dat_X[ind] < 0.0f) ? 0.0f : dat_X[ind];
				dat_X[ind] = (dat_X[ind] > 1.0f) ? 1.0f : dat_X[ind];
			}
		}
	}
}

void semblance_div(std::size_t num_x, std::size_t num_y, std::size_t num_z, float * dat, float * num, float * den)
{
	int nthreads = boost::thread::hardware_concurrency();
	std::size_t dx = num_x / nthreads;
	std::vector<boost::thread*> threads;
	// can parallelize x
	for (std::size_t x = 0; x < num_x; x += dx)
	{
		threads.push_back(new boost::thread(semblance_div_worker
			, x
			, (x + dx < num_x) ? x + dx : num_x
			, num_y
			, num_z
			, dat
			, num
			, den
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

#endif
