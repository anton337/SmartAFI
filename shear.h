#ifndef shear_h
#define shear_h

struct x_range
{
	std::size_t start_x;
	std::size_t end_x;
	std::size_t num_x;
	x_range(std::size_t _start_x
		, std::size_t _end_x
		, std::size_t _num_x
		)
		: start_x(_start_x)
		, end_x(_end_x)
		, num_x(_num_x)
	{

	}
};

enum SHEAR_DIRECTION { FORWARD, BACKWARD };

enum SHEAR_MODE { FFT, SINC8, HERMITIAN, LINEAR, CLOSEST_NEIGHBOR };

// If the data is ordered by {X,Y,Z}
// then we shear along the fastest direction Z, so num_z -> num_sheared_z
// and down the slowest direction X, since {Y,Z} slices are localized in memory
// each {Y,Z} slice is shifted in the Z direction by the same amount
// we can parallelize the X direction, and just process each {Y,Z} slice in a different thread
// FFT shift theorem:
//
//		x(n-d) <---> exp(-i wk d) X(wk)
//
//		we could also use Sinc interpolation
//		or something even cheaper such as closest neighbor interpolation
//		linear interpolation
//		or Hermitian interpolation
//
void shear_1d(SHEAR_MODE mode, std::size_t num_x, std::size_t num_y, std::size_t num_z, std::size_t num_sheared_z, float shear_z, float * in, float * out)
{

}

/*
void compute_convolution_2d_slices_fast_b_c2c_worker(
	std::size_t start_x
	, std::size_t end_x
	, std::size_t num_y
	, std::size_t num_z
	, float * a
	, float * b
	, float * ab
	)
{
	std::size_t padded_num_y = num_y;
	std::size_t padded_num_z = num_z;
	float inv_factor = 1.0f / (padded_num_y*padded_num_z);
	fftwf_complex * slicea = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * sliceb = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * sliceab = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fslicea = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fsliceb = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fsliceab = new fftwf_complex[padded_num_y*padded_num_z];
	_mutex.lock();
	fftwf_plan forwarda = fftwf_plan_dft_2d((int)padded_num_y, (int)padded_num_z, slicea, fslicea, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_plan forwardb = fftwf_plan_dft_2d((int)padded_num_y, (int)padded_num_z, sliceb, fsliceb, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_plan inverseab = fftwf_plan_dft_2d((int)padded_num_y, (int)padded_num_z, fsliceab, sliceab, FFTW_BACKWARD, FFTW_ESTIMATE);
	_mutex.unlock();
	//memcpy(&sliceb[0], &b[x*num_y*num_z], num_y*num_z * 4);
	float * b_ptr = &b[0];
	memset(sliceb, 0, padded_num_y*padded_num_z);
	for (int y = 0, k = 0; y < num_y; y++)
		for (int z = 0; z < num_z; z++, k++)
		{
			sliceb[y*padded_num_z + z][0] = b_ptr[k];
			sliceb[y*padded_num_z + z][1] = 0;
		}
	fftwf_execute(forwardb);
	for (std::size_t x = start_x; x < end_x; x++)
	{
		//memcpy(&slicea[0], &a[x*num_y*num_z], num_y*num_z * 4);
		float * a_ptr = &a[x*num_y*num_z];
		for (int y = 0, k = 0; y < num_y; y++)
			for (int z = 0; z < num_z; z++, k++)
			{
				slicea[y*padded_num_z + z][0] = a_ptr[k];
				slicea[y*padded_num_z + z][1] = 0;
			}
		fftwf_execute(forwarda);

		for (std::size_t i = 0, len = padded_num_y*padded_num_z; i < len; i++)
		{
			fsliceab[i][0] = (fslicea[i][0] * fsliceb[i][0] - fslicea[i][1] * fsliceb[i][1])*inv_factor;
			fsliceab[i][1] = (fslicea[i][0] * fsliceb[i][1] + fslicea[i][1] * fsliceb[i][0])*inv_factor;
		}
		fftwf_execute(inverseab);
		//memcpy(&ab[x*num_y*num_z], &sliceab[0], num_y*num_z * 4);
		float * ab_ptr = &ab[x*num_y*num_z];
		for (int y = 0, k = 0; y < num_y; y++)
			for (int z = 0; z < num_z; z++, k++)
			{
				ab_ptr[k] = sqrtf(sliceab[y*padded_num_z + z][0] * sliceab[y*padded_num_z + z][0] + sliceab[y*padded_num_z + z][1] * sliceab[y*padded_num_z + z][1]);
			}
	}
	_mutex.lock();
	fftwf_destroy_plan(forwarda);
	fftwf_destroy_plan(forwardb);
	fftwf_destroy_plan(inverseab);
	_mutex.unlock();
	delete[] slicea;
	delete[] sliceb;
	delete[] sliceab;
	delete[] fslicea;
	delete[] fsliceb;
	delete[] fsliceab;
}

void compute_convolution_2d_slices_fast_b_c2c(
	std::size_t num_x
	, std::size_t num_y
	, std::size_t num_z
	, float * a
	, float * b
	, float * ab
	)
{
	if (num_z % 2 == 1)
	{
		std::cout << "num_z should be even" << std::endl;
	}
	int nthreads = boost::thread::hardware_concurrency();
	std::size_t dx = num_x / nthreads;
	std::vector<boost::thread * > threads;
	for (std::size_t x = 0; x < num_x; x += dx)
	{
		threads.push_back(
			new boost::thread(
			compute_convolution_2d_slices_fast_b_c2c_worker
			, x
			, (x + dx<num_x) ? x + dx : num_x
			, num_y
			, num_z
			, a
			, b
			, ab
			)
			);
	}
	for (int i = 0; i < threads.size(); i++)
	{
		threads[i]->join();
		delete threads[i];
	}
}
*/

void fft_shear_2d_worker(
	int win
	, float shear_y
	, float shear_z
	, std::size_t start_x
	, std::size_t end_x
	, std::pair<std::size_t, std::size_t> ny
	, std::pair<std::size_t, std::size_t> nz
	, float * in
	, float * out
	)
{
	std::size_t num_y = ny.first;
	std::size_t num_sheared_y = ny.second;
	std::size_t num_z = nz.first;
	std::size_t num_sheared_z = nz.second;
	float Y,Z;
	float taper;
	float max_taper = 32;
	for (std::size_t x = start_x; x < end_x; x++)
	{
		float *  in_x = &in[x*        num_y*        num_z];
		float * out_x = &out[x*num_sheared_y*num_sheared_z];
		for (std::size_t y = 0; y < num_sheared_y; y++)
		{
			for (std::size_t z = 0; z < num_sheared_z; z++)
			{
				Y = (float)y + (int)num_y / 2 - (int)num_sheared_y / 2;
				Z = (float)z + (int)num_z / 2 - (int)num_sheared_z / 2;
				if ((int)Z >= win && (int)Z + 1 + win < num_z && (int)Y >= win && (int)Y + 1 + win < num_y)
				{
					out_x[y*num_sheared_z + z] = in_x[(int)Y*num_z + (int)Z];
				}
				else
				{
					taper = 0;
					if (Z < win){ taper = (fabs(Z - win)>taper) ? fabs(Z - win) : taper; Z = win;  }
					if (Z + win >= num_z){ taper = (fabs(Z + (int)win - (int)num_z)>taper) ? fabs(Z + (int)win - (int)num_z) : taper; Z = (int)num_z - (int)win - 1;  }
					if (Y < win){ taper = (fabs(Y - win)>taper) ? fabs(Y - win) : taper; Y = win; }
					if (Y + win >= num_y){ taper = (fabs(Y + (int)win - (int)num_y)>taper) ? fabs(Y + (int)win - (int)num_y) : taper; Y = (int)num_y - (int)win - 1; }
					out_x[y*num_sheared_z + z] = (taper>max_taper)?0:(1.0f-taper/max_taper)*in_x[(int)Y*num_z + (int)Z];
				}
			}
		}
	}

	

	std::size_t padded_num_y = num_sheared_y;
	std::size_t padded_num_z = num_sheared_z;
	float inv_factor = 1.0f / (padded_num_y*padded_num_z);
	fftwf_complex * slicea = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * sliceab = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fslicea = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fsliceb = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fsliceab = new fftwf_complex[padded_num_y*padded_num_z];
	_mutex.lock();
	fftwf_plan forwarda = fftwf_plan_dft_2d((int)padded_num_y, (int)padded_num_z, slicea, fslicea, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_plan inverseab = fftwf_plan_dft_2d((int)padded_num_y, (int)padded_num_z, fsliceab, sliceab, FFTW_BACKWARD, FFTW_ESTIMATE);
	_mutex.unlock();
	float shear_factor = 1.0f / padded_num_y;
	int _y, _z;
	for (std::size_t x = start_x; x < end_x; x++)
	{
		for (int y = 0, k = 0; y < padded_num_y; y++)
			for (int z = 0; z < padded_num_z; z++, k++)
			{
				_y = y;
				_z = z;
				if (_y>padded_num_y / 2) _y -= padded_num_y;
				if (_z>padded_num_z / 2) _z -= padded_num_z;
				fsliceb[y*padded_num_z + z][0] = cos(x*(_y*shear_y + _z*shear_z)*shear_factor);
				fsliceb[y*padded_num_z + z][1] = sin(x*(_y*shear_y + _z*shear_z)*shear_factor);
			}
		float * a_ptr = &out[x*num_sheared_y*num_sheared_z];
		for (int y = 0, k = 0; y < num_sheared_y; y++)
			for (int z = 0; z < num_sheared_z; z++, k++)
			{
				slicea[y*padded_num_z + z][0] = a_ptr[k];
				slicea[y*padded_num_z + z][1] = 0;
			}
		fftwf_execute(forwarda);

		for (std::size_t i = 0, len = padded_num_y*padded_num_z; i < len; i++)
		{
			fsliceab[i][0] = (fslicea[i][0] * fsliceb[i][0] - fslicea[i][1] * fsliceb[i][1])*inv_factor;
			fsliceab[i][1] = (fslicea[i][0] * fsliceb[i][1] + fslicea[i][1] * fsliceb[i][0])*inv_factor;
		}
		fftwf_execute(inverseab);
		float * ab_ptr = &out[x*num_sheared_y*num_sheared_z];
		for (int y = 0, k = 0; y < num_sheared_y; y++)
			for (int z = 0; z < num_sheared_z; z++, k++)
			{
				ab_ptr[k] = sqrtf(sliceab[y*padded_num_z + z][0] * sliceab[y*padded_num_z + z][0] + sliceab[y*padded_num_z + z][1] * sliceab[y*padded_num_z + z][1]);
			}
	}
	_mutex.lock();
	fftwf_destroy_plan(forwarda);
	fftwf_destroy_plan(inverseab);
	_mutex.unlock();
	delete[] slicea;
	delete[] sliceab;
	delete[] fslicea;
	delete[] fsliceb;
	delete[] fsliceab;

	

}

void fft_unshear_2d_worker(
	int win
	, float shear_y
	, float shear_z
	, std::size_t start_x
	, std::size_t end_x
	, std::pair<std::size_t, std::size_t> ny
	, std::pair<std::size_t, std::size_t> nz
	, float * in
	, float * out
	)
{
	std::size_t num_y = ny.first;
	std::size_t num_sheared_y = ny.second;
	std::size_t num_z = nz.first;
	std::size_t num_sheared_z = nz.second;
	

	
	std::size_t padded_num_y = num_sheared_y;
	std::size_t padded_num_z = num_sheared_z;
	float inv_factor = 1.0f / (padded_num_y*padded_num_z);
	fftwf_complex * slicea = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * sliceab = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fslicea = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fsliceb = new fftwf_complex[padded_num_y*padded_num_z];
	fftwf_complex * fsliceab = new fftwf_complex[padded_num_y*padded_num_z];
	_mutex.lock();
	fftwf_plan forwarda = fftwf_plan_dft_2d((int)padded_num_y, (int)padded_num_z, slicea, fslicea, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_plan inverseab = fftwf_plan_dft_2d((int)padded_num_y, (int)padded_num_z, fsliceab, sliceab, FFTW_BACKWARD, FFTW_ESTIMATE);
	_mutex.unlock();
	float shear_factor = 1.0f / padded_num_y;
	int _y, _z;
	for (std::size_t x = start_x; x < end_x; x++)
	{
		for (int y = 0, k = 0; y < padded_num_y; y++)
			for (int z = 0; z < padded_num_z; z++, k++)
			{
				_y = y;
				_z = z;
				if (_y>padded_num_y / 2) _y -= padded_num_y;
				if (_z>padded_num_z / 2) _z -= padded_num_z;
				fsliceb[y*padded_num_z + z][0] = cos(x*(_y*shear_y + _z*shear_z)*shear_factor);
				fsliceb[y*padded_num_z + z][1] =-sin(x*(_y*shear_y + _z*shear_z)*shear_factor);
			}
		float * a_ptr = &out[x*num_sheared_y*num_sheared_z];
		for (int y = 0, k = 0; y < num_sheared_y; y++)
			for (int z = 0; z < num_sheared_z; z++, k++)
			{
				slicea[y*padded_num_z + z][0] = a_ptr[k];
				slicea[y*padded_num_z + z][1] = 0;
			}
		fftwf_execute(forwarda);

		for (std::size_t i = 0, len = padded_num_y*padded_num_z; i < len; i++)
		{
			fsliceab[i][0] = (fslicea[i][0] * fsliceb[i][0] - fslicea[i][1] * fsliceb[i][1])*inv_factor;
			fsliceab[i][1] = (fslicea[i][0] * fsliceb[i][1] + fslicea[i][1] * fsliceb[i][0])*inv_factor;
		}
		fftwf_execute(inverseab);
		float * ab_ptr = &out[x*num_sheared_y*num_sheared_z];
		for (int y = 0, k = 0; y < num_sheared_y; y++)
			for (int z = 0; z < num_sheared_z; z++, k++)
			{
				ab_ptr[k] = sqrtf(sliceab[y*padded_num_z + z][0] * sliceab[y*padded_num_z + z][0] + sliceab[y*padded_num_z + z][1] * sliceab[y*padded_num_z + z][1]);
			}
	}
	_mutex.lock();
	fftwf_destroy_plan(forwarda);
	fftwf_destroy_plan(inverseab);
	_mutex.unlock();
	delete[] slicea;
	delete[] sliceab;
	delete[] fslicea;
	delete[] fsliceb;
	delete[] fsliceab;
	
	float Y,Z;
	for (std::size_t x = start_x; x < end_x; x++)
	{
		float *  in_x = &in[x*        num_y*        num_z];
		float * out_x = &out[x*num_sheared_y*num_sheared_z];
		
		for (std::size_t y = 0; y < num_sheared_y; y++)
		{
			for (std::size_t z = 0; z < num_sheared_z; z++)
			{
				Y = (float)y + (int)num_y / 2 - (int)num_sheared_y / 2;
				Z = (float)z + (int)num_z / 2 - (int)num_sheared_z / 2;
				if ((int)Z >= win && (int)Z + 1 + win < num_z && (int)Y >= win && (int)Y + 1 + win < num_y)
				{
					in_x[(int)Y*num_z + (int)Z] = out_x[y*num_sheared_z + z];
				}
				else
				{

				}
			}
		}
	}



}

void linear_shear_2d_worker(
	int win
	, float shear_y
	, float shear_z
	, x_range x_rng
	, std::pair<std::size_t, std::size_t> ny
	, std::pair<std::size_t, std::size_t> nz
	, float * in
	, float * out
	)
{
	std::size_t num_x = x_rng.num_x;
	std::size_t num_y = ny.first;
	std::size_t num_sheared_y = ny.second;
	std::size_t num_z = nz.first;
	std::size_t num_sheared_z = nz.second;
	for (std::size_t x = x_rng.start_x; x < x_rng.end_x; x++)
	{
		// in [(x*        num_y*        num_z) + y*        num_z + z]
		// out[(x*num_sheared_y*num_sheared_z) + y*num_sheared_z + z]
		float *  in_x = &in[x*        num_y*        num_z];
		float * out_x = &out[x*num_sheared_y*num_sheared_z];
		float Y, Z;
		float w0, w1;
		float _w0, _w1;
		float s0, s1;
		float w00, w01, w10, w11;
		float taper;
		float max_taper = 32;
		for (std::size_t y = 0; y < num_sheared_y; y++)
		{
			for (std::size_t z = 0; z < num_sheared_z; z++)
			{
				Y = (float)y + shear_y*((int)x) + (int)num_y / 2 - (int)num_sheared_y / 2;
				Z = (float)z + shear_z*((int)x) + (int)num_z / 2 - (int)num_sheared_z / 2;
				if ((int)Z >= win && (int)Z + 1 + win < num_z && (int)Y >= win && (int)Y + 1 + win < num_y)
				{					
					w0 = Z - (int)Z;w1 = Y - (int)Y;
					_w0 = 1 - w0;_w1 = 1 - w1;
					//w0 *= w0;w1 *= w1;w0 *= w0;w1 *= w1;
					//_w0 *= _w0;_w1 *= _w1;_w0 *= _w0;_w1 *= _w1;
					//s0 = 1.0f / (w0 + _w0);s1 = 1.0f / (w1 + _w1);
					//w0 *= s0;w1 *= s1;_w0 *= s0;_w1 *= s1;
					w00 = _w0*_w1;w01 = _w0*w1;w10 = w0*_w1;w11 = w0*w1;
					out_x[y*num_sheared_z + z] =
						  w00*in_x[(int)Y*num_z + (int)Z]
						+ w01*in_x[((int)Y + 1)*num_z + (int)Z]
						+ w10*in_x[(int)Y*num_z + (int)Z + 1]
						+ w11*in_x[((int)Y + 1)*num_z + (int)Z + 1];
				}
				else
				{
					taper = 0;
					if (Z < win){ taper = (fabs(Z - win)>taper) ? fabs(Z - win) : taper; Z = win; }
					if (Z + win >= num_z){ taper = (fabs(Z + (int)win - (int)num_z)>taper) ? fabs(Z + (int)win - (int)num_z) : taper; Z = (int)num_z - (int)win - 1; }
					if (Y < win){ taper = (fabs(Y - win)>taper) ? fabs(Y - win) : taper; Y = win; }
					if (Y + win >= num_y){ taper = (fabs(Y + (int)win - (int)num_y)>taper) ? fabs(Y + (int)win - (int)num_y) : taper; Y = (int)num_y - (int)win - 1; }
					out_x[y*num_sheared_z + z] = (taper>max_taper) ? 0 : (1.0f - taper / max_taper)*in_x[(int)Y*num_z + (int)Z];
				}
			}
		}
	}
}

void linear_unshear_2d_worker(
	float shear_y
	, float shear_z
	, x_range x_rng
	, std::pair<std::size_t, std::size_t> ny
	, std::pair<std::size_t, std::size_t> nz
	, float * in
	, float * out
	)
{
	std::size_t num_x = x_rng.num_x;
	std::size_t num_y = ny.first;
	std::size_t num_sheared_y = ny.second;
	std::size_t num_z = nz.first;
	std::size_t num_sheared_z = nz.second;
	for (std::size_t x = x_rng.start_x; x < x_rng.end_x; x++)
	{
		// in [(x*        num_y*        num_z) + y*        num_z + z]
		// out[(x*num_sheared_y*num_sheared_z) + y*num_sheared_z + z]
		float *  in_x = &in[x*        num_y*        num_z];
		float * out_x = &out[x*num_sheared_y*num_sheared_z];
		memset(in_x, 0, num_y*num_z);
		float Y, Z;
		float w0, w1;
		float _w0, _w1;
		float s0, s1;
		float w00, w01, w10, w11;
		for (std::size_t y = 0; y < num_sheared_y; y++)
		{
			for (std::size_t z = 0; z < num_sheared_z; z++)
			{
				Y = (float)y + shear_y*((int)x) + (int)num_y / 2 - (int)num_sheared_y / 2;
				Z = (float)z + shear_z*((int)x) + (int)num_z / 2 - (int)num_sheared_z / 2;
				if ((int)Z >= 0 && (int)Z + 1 < num_z)
				{
					if ((int)Y >= 0 && (int)Y + 1 < num_y)
					{
						/*
						w0 = Z - (int)Z;
						w1 = Y - (int)Y;
						_w0 = 1 - w0;
						_w1 = 1 - w1;
						w0 *= w0;
						w1 *= w1;
						w0 *= w0;
						w1 *= w1;
						_w0 *= _w0;
						_w1 *= _w1;
						_w0 *= _w0;
						_w1 *= _w1;
						s0 = 1.0f / (w0 + _w0);
						s1 = 1.0f / (w1 + _w1);
						w0 *= s0;
						w1 *= s1;
						_w0 *= s0;
						_w1 *= s1;
						w00 = _w0*_w1;
						w01 = _w0*w1;
						w10 = w0*_w1;
						w11 = w0*w1;
						*/
						in_x[(int)Y*num_z + (int)Z] = out_x[y*num_sheared_z + z];
						/*
						in_x[(int)Y*num_z + (int)Z] += w00*out_x[y*num_sheared_z + z];
						in_x[((int)Y + 1)*num_z + (int)Z] += w10*out_x[y*num_sheared_z + z];
						in_x[(int)Y*num_z + (int)Z + 1] += w11*out_x[y*num_sheared_z + z];
						in_x[((int)Y + 1)*num_z + (int)Z + 1] += w01*out_x[y*num_sheared_z + z];
						*/
					}
				}
			}
		}
	}
}

// If the data is ordered by {X,Y,Z}
// then we shear along the fastest directions Y and Z, so num_y -> num_sheared_y and num_z -> num_sheared_z
// and down the slowest direction X, since {Y,Z} slices are localized in memory
// each {Y,Z} slice is shifted in the (Y,Z) direction by the same amount
// we can parallelize the X direction, and just process each {Y,Z} slice in a different thread
// FFT shift theorem:
//
//		x(ny-dy,nz-dz) <---> exp(-i wky dy) exp(-i wkz dz) X(wky,wkz)
//
//		we could also use Sinc interpolation
//		or something even cheaper such as closest neighbor interpolation
//		linear interpolation
//		or Hermitian interpolation
//
void shear_2d(SHEAR_DIRECTION direction, SHEAR_MODE mode, int win, std::size_t num_x, std::size_t num_y, std::size_t num_z, std::size_t num_sheared_y, std::size_t num_sheared_z, float shear_y, float shear_z, float * in, float * out)
{
	switch (mode)
	{
	case LINEAR:
	{
		std::cout << "linear shear" << std::endl;
		int nthreads = boost::thread::hardware_concurrency();
		std::size_t dx = num_x / nthreads;
		std::vector<boost::thread*> threads;
		// can parallelize x
		for (std::size_t x = 0; x < num_x; x += dx)
		{
			threads.push_back((direction == FORWARD) ?
				new boost::thread(linear_shear_2d_worker
				, win
				, shear_y
				, shear_z
				, x_range( x, (x + dx < num_x) ? x + dx : num_x, num_x)
				, std::pair<std::size_t, std::size_t>(num_y, num_sheared_y)
				, std::pair<std::size_t, std::size_t>(num_z, num_sheared_z)
				, in
				, out
				)
				:
				new boost::thread(linear_unshear_2d_worker
				, shear_y
				, shear_z
				, x_range( x, (x + dx < num_x) ? x + dx : num_x, num_x)
				, std::pair<std::size_t, std::size_t>(num_y, num_sheared_y)
				, std::pair<std::size_t, std::size_t>(num_z, num_sheared_z)
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
		break;
	}
	case FFT:
	{
		std::cout << "fft shear" << std::endl;
		int nthreads = boost::thread::hardware_concurrency();
		std::size_t dx = num_x / nthreads;
		std::vector<boost::thread*> threads;
		// can parallelize x
		for (std::size_t x = 0; x < num_x; x += dx)
		{
			threads.push_back((direction == FORWARD) ?
				new boost::thread(fft_shear_2d_worker
				, win
				, shear_y
				, shear_z
				, x
				, (x + dx < num_x) ? x + dx : num_x
				, std::pair<std::size_t, std::size_t>(num_y, num_sheared_y)
				, std::pair<std::size_t, std::size_t>(num_z, num_sheared_z)
				, in
				, out
				)
				:
				new boost::thread(fft_unshear_2d_worker
				, win
				, shear_y
				, shear_z
				, x
				, (x + dx < num_x) ? x + dx : num_x
				, std::pair<std::size_t, std::size_t>(num_y, num_sheared_y)
				, std::pair<std::size_t, std::size_t>(num_z, num_sheared_z)
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
		break;
	}
	default:
	{
		std::cout << "this shear mode hasn't been implemented yet." << std::endl;
	}
	}
}

#endif shear_h