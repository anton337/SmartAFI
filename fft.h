#ifndef fft_h
#define fft_h

#include <string.h>
#include <math.h>
#include <sstream>
#include <vector>
#include <fftw3.h>
#include <boost/thread.hpp>

boost::mutex _mutex;

void compute_convolution_2d_slices_worker(
	std::size_t start_x
	, std::size_t end_x
	, std::size_t num_y
	, std::size_t num_z
	, std::size_t num_f
	, float * a
	, float * b
	, float * ab
	)
{
	std::size_t id = start_x;
	float inv_factor = 1.0f / (num_y*num_z);
	float * slicea = new float[num_y*num_z];
	float * sliceb = new float[num_y*num_z];
	float * sliceab = new float[num_y*num_z];
	fftwf_complex * fslicea = new fftwf_complex[num_y*num_f];
	fftwf_complex * fsliceb = new fftwf_complex[num_y*num_f];
	fftwf_complex * fsliceab = new fftwf_complex[num_y*num_f];
	_mutex.lock();
	fftwf_plan forwarda = fftwf_plan_dft_r2c_2d((int)num_y, (int)num_z, slicea, fslicea, FFTW_ESTIMATE);
	fftwf_plan forwardb = fftwf_plan_dft_r2c_2d((int)num_y, (int)num_z, sliceb, fsliceb, FFTW_ESTIMATE);
	fftwf_plan inverseab = fftwf_plan_dft_c2r_2d((int)num_y, (int)num_z, fsliceab, sliceab, FFTW_ESTIMATE);
	_mutex.unlock();
	for (std::size_t x = start_x; x < end_x; x++)
	{
		//memcpy(&slicea[0], &a[x*num_y*num_z], num_y*num_z * 4);
		float * a_ptr = &a[x*num_y*num_z];
		for (int k = 0, size = num_y*num_z; k < size; k++)
		{
			slicea[k] = a_ptr[k];
		}
		fftwf_execute(forwarda);
		//memcpy(&sliceb[0], &b[x*num_y*num_z], num_y*num_z * 4);
		float * b_ptr = &b[x*num_y*num_z];
		for (int k = 0, size = num_y*num_z; k < size; k++)
		{
			sliceb[k] = b_ptr[k];
		}
		fftwf_execute(forwardb);
		for (std::size_t i = 0, len = num_y*num_f; i < len; i++)
		{
			fsliceab[i][0] = (fslicea[i][0] * fsliceb[i][0] - fslicea[i][1] * fsliceb[i][1])*inv_factor;
			fsliceab[i][1] = (fslicea[i][0] * fsliceb[i][1] + fslicea[i][1] * fsliceb[i][0])*inv_factor;
		}
		fftwf_execute(inverseab);
		//memcpy(&ab[x*num_y*num_z], &sliceab[0], num_y*num_z * 4);
		float * ab_ptr = &ab[x*num_y*num_z];
		for (int k = 0, size = num_y*num_z; k < size; k++)
		{
			ab_ptr[k] = sliceab[k];
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

void compute_convolution_2d_slices(
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
	std::size_t num_f = num_z / 2 + 1;
	std::vector<boost::thread * > threads;
	for (std::size_t x = 0; x < num_x; x += dx)
	{
		threads.push_back(
			new boost::thread(
			compute_convolution_2d_slices_worker
			, x
			, (x + dx<num_x) ? x + dx : num_x
			, num_y
			, num_z
			, num_f
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

void compute_convolution_2d_slices_fast_b_worker(
	std::size_t start_x
	, std::size_t end_x
	, std::size_t num_y
	, std::size_t num_z
	, float * a
	, float * b
	, float * ab
	)
{
	std::size_t padded_num_y = /*2 */ num_y;
	std::size_t padded_num_z = /*2 */ num_z;
	std::size_t padded_num_f = padded_num_z/2 + 1;
	float inv_factor = 1.0f / (padded_num_y*padded_num_z);
	float * slicea  = new float[padded_num_y*padded_num_z];
	float * sliceb  = new float[padded_num_y*padded_num_z];
	float * sliceab = new float[padded_num_y*padded_num_z];
	fftwf_complex * fslicea  = new fftwf_complex[padded_num_y*padded_num_f];
	fftwf_complex * fsliceb  = new fftwf_complex[padded_num_y*padded_num_f];
	fftwf_complex * fsliceab = new fftwf_complex[padded_num_y*padded_num_f];
	_mutex.lock();
	fftwf_plan forwarda  = fftwf_plan_dft_r2c_2d((int)padded_num_y, (int)padded_num_z, slicea, fslicea, FFTW_ESTIMATE);
	fftwf_plan forwardb  = fftwf_plan_dft_r2c_2d((int)padded_num_y, (int)padded_num_z, sliceb, fsliceb, FFTW_ESTIMATE);
	fftwf_plan inverseab = fftwf_plan_dft_c2r_2d((int)padded_num_y, (int)padded_num_z, fsliceab, sliceab, FFTW_ESTIMATE);
	_mutex.unlock();
	//memcpy(&sliceb[0], &b[x*num_y*num_z], num_y*num_z * 4);
	float * b_ptr = &b[0];
	memset(sliceb,0,padded_num_y*padded_num_z);
	for (int y = 0,k=0; y < num_y; y++)
	for (int z = 0; z < num_z; z++,k++)
	{
		sliceb[y*padded_num_z+z] = b_ptr[k];
	}
	fftwf_execute(forwardb);
	for (std::size_t x = start_x; x < end_x; x++)
	{
		//memcpy(&slicea[0], &a[x*num_y*num_z], num_y*num_z * 4);
		float * a_ptr = &a[x*num_y*num_z];
		for (int y = 0, k = 0; y < num_y; y++)
		for (int z = 0; z < num_z; z++, k++)
		{
			slicea[y*padded_num_z + z] = a_ptr[k];
		}
		fftwf_execute(forwarda);
		
		for (std::size_t i = 0, len = padded_num_y*padded_num_f; i < len; i++)
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
			ab_ptr[k] = sliceab[y*padded_num_z + z];
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

void compute_convolution_2d_slices_fast_b(
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
			compute_convolution_2d_slices_fast_b_worker
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
	std::size_t padded_num_y = /*2 */ num_y;
	std::size_t padded_num_z = /*2 */ num_z;
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

#endif
