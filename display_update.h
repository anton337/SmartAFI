#ifndef display_update_h
#define display_update_h

#include <GL/glut.h>

#include "array_verify.h"

struct DisplayUpdate
{
	bool comprehensive;
	std::string message;
	bool pause;
	bool continuous_processing;
	boost::mutex * _mutex;
	std::size_t layer_index;
	std::size_t layers; // slowest
	std::size_t width;
	std::size_t height; // fastest
	std::size_t layers1; // slowest
	std::size_t width1;
	std::size_t height1; // fastest
	std::size_t layers2; // slowest
	std::size_t width2;
	std::size_t height2; // fastest
	float * data;
	float * data1;
	float * data2;
	DisplayUpdate()
	{
		comprehensive = true;
		message = "Not initialized";
		pause = true;
		continuous_processing = true;
		_mutex = new boost::mutex();
		layers = -1;
		width = -1;
		height = -1;
		layers1 = -1;
		width1 = -1;
		height1 = -1;
		layers2 = -1;
		width2 = -1;
		height2 = -1;
		layer_index = 0;
		data = NULL;
		data1 = NULL;
		data2 = NULL;
	}
	void set_comprehensive()
	{
		comprehensive = !comprehensive;
	}
	void increment_layer()
	{
		layer_index++;
	}
	void decrement_layer()
	{
		layer_index--;
	}
	void clear()
	{
		_mutex->lock();
		data = NULL;
		_mutex->unlock();
	}
	void clear1()
	{
		_mutex->lock();
		data1 = NULL;
		_mutex->unlock();
	}
	void clear2()
	{
		_mutex->lock();
		data2 = NULL;
		_mutex->unlock();
	}
	void update(std::string _message,std::size_t _layers, std::size_t _width, std::size_t _height, float * _data)
	{
		_mutex->lock();
		message = _message;
		width = _width;
		height = _height;
		layers = _layers;
		data = _data;
		_mutex->unlock();
		verify(message, layers, width, height, data);
		pause = true;
		if (!continuous_processing)
		{
			std::cout << "[Paused]" << std::endl;
		}
		while (pause&&!continuous_processing)
		{
			boost::this_thread::sleep(boost::posix_time::milliseconds(100));
		}
		std::cout << "[Continue]" << std::endl;
	}
	void update1(std::string _message, std::size_t _layers, std::size_t _width, std::size_t _height, float * _data)
	{
		_mutex->lock();
		message = _message;
		width1 = _width;
		height1 = _height;
		layers1 = _layers;
		data1 = _data;
		_mutex->unlock();
		verify(message, layers1, width1, height1, data1);
		pause = true;
		if (!continuous_processing)
		{
			std::cout << "[Paused]" << std::endl;
		}
		while (pause&&!continuous_processing)
		{
			boost::this_thread::sleep(boost::posix_time::milliseconds(100));
		}
		std::cout << "[Continue]" << std::endl;
	}
	void update2(std::string _message, std::size_t _layers, std::size_t _width, std::size_t _height, float * _data)
	{
		_mutex->lock();
		message = _message;
		width2 = _width;
		height2 = _height;
		layers2 = _layers;
		data2 = _data;
		_mutex->unlock();
		verify(message, layers2, width2, height2, data2);
		pause = true;
		if (!continuous_processing)
		{
			std::cout << "[Paused]" << std::endl;
		}
		while (pause&&!continuous_processing)
		{
			boost::this_thread::sleep(boost::posix_time::milliseconds(100));
		}
		std::cout << "[Continue]" << std::endl;
	}
	void draw()
	{
		_mutex->lock();
		if (data != NULL)
		{
			int _layer_index = layer_index;
			if (layer_index >= 0 && layer_index < layers)
			{

			}
			else
			{
				_layer_index = layers - 1;
			}

			if (_layer_index >= 0 && _layer_index < layers)
			{
				float dw = 2.0f / width;
				float dh = 2.0f / height;
				float val;
				glBegin(GL_QUADS);
				float * data_ptr = &data[_layer_index*width*height];
				float min_val = 1000000;
				float max_val = -1000000;
				for (std::size_t w = 0, k = 0; w < width; w++)
				{
					for (std::size_t h = 0; h < height; h++, k++)
					{
						val = data_ptr[k];
						max_val = (val > max_val) ? val : max_val;
						min_val = (val < min_val) ? val : min_val;
					}
				}
				//std::cout << min_val << " --- " << max_val << std::endl;
				float factor = 1.0f / (1e-5 + fabs(max_val - min_val));
				for (std::size_t w = 0, k = 0; w < width; w++)
				{
					for (std::size_t h = 0; h < height; h++, k++)
					{
						val = (data_ptr[k] - min_val)*factor;
						glColor3f(val, val, val);
						glVertex3f(-1 + (w+1)*dw, -1 + (h+1)*dh, 0);
						glVertex3f(-1 +  w   *dw, -1 + (h+1)*dh, 0);
						glVertex3f(-1 +  w   *dw, -1 +  h   *dh, 0);
						glVertex3f(-1 + (w+1)*dw, -1 +  h   *dh, 0);
					}
				}
				glEnd();
			}
		}
		if (data1 != NULL)
		{
			int _layer_index = layer_index;
			if (layer_index >= 0 && layer_index < layers1)
			{

			}
			else
			{
				_layer_index = layers1 - 1;
			}

			if (_layer_index >= 0 && _layer_index < layers1)
			{
				float dw = 2.0f / width1;
				float dh = 2.0f / height1;
				float val;
				glBegin(GL_QUADS);
				float * data_ptr = &data1[_layer_index*width1*height1];
				float min_val = 1000000;
				float max_val = -1000000;
				for (std::size_t w = 0, k = 0; w < width1; w++)
				{
					for (std::size_t h = 0; h < height1; h++, k++)
					{
						val = data_ptr[k];
						max_val = (val > max_val) ? val : max_val;
						min_val = (val < min_val) ? val : min_val;
					}
				}
				//std::cout << min_val << " --- " << max_val << std::endl;
				float factor = 1.0f / (1e-5 + fabs(max_val - min_val));
				for (std::size_t w = 0, k = 0; w < width1; w++)
				{
					for (std::size_t h = 0; h < height1; h++, k++)
					{
						val = (data_ptr[k] - min_val)*factor;
						glColor3f(val, val, val);
						glVertex3f(-3 + (w + 1)*dw, -1 + (h + 1)*dh, 0);
						glVertex3f(-3 + w   *dw, -1 + (h + 1)*dh, 0);
						glVertex3f(-3 + w   *dw, -1 + h   *dh, 0);
						glVertex3f(-3 + (w + 1)*dw, -1 + h   *dh, 0);
					}
				}
				glEnd();
			}
		}
		if (data2 != NULL)
		{
			int _layer_index = layer_index;
			if (layer_index >= 0 && layer_index < layers2)
			{

			}
			else
			{
				_layer_index = layers2 - 1;
			}

			if (_layer_index >= 0 && _layer_index < layers2)
			{
				float dw = 2.0f / width2;
				float dh = 2.0f / height2;
				float val;
				glBegin(GL_QUADS);
				float * data_ptr = &data2[_layer_index*width2*height2];
				float min_val = 1000000;
				float max_val = -1000000;
				for (std::size_t w = 0, k = 0; w < width2; w++)
				{
					for (std::size_t h = 0; h < height2; h++, k++)
					{
						val = data_ptr[k];
						max_val = (val > max_val) ? val : max_val;
						min_val = (val < min_val) ? val : min_val;
					}
				}
				//std::cout << min_val << " --- " << max_val << std::endl;
				float factor = 1.0f / (1e-5 + fabs(max_val - min_val));
				for (std::size_t w = 0, k = 0; w < width2; w++)
				{
					for (std::size_t h = 0; h < height2; h++, k++)
					{
						val = (data_ptr[k] - min_val)*factor;
						glColor3f(val, val, val);
						glVertex3f(1 + (w + 1)*dw, -1 + (h + 1)*dh, 0);
						glVertex3f(1 + w   *dw, -1 + (h + 1)*dh, 0);
						glVertex3f(1 + w   *dw, -1 + h   *dh, 0);
						glVertex3f(1 + (w + 1)*dw, -1 + h   *dh, 0);
					}
				}
				glEnd();
			}
		}
		_mutex->unlock();
	}
};

#endif
