//#define BOOST_ALL_DYN_LINK

#define _USE_MATH_DEFINES
#include <cmath>

#include <GL/glut.h>

#include <jni.h>

#include <cuda.h>

#include <cufft.h>

#include <helper_cuda_drvapi.h>

#include <cuda_profiler_api.h>

#include <cuda_runtime.h>

#include "AFI.h"

#include "unit_test.h"

#include "cuda_stuff.h"

#include "data_array.h"

#include "gpu_array.h"

#include "cpu_array.h"

#include "freq_gpu_array.h"

#include "freq_cpu_array.h"

#include "compute_device.h"

#include "gpu_device.h"

#include "cpu_device.h"

#include "compute_manager.h"

#include "random_stuff.h"

GLfloat xRotated, yRotated, zRotated;

float Distance = 5;

DisplayUpdate * d_global_update = new DisplayUpdate();
DisplayUpdate * d_local_update = new DisplayUpdate();

void init(void)
{
	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 0);
}

void drawstring(float x, float y, float z, const char *str)
{
	const char *c;
	glRasterPos3f(x, y, z);
	for (c = str; *c != '\0'; c++)
	{
		glColor3f(0.0, 0.0, 0.0);
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
	}
}

void DrawLocal(void)
{
	glMatrixMode(GL_MODELVIEW);
	// clear the drawing buffer.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glColor3f(1, 1, 1);
	d_local_update->_mutex->lock();
	std::stringstream ss;
	ss << d_local_update->message;
	ss << ":" << d_local_update->layer_index << "/" << d_local_update->layers;
	drawstring(3, 3, -10, ss.str().c_str());
	d_local_update->_mutex->unlock();
	glTranslatef(0.0, 0.0, -Distance);
	glRotatef(xRotated, 1.0, 0.0, 0.0);
	// rotation about Y axis
	glRotatef(yRotated, 0.0, 1.0, 0.0);
	// rotation about Z axis
	glRotatef(zRotated, 0.0, 0.0, 1.0);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	d_local_update->draw();
	glFlush();
}

void animation(void)
{
	DrawLocal();
	boost::this_thread::sleep(boost::posix_time::milliseconds(500));
}

void Draw(void)
{
	DrawLocal();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:exit(0);
	case 'p':d_local_update->pause = false; break;
	case 'c':d_local_update->continuous_processing = !d_local_update->continuous_processing; break;
	case 'w':d_local_update->increment_layer(); break;
	case 's':d_local_update->decrement_layer(); break;
	case 'v':d_local_update->set_comprehensive(); break;
	case 'q':Distance /= 1.01f; break;
	case 'a':Distance *= 1.01f; break;
	}
}

void reshape(int x, int y)
{
	if (y == 0 || x == 0) return;  //Nothing is visible then, so return
	//Set a new projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//Angle of view:40 degrees
	//Near clipping plane distance: 0.5
	//Far clipping plane distance: 20.0

	gluPerspective(40.0, (GLdouble)x / (GLdouble)y, 0.5, 20.0);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, x, y);  //Use the whole window for rendering
}

JNIEXPORT void JNICALL Java_AFI_AFI
(JNIEnv * env, jobject obj, jlong nx, jlong ny, jlong nz, jfloatArray arr)
{
	printf("hello world\n");

	std::size_t _nx = (std::size_t)nx;
	std::size_t _ny = (std::size_t)ny;
	std::size_t _nz = (std::size_t)nz;

	std::cout << "nx:" << nx << "   ny:" << ny << "   nz:" << nz << std::endl;

	std::size_t len = (std::size_t)env->GetArrayLength(arr);
	std::cout << "len:" << len << std::endl;

	float * _arr = new float[len]; 

	memset(_arr, 0, len);

	
	jfloat * tmp = new jfloat[800];
	
	float val;
	int size = 800;
	std::cout << "get elements" << std::endl;
	for (jsize k(0); k+size < len;k+=size)
	{
		//std::cout << "k=" << k << std::endl;
		env->GetFloatArrayRegion(arr, k, size, tmp);
		for (int i = 0; i < 800; i++)
		{
			//std::cout << i << " " << (float)tmp[i] << std::endl;
			val = (float)tmp[i];
			//val = (val>10000) ? 10000 : (val < -10000) ? -10000 : val;
			_arr[(int)k + i] = val;
		}
	}

	float max_val = -10000000;
	float min_val = 10000000;
	for (std::size_t k(0); k < len; k++)
	{
		_arr[k] = (_arr[k]>max_val) ? _arr[k] : max_val;
		_arr[k] = (_arr[k]<min_val) ? _arr[k] : min_val;
	}
	float fact = 1.0f / (max_val-min_val);
	for (std::size_t k(0); k < len; k++)
	{
		_arr[k] *= fact;
	}

	//std::cout << "release" << std::endl;
	//env->ReleaseFloatArrayElements(arr, _arr, 0);
	
	std::cout << "done memory swap" << std::endl;
	UnitTest * u_test = new UnitTest();
	u_test -> operator () (d_global_update,d_local_update,_nx,_ny,_nz,_arr);

	int argc = 1;
	char** argv = new char*[1];
	argv[0] = "AFI";

	glutInit(&argc, argv);
	//we initizlilze the glut. functions
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(1200, 800);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(Draw);
	glutReshapeFunc(reshape);
	//Set the function for the animation.
	glutIdleFunc(animation);
	glutKeyboardFunc(keyboard);
	glutMainLoop();

}

int main(int argc, char** argv)
{

	//quick_zsmooth_sanity_check();

	//char ch;
	//std::cin >> ch;

  //boost::thread * t = new boost::thread(test_gpu_afi);
  //test_gpu_afi();
	//test_gpu();
	//test_shear();
	
  int nx = 1024;
  int ny = 1024;
  int nz = 1024;
  float * arr = new float[nx*ny*nz];
  for(int x=0,k=0;x<nx;x++)
    for(int y=0;y<ny;y++)
      for(int z=0;z<nz;z++,k++)
      {
        arr[k] = 5;
      }
	UnitTest * u_test = new UnitTest();
	u_test -> operator () (d_global_update,d_local_update,nx,ny,nz,arr);
	
	glutInit(&argc, argv);
	//we initizlilze the glut. functions
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(1200, 800);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(Draw);
	glutReshapeFunc(reshape);
	//Set the function for the animation.
	glutIdleFunc(animation);
	glutKeyboardFunc(keyboard);
	glutMainLoop();

	return 0;
}

