//#define BOOST_ALL_DYN_LINK

#define _USE_MATH_DEFINES
#include <cmath>

#include <GL/glut.h>

//#include <jni.h>

#include <cuda.h>

#include <cufft.h>

#include <helper_cuda_drvapi.h>

#include <cuda_profiler_api.h>

#include <cuda_runtime.h>

//#include "AFI.h"

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

#include "sep_reader.h"

#include "sep_writer.h"

#include "marching_cubes.h"

#include "fault_sorter.h"

GLfloat xRotated, yRotated, zRotated;

float Distance = 5;

bool global_toggle = false;

DisplayUpdate * d_global_update = new DisplayUpdate();
DisplayUpdate * d_local_update = new DisplayUpdate();

MarchingCubes * marching_cubes = new MarchingCubes();

void init(void)
{
	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 0);
  glCullFace(GL_FRONT);
  glEnable(GL_CULL_FACE);
}

inline float min(float a,float b)
{
  return (a>b)?b:a;
}

inline float max(float a,float b)
{
  return (a>b)?a:b;
}

void draw_iso_surface()
{
  glCullFace(GL_FRONT);
  std::vector<polygon> const & vec = marching_cubes->get_polygons();
  float x_min = 1000000, x_max = -1000000;
  float y_min = 1000000, y_max = -1000000;
  float z_min = 1000000, z_max = -1000000;
  for(int i=0;i<vec.size();i++)
  {
    x_min = min(x_min,vec[i].p1.x);
    x_min = min(x_min,vec[i].p2.x);
    x_min = min(x_min,vec[i].p3.x);
    x_max = max(x_max,vec[i].p1.x);
    x_max = max(x_max,vec[i].p2.x);
    x_max = max(x_max,vec[i].p3.x);

    y_min = min(y_min,vec[i].p1.y);
    y_min = min(y_min,vec[i].p2.y);
    y_min = min(y_min,vec[i].p3.y);
    y_max = max(y_max,vec[i].p1.y);
    y_max = max(y_max,vec[i].p2.y);
    y_max = max(y_max,vec[i].p3.y);

    z_min = min(z_min,vec[i].p1.z);
    z_min = min(z_min,vec[i].p2.z);
    z_min = min(z_min,vec[i].p3.z);
    z_max = max(z_max,vec[i].p1.z);
    z_max = max(z_max,vec[i].p2.z);
    z_max = max(z_max,vec[i].p3.z);

  }
  glBegin(GL_TRIANGLES);
  float factor_x = 2.0f/(x_max-x_min);
  float factor_y = 2.0f/(y_max-y_min);
  float factor_z = 2.0f/(z_max-z_min);
  for(int i=0;i<vec.size();i++)
  {
    glColor3f ( 1.0f
              , 1.0f
              , 1.0f
              );
    glVertex3f  ( -1.0f+factor_x*(vec[i].p1.x-x_min)
                , -1.0f+factor_y*(vec[i].p1.y-y_min)
                , -1.0f+factor_z*(vec[i].p1.z-z_min)
                );
    glVertex3f  ( -1.0f+factor_x*(vec[i].p2.x-x_min)
                , -1.0f+factor_y*(vec[i].p2.y-y_min)
                , -1.0f+factor_z*(vec[i].p2.z-z_min)
                );
    glVertex3f  ( -1.0f+factor_x*(vec[i].p3.x-x_min)
                , -1.0f+factor_y*(vec[i].p3.y-y_min)
                , -1.0f+factor_z*(vec[i].p3.z-z_min)
                );
  }
  glEnd();
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
	glTranslatef(0.0, 0.0, -Distance);
	glRotatef(xRotated, 1.0, 0.0, 0.0);
	// rotation about Y axis
	glRotatef(yRotated, 0.0, 1.0, 0.0);
	// rotation about Z axis
	glRotatef(zRotated, 0.0, 0.0, 1.0);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  //draw_iso_surface();
  
  glCullFace(GL_BACK);
  if(global_toggle)
  {
	  glColor3f(1, 1, 1);
	  d_global_update->_mutex->lock();
	  std::stringstream ss;
	  ss << d_global_update->message;
	  ss << ":" << d_global_update->layer_index << "/" << d_global_update->layers;
	  drawstring(3, 3, -10, ss.str().c_str());
	  d_global_update->_mutex->unlock();
	  d_global_update->draw();
  }
  else
  {
	  glColor3f(1, 1, 1);
	  d_local_update->_mutex->lock();
	  std::stringstream ss;
	  ss << d_local_update->message;
	  ss << ":" << d_local_update->layer_index << "/" << d_local_update->layers;
	  drawstring(3, 3, -10, ss.str().c_str());
	  d_local_update->_mutex->unlock();
	  d_local_update->draw();
  }
  
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
  case 'g':
          global_toggle=!global_toggle;
          break;
	case 'p':
          d_global_update->pause = false; 
          d_local_update->pause = false; 
          break;
	case 'c':
          d_global_update->continuous_processing = !d_global_update->continuous_processing; 
          d_local_update->continuous_processing = !d_local_update->continuous_processing; 
          break;
	case 'w':
          d_global_update->increment_layer(); 
          d_local_update->increment_layer(); 
          break;
	case 's':
          d_global_update->decrement_layer(); 
          d_local_update->decrement_layer(); 
          break;
	case 'v':
          d_global_update->set_comprehensive(); 
          d_local_update->set_comprehensive(); 
          break;
	case 'q':Distance /= 1.01f; break;
	case 'a':Distance *= 1.01f; break;
  case 'j':xRotated++;break;
  case 'l':xRotated--;break;
  case 'i':yRotated++;break;
  case 'k':yRotated--;break;
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

//JNIEXPORT void JNICALL Java_AFI_AFI
//(JNIEnv * env, jobject obj, jlong nx, jlong ny, jlong nz, jfloatArray arr)
//{
//	printf("hello world\n");
//
//	std::size_t _nx = (std::size_t)nx;
//	std::size_t _ny = (std::size_t)ny;
//	std::size_t _nz = (std::size_t)nz;
//
//	std::cout << "nx:" << nx << "   ny:" << ny << "   nz:" << nz << std::endl;
//
//	std::size_t len = (std::size_t)env->GetArrayLength(arr);
//	std::cout << "len:" << len << std::endl;
//
//	float * _arr = new float[len]; 
//
//	memset(_arr, 0, len);
//
//	
//	jfloat * tmp = new jfloat[800];
//	
//	float val;
//	int size = 800;
//	std::cout << "get elements" << std::endl;
//	for (jsize k(0); k+size < len;k+=size)
//	{
//		//std::cout << "k=" << k << std::endl;
//		env->GetFloatArrayRegion(arr, k, size, tmp);
//		for (int i = 0; i < 800; i++)
//		{
//			//std::cout << i << " " << (float)tmp[i] << std::endl;
//			val = (float)tmp[i];
//			//val = (val>10000) ? 10000 : (val < -10000) ? -10000 : val;
//			_arr[(int)k + i] = val;
//		}
//	}
//
//	float max_val = -10000000;
//	float min_val = 10000000;
//	for (std::size_t k(0); k < len; k++)
//	{
//		max_val = (_arr[k]>max_val) ? _arr[k] : max_val;
//		min_val = (_arr[k]<min_val) ? _arr[k] : min_val;
//	}
//	float fact = 1.0f / (max_val-min_val);
//	for (std::size_t k(0); k < len; k++)
//	{
//		_arr[k] *= fact;
//	}
//
//	//std::cout << "release" << std::endl;
//	//env->ReleaseFloatArrayElements(arr, _arr, 0);
//	
//	std::cout << "done memory swap" << std::endl;
//	UnitTest * u_test = new UnitTest();
//	u_test -> operator () (d_global_update,d_local_update,_nx,_ny,_nz,_arr);
//
//	int argc = 1;
//	char** argv = new char*[1];
//	argv[0] = "AFI";
//
//	glutInit(&argc, argv);
//	//we initizlilze the glut. functions
//	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
//	glutInitWindowPosition(0, 0);
//	glutInitWindowSize(1200, 800);
//	glutCreateWindow(argv[0]);
//	init();
//	glutDisplayFunc(Draw);
//	glutReshapeFunc(reshape);
//	//Set the function for the animation.
//	glutIdleFunc(animation);
//	glutKeyboardFunc(keyboard);
//	glutMainLoop();
//
//}

int main(int argc, char** argv)
{

  {
    std::cout << "marching cubes: start" << std::endl;
    int nx=100;
    int ny=100;
    int nz=100;
    float * dat = new float[nx*ny*nz];
    for(int x=0,k=0;x<nx;x++)
      for(int y=0;y<ny;y++)
        for(int z=0;z<nz;z++,k++)
          dat[k] = (fabs(sqrt(pow(x-nx/2,2)+pow(y-ny/2,2))-nx/4)<0.5f)?1:0;
          //dat[k] = ((sqrt(pow(x-nx/2,2)+pow(y-ny/2,2))-nx/4)<0.0f)?1:0;
          //dat[k] = ((sqrt(pow(x-nx/2,2)+pow(y-ny/2,2)+pow(z-nz/2,2))-nx/4)<0.0f)?1:0;
    marching_cubes -> operator()(nx,ny,nz,dat);
    std::cout << "marching cubes: end" << std::endl;
    FaultSorter faultSorter;
    std::cout << "num explored:" << faultSorter(nx,ny,nz,dat) << std::endl;
    std::cout << "fault sorter: end" << std::endl;
  }

	//{
	//	std::cout << "press key to start" << std::endl;
	//	char ch;
	//	std::cin >> ch;
	//}

  	//SEPReader reader("/home/antonk/OpendTectData/Data/F3_Demo_2016_training_v6/SEP/fault_cube.sep");
	//SEPReader reader("C:/Users/H181523/OpendTect/oxy/oxy.hdr");
  	//SEPReader reader("/home/antonk/OpendTectData/Data/oxy/oxy.hdr");
  	SEPReader reader("/media/antonk/FreeAgent Drive/OpendTectData/Data/oxy/oxy.hdr");
	//SEPReader reader("/export/home/r5000/OXY/oxy.hdr");
  	//SEPReader reader("/d01/home/H181523/OXY/oxy.hdr");

	int ox = reader.o3;
	int oy = reader.o2;
	int oz = reader.o1;

	int nx = reader.n3;
	int ny = reader.n2;
	int nz = reader.n1;

	//std::cout << "O:" << ox << " " << oy << " " << oz << std::endl;

	//std::cout << "N:" << nx << " " << ny << " " << nz << std::endl;

	//{
	//  char ch;
	//  std::cin >> ch;
	//  std::cout << "press any key to continue..." << std::endl;
	//}

	
	float * arr = new float[nx*ny*nz];
	reader.read_sepval(&arr[0]
		, reader.o1
		, reader.o2
		, reader.o3
		, reader.n1
		, reader.n2
		, reader.n3
		);
		

	//int nx = 256;
	//int ny = 256;
	//int nz = 256;

	//float * arr = new float[nx*ny*nz];

	UnitTest * u_test = new UnitTest();
	u_test -> operator () (d_global_update, d_local_update, nx, ny, nz, arr);
	
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
	
	{
		char ch;
		std::cin >> ch;
	}
	return 0;
}

