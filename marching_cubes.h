#ifndef marching_cubes_h
#define marching_cubes_h

#include <vector>

#include "utility.h"

class MarchingCubes
{

  int ** faces;

  int ** edges;

  std::vector<std::vector<std::vector<int> > > cycleTable;

  char * type = NULL;

  struct point
  {
    float x,y,z;
    point(float _x,float _y,float _z)
      : x(_x) , y(_y) , z(_z) {   }
  };

  struct polygon
  {
    point p1,p2,p3;
    polygon(point _p1,point _p2,point _p3)
      : p1(_p1) , p2(_p2) , p3(_p3) {   }
  };

public:

  MarchingCubes()
  {
    faces = new int*[6];
    for(int i=0;i<6;i++)
    {
      faces[i] = new int[4];
    }
    faces[0][0] = 0;faces[0][1] = 2;faces[0][2] = 6;faces[0][3] = 4;
    faces[1][0] = 1;faces[1][1] = 3;faces[1][2] = 7;faces[1][3] = 5;
    faces[2][0] = 0;faces[2][1] = 4;faces[2][2] = 5;faces[2][3] = 1;
    faces[3][0] = 2;faces[3][1] = 6;faces[3][2] = 7;faces[3][3] = 3;
    faces[4][0] = 0;faces[4][1] = 1;faces[4][2] = 3;faces[4][3] = 2;
    faces[5][0] = 4;faces[5][1] = 5;faces[5][2] = 7;faces[5][3] = 6;

    edges = new int*[12];
    for(int i=0;i<12;i++)
    {
      edges[i] = new int[2];
    }
    edges[0] [0] = 0;edges[0] [1] = 1;
    edges[1] [0] = 2;edges[1] [1] = 3;
    edges[2] [0] = 6;edges[2] [1] = 7;
    edges[3] [0] = 4;edges[3] [1] = 5;
    edges[4] [0] = 0;edges[4] [1] = 2;
    edges[5] [0] = 4;edges[5] [1] = 6;
    edges[6] [0] = 5;edges[6] [1] = 7;
    edges[7] [0] = 1;edges[7] [1] = 3;
    edges[8] [0] = 0;edges[8] [1] = 4;
    edges[9] [0] = 1;edges[9] [1] = 5;
    edges[10][0] = 3;edges[10][1] = 7;
    edges[11][0] = 2;edges[11][1] = 6;

    initCycleTable();
  }

  void initCycleTable(){
    std::cout << "opening file" << std::endl;
    std::string line;
    std::ifstream myfile ("cycleTable");
    if (myfile.is_open())
    {
      while ( myfile.good() )
      {
        getline (myfile,line);
        std::vector<std::string> temp;
        stringtok(temp,line,"{}",false);
        std::vector<vector<int> > in1;
        for(int i=0;i<temp.size();i++){
          if(temp[i][0] != ','){
            std::vector<std::string> input;
            stringtok(input,temp[i],", ",false);
            std::vector<int> in2;
            for(int j=0;j<input.size();j++)
              in2.push_back(atoi(input[j].c_str()));
            in1.push_back(in2);
          }
        }
        cycleTable.push_back(in1);
      }
      myfile.close();
    }
    else std::cout << "Unable to open file" << std::endl; 
    std::cout << "done parsing cycleTable" << std::endl;
  }

  void operator()(std::size_t nx,std::size_t ny,std::size_t nz,float * dat)
  {
    std::cout << "p1" << std::endl;
    std::size_t size = nx*ny*nz;
    type = new char[size];

    for(int i=0;i<size;i++)
    {
      type[i] = 0;
    }
    std::cout << "p2" << std::endl;

    for(int i=0;i<size;i++)
    {
      if(fabs(dat[i])>1e-5)
      {
        type[i] |= 1 << 0;
      }
    }
    std::cout << "p3" << std::endl;

    float * tmp = &dat[1];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      if(z+1<nz)
      {
        if(fabs(tmp[i])>1e-5)
        {
          type[i] |= 1 << 1;
        }
      }
    }
    std::cout << "p4" << std::endl;

    tmp = &dat[nz];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      if(y+1<ny)
      {
        if(fabs(tmp[i])>1e-5)
        {
          type[i] |= 1 << 2;
        }
      }
    }
    std::cout << "p5" << std::endl;

    tmp = &dat[nz+1];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      if(z+1<nz && y+1<ny)
      {
        if(fabs(tmp[i])>1e-5)
        {
          type[i] |= 1 << 3;
        }
      }
    }
    std::cout << "p6" << std::endl;

    std::size_t x_offset = nz*ny;
    tmp = &dat[x_offset];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      if(x+1<x)
      {
        if(fabs(tmp[i])>1e-5)
        {
          type[i] |= 1 << 4;
        }
      }
    }
    std::cout << "p7" << std::endl;

    tmp = &dat[x_offset+1];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      if(x+1<nx && z+1<nz)
      {
        if(fabs(tmp[i])>1e-5)
        {
          type[i] |= 1 << 5;
        }
      }
    }
    std::cout << "p8" << std::endl;

    tmp = &dat[x_offset+nz];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      if(x+1<nx && y+1<ny)
      {
        if(fabs(tmp[i])>1e-5)
        {
          type[i] |= 1 << 6;
        }
      }
    }
    std::cout << "p9" << std::endl;

    tmp = &dat[x_offset+nz+1];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      if(x+1<nx && z+1<nz && y+1<ny)
      {
        if(fabs(tmp[i])>1e-5)
        {
          type[i] |= 1 << 7;
        }
      }
    }
    std::cout << "p10" << std::endl;

    std::vector<polygon> polygons;

    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      std::vector<std::vector<int> > const & cycle = cycleTable[(int)(unsigned char)type[i]];
      for(int k=0;k<cycle.size();k++)
      {
        std::vector<point> points;
        for(int j=0;j<cycle[k].size();j++)
        {
          int edge_ind = cycle[k][j];
          int pt1_ind = edges[edge_ind][0];
          int pt2_ind = edges[edge_ind][1];
          points.push_back( point ( x+0.5f*((int)((pt1_ind/4)%2>0) + (int)((pt2_ind/4)%2>0))
                                  , y+0.5f*((int)((pt1_ind/2)%2>0) + (int)((pt2_ind/2)%2>0))
                                  , z+0.5f*((int)((pt1_ind  )%2>0) + (int)((pt2_ind  )%2>0))
                                  )
                          );
        }
        switch(cycle[k].size())
        {
          case 3:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            break;
          case 4:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            polygons.push_back(polygon(points[2],points[1],points[3]));
            break;
          case 5:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            polygons.push_back(polygon(points[2],points[1],points[3]));
            polygons.push_back(polygon(points[1],points[3],points[4]));
            break;
          case 6:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            polygons.push_back(polygon(points[2],points[1],points[3]));
            polygons.push_back(polygon(points[2],points[3],points[4]));
            polygons.push_back(polygon(points[4],points[3],points[5]));
            break;
          case 7:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            polygons.push_back(polygon(points[2],points[1],points[3]));
            polygons.push_back(polygon(points[2],points[3],points[4]));
            polygons.push_back(polygon(points[4],points[3],points[5]));
            polygons.push_back(polygon(points[4],points[5],points[6]));
            break;
          default:
            break;
        }
      }
    }

    std::cout << "polygons:" << polygons.size() << std::endl;

    delete [] type;
    type = NULL;
  }
};

#endif

