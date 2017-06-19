#ifndef marching_cubes_h
#define marching_cubes_h

#include <vector>

#include "utility.h"

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


class MarchingCubes
{

  int ** faces;

  int ** edges;

  std::vector<std::vector<std::vector<int> > > cycleTable;


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
    polygons.clear();

    bool c[8];

    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {

      int index = 0;
      c[0] = false;
      c[1] = false;
      c[2] = false;
      c[3] = false;
      c[4] = false;
      c[5] = false;
      c[6] = false;
      c[7] = false;

      if(true                  )c[0] = fabs(dat[i           ])>1e-5;
      if(z+1<nz                )c[1] = fabs(dat[i+1         ])>1e-5;
      if(        y+1<ny        )c[2] = fabs(dat[i  +nz      ])>1e-5;
      if(z+1<nz&&y+1<ny        )c[3] = fabs(dat[i+1+nz      ])>1e-5;
      if(                x+1<nx)c[4] = fabs(dat[i     +nz*ny])>1e-5;
      if(z+1<nz        &&x+1<nx)c[5] = fabs(dat[i+1   +nz*ny])>1e-5;
      if(        y+1<ny&&x+1<nx)c[6] = fabs(dat[i  +nz+nz*ny])>1e-5;
      if(z+1<nz&&y+1<ny&&x+1<nx)c[7] = fabs(dat[i+1+nz+nz*ny])>1e-5;

      for(int i=1,j=0;i<256;i*=2,j++)
        if(c[j])
          index += i;

      std::vector<std::vector<int> > const & cycle = cycleTable[index];

      for(int k=0;k<cycle.size();k++)
      {
        std::vector<point> points;
        for(int j=0;j<cycle[k].size();j++)
        {
          int edge_ind = cycle[k][j];
          int pt1_ind = edges[edge_ind][0];
          int pt2_ind = edges[edge_ind][1];
          float wt1 = 0;
          int x1 = x + (pt1_ind/4)%2;
          int y1 = y + (pt1_ind/2)%2;
          int z1 = z + (pt1_ind  )%2;
          if(x1<nx&&y1<ny&&z1<nz)
          {
            wt1 = dat[z1+nz*(y1+ny*x1)];
          }
          float wt2 = 0;
          int x2 = x + (pt2_ind/4)%2;
          int y2 = y + (pt2_ind/2)%2;
          int z2 = z + (pt2_ind  )%2;
          if(x2<nx&&y2<ny&&z2<nz)
          {
            wt2 = dat[z2+nz*(y2+ny*x2)];
          }
          points.push_back( point ( x+(wt1*(int)((pt1_ind/4)%2>0) + wt2*(int)((pt2_ind/4)%2>0))
                                  , y+(wt1*(int)((pt1_ind/2)%2>0) + wt2*(int)((pt2_ind/2)%2>0))
                                  , z+(wt1*(int)((pt1_ind  )%2>0) + wt2*(int)((pt2_ind  )%2>0))
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
            polygons.push_back(polygon(points[0],points[2],points[3]));
            break;
          
          case 5:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            polygons.push_back(polygon(points[0],points[2],points[3]));
            polygons.push_back(polygon(points[0],points[3],points[4]));
            break;

          case 6:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            polygons.push_back(polygon(points[0],points[2],points[3]));
            polygons.push_back(polygon(points[0],points[3],points[4]));
            polygons.push_back(polygon(points[0],points[4],points[5]));
            break;

          case 7:
            polygons.push_back(polygon(points[0],points[1],points[2]));
            polygons.push_back(polygon(points[0],points[2],points[3]));
            polygons.push_back(polygon(points[0],points[3],points[4]));
            polygons.push_back(polygon(points[0],points[4],points[5]));
            polygons.push_back(polygon(points[0],points[5],points[6]));
            break;
          
          default:
            break;
        }
      }
    }

    std::cout << "polygons:" << polygons.size() << std::endl;

  }

public:
  std::vector<polygon> const & get_polygons()
  {
    return polygons;
  }

private:
  std::vector<polygon> polygons;

};

#endif

