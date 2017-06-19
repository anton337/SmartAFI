#ifndef fault_sorter_h
#define fault_sorter_h

#include <queue>

#include <map>

class FaultSorter
{

  struct fault_details
  {
    std::size_t ind;
    std::size_t num_elements;
    float max_elem;
    fault_details(std::size_t _ind,std::size_t _num_elements,float _max_elem)
      : ind(_ind), num_elements(_num_elements), max_elem(_max_elem)
    {

    }
  };

  std::map<std::size_t,fault_details*> fault_map;

  // bfs : bredth first search
  std::size_t bfs(std::size_t cluster_ind,std::size_t start,int nx,int ny,int nz,float * dat,std::size_t * neigh)
  {
    float max_val = dat[start];
    int w=2;
    std::size_t num_explored = 0;
    std::size_t size = nx*ny*nz;
    std::queue<std::size_t> Q;
    Q.push(start);
    while(!Q.empty())
    {
      std::size_t ind = Q.front(),ind_next;
      Q.pop();
      if(neigh[ind]==0&&dat[ind]>1e-5)
      {
        num_explored++;
        neigh[ind] = cluster_ind;
        for(int dx=-w;dx<=w;dx++)
        for(int dy=-w;dy<=w;dy++)
        for(int dz=-w;dz<=w;dz++)
        if(dx!=0&&dy!=0&&dz!=0)
        {
          ind_next = ind+dz+nz*dy+nz*ny*dx;
          if(ind_next<size&&dat[ind_next]>1e-5&&0.95f*max_val<dat[ind_next])
          {
            Q.push(ind_next);
          }
        }
      }
    }
    return num_explored;
  }
  public:
  // finds faults and categorizes them
  //      + dat = thin data set
  std::size_t operator()(int nx,int ny,int nz,float * dat)
  {
    std::size_t num_explored = 0;
    std::size_t * index = new std::size_t[nx*ny*nz];
    for(int x=0,i=0;x<nx;x++)
    for(int y=0;y<ny;y++)
    for(int z=0;z<nz;z++,i++)
    {
      index[i] = 0;
    }
    std::size_t num = 100000;
    std::size_t max_num = 100;
    std::size_t num_connected_component = 0;
    std::size_t connected_component_ind = 1;
    for(std::size_t ind=1;ind<=num&&connected_component_ind<max_num;ind++)
    {
      std::size_t max_ind = -1;
      float max_dat = 0;
      for(int x=0,i=0;x<nx;x++)
      for(int y=0;y<ny;y++)
      for(int z=0;z<nz;z++,i++)
      {
        if(dat[i]>max_dat&&index[i]==0)
        {
          max_dat = dat[i];
          max_ind = i;
        }
      }
      num_connected_component = bfs(ind,max_ind,nx,ny,nz,dat,index);
      if(num_connected_component==0)
      {
        break;
      }
      if(num_connected_component>100)
      {
        std::cout << "ind:" << connected_component_ind << "   " << max_dat << "   " << num_connected_component << std::endl;
        connected_component_ind++;
        fault_map[connected_component_ind] = 
          new fault_details(connected_component_ind,num_connected_component,max_dat);
      }
      num_explored += num_connected_component;
    }
    for(int i=0;i<nx*ny*nz;i++)
    {
      if(index[i]==0)
      {
        dat[i] = 0;
      }
    }
    delete [] index;
    return num_explored;
  }

};

#endif

