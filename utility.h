#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <vector>

using namespace std;

static void stringtok(vector<string> &v, string const &in,
               const char * const delim = " \t\n",
               bool includeDelim = false)
{
  const string::size_type len = in.length();
  string::size_type i;
  string::size_type j = 0;
  unsigned int k;

  while ( j < len )
  {
    // find the start of the token
    i = in.find_first_not_of(delim, j);
    
    if (i == string::npos) { // nothing left but delim
      if (includeDelim) {
        string s = in.substr(j);
        for (k = 0; k < s.length(); k++)
          v.push_back(s.substr(k,1));
      }
      return;
    } else
      if (includeDelim)
        if (i-j!=0) {
          string s = in.substr(j, i-j);
          for (k = 0; k < s.length(); k++)
            v.push_back(s.substr(k,1));
        }

    // find the end of the token
    j = in.find_first_of(delim, i);

    // push token
    if (j == string::npos) { // nothing left but a token
      v.push_back(in.substr(i));
      return;
    } else
      v.push_back(in.substr(i, j-i));
  }
}

#endif


