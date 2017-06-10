#ifndef token_h
#define token_h

enum TYPE { FREQ_CPU, CPU, FREQ_GPU, GPU };

struct Token
{
	const std::string name;
	const TYPE type;
	const std::size_t nx, ny, nz;
	Token(
		std::string _name
		, TYPE _type
		, std::size_t _nx
		, std::size_t _ny
		, std::size_t _nz
		)
		: name(_name)
		, type(_type)
		, nx(_nx)
		, ny(_ny)
		, nz(_nz)
	{}
};

class TokenComparator 
{ 
	// simple comparison function
	public:
		bool operator()(const Token & a, const Token & b) { return a.name > b.name; } 
};

#endif
