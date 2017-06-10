#ifndef message_assert_h
#define message_assert_h

#define MESSAGE_ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
				        } \
		    } while (false)

#endif
