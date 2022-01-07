#include "vector.h"

namespace moon::math {

#define VEC_TMEP(n) extern template class Vector<float, n>; extern template class Vector<double, n>;

VEC_TMEP(2)
VEC_TMEP(3)
VEC_TMEP(4)
#undef VEC_EXT_TMEP

}
