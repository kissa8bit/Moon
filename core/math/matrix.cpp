#include "matrix.h"

namespace moon::math {

#define MAT_TMEP(n) template class Matrix<float, n, n>; extern template class Matrix<double, n, n>;

MAT_TMEP(2)
MAT_TMEP(3)
MAT_TMEP(4)
#undef MAT_TMEP

}
