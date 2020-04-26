#ifndef UTIL_H_
#define UTIL_H_

#ifndef OMP
#define OMP 0
#endif

#if OMP
#include <omp.h>
#else
#include "fake_omp.h"
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#endif