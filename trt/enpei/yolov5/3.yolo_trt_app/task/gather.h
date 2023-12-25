#ifndef TASK_GATHER_H_
#define TASK_GATHER_H_

#include <vector>

#include "types.h"

std::vector<std::vector<Point>> gather(const std::vector<Point>& points);

std::vector<std::vector<Point>> gather_rule(const std::vector<Point>& points, float threshold);

#endif