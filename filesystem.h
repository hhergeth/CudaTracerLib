#pragma once

// filesystem is still experimental in visual c++, remove this later on and replace by <filesystem> include
#include <experimental/filesystem>
namespace std {
namespace filesystem {
using namespace experimental::filesystem;
}  // namespace filesystem
}  // namespace std;