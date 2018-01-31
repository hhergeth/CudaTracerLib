#pragma once

#include <string>
#include <vector>
#include <map>
#include <boost/optional.hpp>

namespace CudaTracerLib {

class DynamicScene;
class Vec2i;

void ParseMitsubaScene(DynamicScene& scene, const std::string& scene_file, const std::map<std::string, std::string>& cmd_def_storage, boost::optional<Vec2i>& image_res, bool assume_rotated_coords, bool create_exterior_bssrdf, bool create_interior_bssrdf);

}