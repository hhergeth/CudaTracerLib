#include <StdAfx.h>
#include "MitsubaLoader.h"

#include "Utils.h"
#include <iostream>
#include <boost/property_tree/xml_parser.hpp>
#include "PropertyParser.h"
#include "ObjectParser.h"

namespace CudaTracerLib {

void ParseMitsubaScene(DynamicScene& scene, const std::string& scene_file, const std::map<std::string, std::string>& cmd_def_storage, boost::optional<Vec2i>& image_res, bool assume_rotated_coords, bool create_exterior_bssrdf, bool create_interior_bssrdf)
{
	ParserState S(scene, boost::filesystem::path(scene_file).parent_path().string(), assume_rotated_coords, create_exterior_bssrdf, create_interior_bssrdf);
	for (auto ent : cmd_def_storage)
		S.def_storage.add(ent.first, ent.second);

	boost::property_tree::ptree pt;
	read_xml(scene_file, pt);
	CudaTracerLib::XMLNode root("", pt);

	root.get_child_node("scene").iterate_child_nodes([&](const CudaTracerLib::XMLNode& n)
	{
		if (n.name() == "include")
		{
			auto filename = S.map_asset_filepath(S.def_storage.as_string(n, "filename"));
			ParseMitsubaScene(scene, filename, cmd_def_storage, image_res, assume_rotated_coords, create_exterior_bssrdf, create_interior_bssrdf);
		}
		else if (n.name() == "default")
		{
			auto name = n.get_attribute("name"), value = n.get_attribute("value");
			S.def_storage.add(name, value);
		}
		else if (n.name() == "alias")
		{
			auto id = n.get_attribute("id"), as = n.get_attribute("as");
			S.ref_storage.add(as, S.ref_storage.get<void>(id));
		}
		else if (n.name() == "sensor")
		{
			auto C = SensorParser::parse(n, S);
			scene.setCamera(new Sensor(C));
		}
		else if (n.name() == "emitter")
		{
			LightParser::parse(n, S);
		}
		else if (n.name() == "bsdf")
		{
			BsdfParser::parse(n, S, 0);
		}
		else if (n.name() == "shape")
		{
			ShapeParser::parse(n, S);
		}
		else if (n.name() == "texture")
		{
			TextureParser::parse(n, S);
		}
		else if (n.name() == "medium")
		{
			MediumParser::parse(n, S);
		}
		//else throw std::runtime_error("Unrecognized token : " + n.name());
	});
	scene.UpdateScene();

	for (auto obj : S.nodes_to_remove)
		scene.DeleteNode(obj);
	scene.UpdateScene();

	image_res = S.film_size;
}

}