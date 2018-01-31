#pragma once
#define _SCL_SECURE_NO_WARNINGS

#include <map>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <Engine/DynamicScene.h>

namespace CudaTracerLib {

inline std::string str_tolower(const std::string& s)
{
    return boost::algorithm::to_lower_copy(s);
}

//https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
inline bool endsWith(const std::string& s, const std::string& suffix)
{
	return s.size() >= suffix.size() &&
		s.substr(s.size() - suffix.size()) == suffix;
}

inline std::vector<std::string> split_string(const std::string& s, const std::vector<std::string>& delimiters, const bool& removeEmptyEntries = false)
{
	std::vector<std::string> tokens;

	size_t smallest_delimiter = delimiters[0].length();
	for (auto& del : delimiters)
		smallest_delimiter = std::min(smallest_delimiter, del.length());

	for (size_t start = 0, end; start < s.length(); start = end + smallest_delimiter)
	{
		size_t position = size_t(-1);
		for (auto& del : delimiters)
			position = std::min(position, s.find(del, start));

		end = position != std::string::npos ? position : s.length();

		std::string token = s.substr(start, end - start);
		if (!removeEmptyEntries || !token.empty())
		{
			tokens.push_back(token);
		}
	}

	bool any_ends = false;
	for (auto& del : delimiters)
		any_ends |= endsWith(s, del);

	if (!removeEmptyEntries &&
		(s.empty() || any_ends))
	{
		tokens.push_back("");
	}

	return tokens;
}

inline std::vector<std::string> split_string_array(const std::string& data)
{
	return split_string(data, {", ", " "}, true);
}

class XMLNode
{
	public:
	std::string m_name;
	boost::property_tree::ptree m_pt;

	boost::optional<const boost::property_tree::ptree&> get_attribs() const
	{
		return m_pt.get_child_optional("<xmlattr>");
	}

public:
	XMLNode(const std::string& n, const boost::property_tree::ptree& pt)
		: m_name(n), m_pt(pt)
	{

	}

	std::string name() const
	{
		return str_tolower(m_name);
	}

	template<typename F> void iterate_child_nodes(F clb) const
	{
		for (auto& x : m_pt)
			if(x.first != "<xmlattr>" && x.first != "<xmlcomment>")
				clb(XMLNode(x.first, x.second));
	}

	XMLNode get_child_node(const std::string& name) const
	{
		return XMLNode(name, m_pt.get_child(str_tolower(name)));
	}

	bool has_child_node(const std::string& name) const
	{
		return (bool)m_pt.get_child_optional(str_tolower(name));
	}

	template<typename F> void iterate_attributes(F clb) const
	{
		auto q = get_attribs();
		if(q)
			for (auto& x : q.get())
				clb(str_tolower(x.first), x.second.data());
	}

	std::string get_attribute(const std::string& name) const
	{
		auto q = get_attribs();
		if (q)
			return q.get().get_child(str_tolower(name)).get_value<std::string>();
		else throw std::runtime_error("no attributes in node!");
	}

	bool has_attribute(const std::string& name) const
	{
		auto q = get_attribs();
		if (q)
			return (bool)q.get().get_child_optional(str_tolower(name));
		else return false;
	}

	//helper functions specifically for mitsuba xmls
	bool has_property(const std::string& name) const
	{
		for (auto& x : m_pt)
		{
			auto n = XMLNode(x.first, x.second);
			if (n.has_attribute("name") && str_tolower(n.get_attribute("name")) == str_tolower(name))
				return true;
		}
		return false;
	}

	XMLNode get_property(const std::string& name) const
	{
		for (auto& x : m_pt)
		{
			auto n = XMLNode(x.first, x.second);
			if (n.has_attribute("name") && str_tolower(n.get_attribute("name")) == str_tolower(name))
				return n;
		}
		throw std::runtime_error("no property of that name : " + name);
	}
};

class DefaultValueStorage
{
	std::map<std::string, std::string> m_values;
public:
	DefaultValueStorage()
	{

	}

	void add(const std::string key, const std::string& val)
	{
		m_values[key] = val;
	}

	std::string map(const std::string& data) const
	{
		size_t start_key_idx = data.find_first_of('$', 0);
		if (start_key_idx != size_t(-1)) {
			size_t end_key_idx = data.find_first_of(' ', start_key_idx);
			if (end_key_idx == size_t (-1))
				end_key_idx = data.size();
			auto key = data.substr(start_key_idx + 1, end_key_idx - start_key_idx - 1);
			auto val_it = m_values.find(key);
			if (val_it == m_values.end())
				throw std::runtime_error("invalid default value");
			auto val = val_it->second;
			auto data_repl = (start_key_idx != 0 ? data.substr(0, start_key_idx + 1) : "") + val + (end_key_idx != data.size() ? data.substr(end_key_idx) : "");
			return map(data_repl);
		}
		else return data;
	}

	float as_float(const std::string& data) const
	{
		auto val = map(data);
		return std::stof(val);
	}

	int as_int(const std::string& data) const
	{
		auto val = map(data);
		return std::stoi(val);
	}

	std::string as_string(const std::string& data) const
	{
		auto val = map(data);
		return val;
	}

	bool as_bool(const std::string& data) const
	{
		auto val = map(data);
		return val == "True";
	}

	float as_float(const XMLNode& node, const std::string& attrib_name, boost::optional<float> def_val = boost::none) const
	{
		if (!node.has_attribute(attrib_name) && def_val)
			return def_val.get();
		else return as_float(node.get_attribute(attrib_name));
	}

	int as_int(const XMLNode& node, const std::string& attrib_name, boost::optional<int> def_val = boost::none) const
	{
		if (!node.has_attribute(attrib_name) && def_val)
			return def_val.get();
		else return as_int(node.get_attribute(attrib_name));
	}

	std::string as_string(const XMLNode& node, const std::string& attrib_name, boost::optional<std::string> def_val = boost::none) const
	{
		if (!node.has_attribute(attrib_name) && def_val)
			return def_val.get();
		else return as_string(node.get_attribute(attrib_name));
	}

	bool as_bool(const XMLNode& node, const std::string& attrib_name, boost::optional<bool> def_val = boost::none) const
	{
		if (!node.has_attribute(attrib_name) && def_val)
			return def_val.get();
		else return as_bool(node.get_attribute(attrib_name));
	}

	float prop_float(const XMLNode& node, const std::string& prop_name, boost::optional<float> def_val = boost::none, const std::string& attrib_name = "value") const
	{
		if (node.has_property(prop_name))
			return as_float(node.get_property(prop_name), attrib_name, def_val);
		else
		{
			if (!def_val)
				throw std::runtime_error("no default value passed but property doesn't exist!");
			return def_val.get();
		}
	}

	int prop_int(const XMLNode& node, const std::string& prop_name, boost::optional<int> def_val = boost::none, const std::string& attrib_name = "value") const
	{
		if (node.has_property(prop_name))
			return as_int(node.get_property(prop_name), attrib_name, def_val);
		else
		{
			if (!def_val)
				throw std::runtime_error("no default value passed but property doesn't exist!");
			return def_val.get();
		}
	}

	std::string prop_string(const XMLNode& node, const std::string& prop_name, boost::optional<std::string> def_val = boost::none, const std::string& attrib_name = "value") const
	{
		if (node.has_property(prop_name))
			return as_string(node.get_property(prop_name), attrib_name, def_val);
		else
		{
			if (!def_val)
				throw std::runtime_error("no default value passed but property doesn't exist!");
			return def_val.get();
		}
	}

	bool prop_bool(const XMLNode& node, const std::string& prop_name, boost::optional<bool> def_val = boost::none, const std::string& attrib_name = "value") const
	{
		if (node.has_property(prop_name))
			return as_bool(node.get_property(prop_name), attrib_name, def_val);
		else
		{
			if (!def_val)
				throw std::runtime_error("no default value passed but property doesn't exist!");
			return def_val.get();
		}
	}
};

class IoRLibrary
{
	std::map<std::string, float> entries;
public:
	IoRLibrary()
	{
		entries = {
			{ "vacuum", 1.0f },
			{ "helium", 1.00004f },
			{ "hydrogen", 1.00013f },
			{ "air", 1.00028f },
			{ "carbon dioxide", 1.00045f },

			{ "water", 1.3330f },
			{ "acetone", 1.36f },
			{ "ethanol", 1.361f },
			{ "carbon tetrachloride", 1.451f },
			{ "glycerol", 1.4729f },
			{ "benzene", 1.501f },
			{ "silicone oil", 1.52045f },

			{ "bromine", 1.661f },
			{ "water ice", 1.31f },
			{ "fused quartz", 1.458f },

			{ "pyrex", 1.470f },
			{ "acrylic glass ", 1.49f },
			{ "polypropylene", 1.49f },
			{ "bk7", 1.5046f },
			{ "sodium chloride", 1.544f },
			{ "amber", 1.55f },
			{ "pet", 1.575f },
			{ "diamond", 2.419f },
		};
	}

	float get(const XMLNode& node, const DefaultValueStorage& def_storage, const boost::optional<std::string>& def_name = boost::none) const
	{
		auto xml_val = node.get_attribute("value");
		xml_val = def_storage.as_string(xml_val);
		try
		{
			return boost::lexical_cast<float>(xml_val);
		}
		catch (...)
		{
			auto ent = entries.find(xml_val);
			if (ent == entries.end() && !def_name)
				throw std::runtime_error("Invalid IOR material name");
			else if (ent == entries.end())
				return get(def_name.get());
			else return ent->second;
		}
	}

	float get(const std::string& name) const
	{
		auto ent = entries.find(name);
		if (ent == entries.end())
			throw std::runtime_error("Invalid IOR material name");
		else return ent->second;
	}
};

class RefStorage
{
	std::map<std::string, const void*> m_ref_objects;
public:
	RefStorage()
	{

	}

	template<typename T> void add(const std::string& key, const T* ptr)
	{
		m_ref_objects[key] = ptr;
	}

	bool is_ref(const XMLNode& node)
	{
		return node.name() == "ref";
	}

	bool has_ref_id(const XMLNode& node)
	{
		return node.has_attribute("id");
	}

	template<typename T> void add_if_ref(const XMLNode& node, const T* ptr)
	{
		if(has_ref_id(node))
			add(node.get_attribute("id"), ptr);
	}

	template<typename T> T* get(const std::string& name)
	{
		return (T*)m_ref_objects[name];
	}

	template<typename T> T* get(const XMLNode& node)
	{
		return get<T>(node.get_attribute("id"));
	}

	template<typename T> T add_helper(const XMLNode& node, T val)
	{
		add_if_ref(node, new T(val));
		return val;
	}
};

class ParserState
{
public:
	DefaultValueStorage def_storage;
	RefStorage ref_storage;
	IoRLibrary ior_lib;
	DynamicScene& scene;
	std::string scenefile_location;
	boost::optional<Vec2i> film_size;
	float4x4 id_matrix;
	bool create_interior_bssrdf;
	bool create_exterior_bssrdf;
	std::vector<StreamReference<Node>> nodes_to_remove;

	ParserState(DynamicScene& scene, const std::string& mitsuba_scene_xml_loc, bool assume_rotated_coords, bool create_exterior_bssrdf, bool create_interior_bssrdf)
		: scene(scene), scenefile_location(mitsuba_scene_xml_loc), film_size(boost::none), create_exterior_bssrdf(create_exterior_bssrdf), create_interior_bssrdf(create_interior_bssrdf)
	{
		id_matrix = assume_rotated_coords ? float4x4::RotateX(PI) : float4x4::Identity();
	}

	std::string map_spd_filepath(const std::string& spd_name) const
	{
		return scene.getFileManager()->getDataPath() + "/Data/" + spd_name;
	}

	//used for textures and meshes
	std::string map_asset_filepath(const std::string& asset) const
	{
		return scenefile_location + "/" + asset;
	}

	void set_film_resolution(int x, int y)
	{
		film_size = Vec2i(x,y);
	}

	std::string get_scene_name() const
	{
		return boost::filesystem::path(scenefile_location).stem().string();
	}
};


}