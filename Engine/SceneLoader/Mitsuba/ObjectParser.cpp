#include <StdAfx.h>
#include "ObjectParser.h"
#include <filesystem.h>
#include <miniz/miniz.h>

namespace CudaTracerLib {

ShapeParser::ShapeParseResult ShapeParser::serialized(const XMLNode& node, ParserState& S)
{
	auto filename = S.map_asset_filepath(S.def_storage.prop_string(node, "filename"));
	int submesh_index = S.def_storage.prop_int(node, "shapeIndex");
	bool flipNormals = S.def_storage.prop_bool(node, "flipNormals", false);
	bool faceNormals = S.def_storage.prop_bool(node, "faceNormals", false);
	float maxSmoothAngle = S.def_storage.prop_float(node, "maxSmoothAngle", 0.0f);

	auto name = std::filesystem::path(filename).stem().string();
	auto compiled_tar_folder = S.scene.getFileManager()->getCompiledMeshPath("") + name + "/";

	auto get_compiled_submesh_filename = [&](size_t i)
	{
		return compiled_tar_folder + std::to_string(i) + ".xmsh";
	};

	if (!std::filesystem::exists(compiled_tar_folder) || !std::filesystem::exists(get_compiled_submesh_filename(0)))
	{
		std::filesystem::create_directory(compiled_tar_folder);

		enum DataPresentFlag : uint32_t
		{
			VertexNormals = 0x0001,
			TextureCoords = 0x0002,
			VertexColors = 0x0008,
			UseFaceNormals = 0x0010,
			SinglePrecision = 0x1000,
			DoublePrecision = 0x2000,
		};

		struct inflateStream
		{
			std::ifstream& m_childStream;
			size_t str_length;
			z_stream m_inflateStream;
			uint8_t m_inflateBuffer[32768];
			inflateStream(std::ifstream& str)
				:m_childStream(str)
			{
				size_t pos = m_childStream.tellg();
				m_childStream.seekg(0, m_childStream.end);
				str_length = m_childStream.tellg();
				m_childStream.seekg(pos, m_childStream.beg);

				m_inflateStream.zalloc = Z_NULL;
				m_inflateStream.zfree = Z_NULL;
				m_inflateStream.opaque = Z_NULL;
				m_inflateStream.avail_in = 0;
				m_inflateStream.next_in = Z_NULL;

				int windowBits = 15;
				auto retval = inflateInit2(&m_inflateStream, windowBits);
				if (retval != Z_OK)
					std::cout << "erro, ret : " << retval << std::endl;
			}

			void read(void *ptr, size_t size)
			{
				uint8_t *targetPtr = (uint8_t *)ptr;
				while (size > 0) {
					if (m_inflateStream.avail_in == 0) {
						size_t remaining = str_length - m_childStream.tellg();
						m_inflateStream.next_in = m_inflateBuffer;
						m_inflateStream.avail_in = (uInt)std::min(remaining, sizeof(m_inflateBuffer));
						if (m_inflateStream.avail_in == 0)
							std::cout << "more bytes req : " << size << std::endl;
						m_childStream.read((char*)m_inflateBuffer, m_inflateStream.avail_in);
					}

					m_inflateStream.avail_out = (uInt)size;
					m_inflateStream.next_out = targetPtr;

					int retval = inflate(&m_inflateStream, Z_NO_FLUSH);
					switch (retval) {
					case Z_STREAM_ERROR:
						throw std::runtime_error("inflate(): stream error!");
					case Z_NEED_DICT:
						throw std::runtime_error("inflate(): need dictionary!");
					case Z_DATA_ERROR:
						throw std::runtime_error("inflate(): data error!");
					case Z_MEM_ERROR:
						throw std::runtime_error("inflate(): memory error!");
					};

					size_t outputSize = size - (size_t)m_inflateStream.avail_out;
					targetPtr += outputSize;
					size -= outputSize;

					if (size > 0 && retval == Z_STREAM_END)
						throw std::runtime_error("inflate(): attempting to read past the end of the stream!");
				}
			}
		};

		std::ifstream ser_str(filename, std::ios::binary);

		uint16_t magic_maj, version_maj;
		ser_str.read((char*)&magic_maj, 2);
		if (magic_maj != 1052)
			throw std::runtime_error("corrupt file");
		ser_str.read((char*)&version_maj, 2);

		ser_str.seekg(-4, ser_str.end);
		uint32_t n_meshes;
		ser_str.read((char*)&n_meshes, sizeof(n_meshes));
		ser_str.seekg((sizeof(uint32_t) + (version_maj == 4 ? sizeof(uint64_t) : sizeof(uint32_t)) * n_meshes), ser_str.end);
		std::vector<uint64_t> mesh_offsets(n_meshes);
		if (version_maj == 4)
			ser_str.read((char*)mesh_offsets.data(), n_meshes * sizeof(uint64_t));
		else
		{
			auto q = std::vector<uint32_t>(n_meshes);
			ser_str.read((char*)q.data(), n_meshes * sizeof(uint32_t));
			for (size_t i = 0; i < n_meshes; i++)
				mesh_offsets[i] = q[i];
		}

		for (size_t num_submesh = 0; num_submesh < n_meshes; num_submesh++)
		{
			ser_str.seekg(mesh_offsets[num_submesh], ser_str.beg);
			uint16_t magic, version;
			ser_str.read((char*)&magic, 2);
			if (magic == 0)
				break;
			ser_str.read((char*)&version, 2);
			if (version != 3 && version != 4)
				throw std::runtime_error("invalid version in serialized mesh file");

			inflateStream comp_str(ser_str);
			DataPresentFlag flag;
			comp_str.read(&flag, sizeof(flag));
			std::string name = "default";
			if (version == 4)
			{
				name = "";
				char last_read;
				do
				{
					comp_str.read(&last_read, sizeof(last_read));
					name += last_read;
				} while (last_read != 0);
			}
			uint64_t nVertices, nTriangles;
			comp_str.read(&nVertices, sizeof(nVertices));
			comp_str.read(&nTriangles, sizeof(nTriangles));

			std::vector<Vec3f> positions(nVertices), normals(nVertices), colors(nVertices);
			std::vector<Vec2f> uvcoords(nVertices);
			std::vector<uint32_t> indices(nTriangles * 3);

			bool isSingle = true;

			auto read_n_vector = [&](int dim, float* buffer)
			{
				if (isSingle)
					comp_str.read((char*)buffer, sizeof(float) * dim * nVertices);
				else
				{
					double* double_storage = (double*)alloca(dim * sizeof(double));
					for (size_t i = 0; i < nVertices; i++)
					{
						comp_str.read((char*)double_storage, dim * sizeof(double));
						for (int j = 0; j < dim; j++)
							buffer[i * dim + j] = float(double_storage[j]);
					}
				}
			};

			read_n_vector(3, (float*)positions.data());
			if ((flag & DataPresentFlag::VertexNormals) == DataPresentFlag::VertexNormals)
				read_n_vector(3, (float*)normals.data());
			if ((flag & DataPresentFlag::TextureCoords) == DataPresentFlag::TextureCoords)
				read_n_vector(2, (float*)uvcoords.data());
			else std::fill(uvcoords.begin(), uvcoords.end(), Vec2f(0.0f));
			if ((flag & DataPresentFlag::VertexColors) == DataPresentFlag::VertexColors)
				read_n_vector(3, (float*)colors.data());

			comp_str.read((char*)indices.data(), sizeof(uint32_t) * nTriangles * 3);
			for (size_t i = 0; i < nTriangles * 3; i += 3)
				std::swap(indices[i + 0], indices[i + 2]);

			auto compiled_submesh_filename = get_compiled_submesh_filename(num_submesh);
			FileOutputStream fOut(compiled_submesh_filename);
			fOut << (unsigned int)MeshCompileType::Static;
			auto mat = Material(name.size() > 60 ? name.substr(0, 60) : name);
			mat.bsdf = CreateAggregate<BSDFALL>(diffuse());
			Mesh::CompileMesh(positions.data(), (int)positions.size(), normals.data(), uvcoords.data(), indices.data(), (int)indices.size(), mat, 0.0f, fOut, flipNormals, faceNormals, maxSmoothAngle);
			fOut.Close();
		}
		ser_str.close();
	}

	auto obj = S.scene.CreateNode(get_compiled_submesh_filename(submesh_index));
	parseGeneric(obj, node, S);
	return obj;
}

}
