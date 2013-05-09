#include "stdafx.h"
#include "SkyrimImporter.h"
#include <vector>
#include <zlib.h>
#include "Base\Timer.h"
#include "Base\FrameworkInterop.h"
#include <iostream>
#include <algorithm>
#include <string>
#include "Engine\e_Terrain.h"

struct TYPE4
{
	char data[4];
	TYPE4(){}
	TYPE4(char* a)
	{
		for(int i = 0; i < 4; i++)
			data[i] = a[i];
	}
	friend bool operator== (const TYPE4 &cP1, const TYPE4 &cP2);
	friend bool operator== (const TYPE4 &cP1, const char* cP2);
};
bool operator== (const TYPE4 &cP1, const TYPE4 &cP2)
{
	for(int i = 0; i < 4; i++)
		if(cP1.data[i] != cP2.data[i])
			return false;
	return true;
}
bool operator== (const TYPE4 &cP1, const char* cP2)
{
	for(int i = 0; i < 4; i++)
		if(cP1.data[i] != cP2[i])
			return false;
	return true;
}
typedef DWORD formid;
typedef COLORREF rgb;
#define IS(word) (type.data[0] == word[0] && type.data[1] == word[1] && type.data[2] == word[2] && type.data[3] == word[3])
#define RE(WORD, TYPE) if(IS(WORD)) return new TYPE(S, type);
#define RET(WORD) RE(#WORD, WORD)
unsigned count_arguments(char *s){
	unsigned i,argc = 1;
		for(i = 0; s[i]; i++)
			if(s[i] == ',')
				argc++;
	return argc;
}
#define GET(...) getA(count_arguments(#__VA_ARGS__), __VA_ARGS__)

class TES_FileStream;
class IMember;
class Field;
class Record;
IMember* CreateField(TES_FileStream& S, TYPE4 record, Field* lastField);
IMember* CreateRecord(TES_FileStream& S);

class TES_FileStream : public IInStream
{
private:
	char* data;
	unsigned long long pos, size;
public:
	TES_FileStream(char* name)
	{
		InputStream A(name);
		data = A.ReadToEnd<char>();
		pos = 0;
		size = A.getFileSize();
		A.Close();
	}
	TES_FileStream(char* _data, unsigned long long _size)
	{
		data = _data;
		size = _size;
		pos = 0;
	}
	template<typename T> void Read(T* a_Data, unsigned int a_Size)
	{
		memcpy(a_Data, data + pos, a_Size);
		pos += a_Size;
	}
	template<typename T> void Read(const T& a_Data)
	{
		memcpy(&a_Data, data + pos, sizeof(T));
		pos += sizeof(T);
	}
	virtual void Read(void* a_Out, unsigned int a_Size)
	{
		Read<char>((char*)a_Out, a_Size);
	}
	virtual unsigned long long getPos()
	{
		return pos;
	}
	unsigned long long getFileSize()
	{
		return size;
	}
	void setPos(unsigned long long p)
	{
		pos = p;
	}
	void movePos(unsigned long long o)
	{
		pos += o;
	}

	char* ReadZString()
	{
		int l0 = 0;
		char* c = 0;
		char q = 0;
		int i = 0;
		do
		{
			if(i == l0)
			{
				char* o = c;
				int ol = i;
				l0 = MAX(32, i * 2);
				c = new char[l0];
				if(o)
				{
					memcpy(c, o, ol);
					delete [] o;
				}
			}
			*this >> q;
			c[i++] = q;
		}
		while(q);
		return c;
	}
	char* ReadBString()
	{
		unsigned char l;
		*this >> l;
		char* q = new char[l];
		Read(q, l);
		return q;
	}
	char* ReadBZString()
	{
		unsigned char l;
		*this >> l;
		l++;
		char* q = new char[l];
		Read(q, l);
		return q;
	}
	char* ReadWString()
	{
		unsigned short l;
		*this >> l;
		char* q = new char[l];
		Read(q, l);
		return q;
	}
	char* ReadWZString()
	{
		unsigned short l;
		*this >> l;
		l++;
		char* q = new char[l];
		Read(q, l);
		return q;
	}
	DWORD ReadVSVal()
	{
		struct data
		{
			unsigned char c0, c1;
			unsigned short high;
			DWORD res()
			{
				return *(DWORD*)this << 2;
			}
		};
		data d;
		d.high = d.c1 = d.c0 = 0;
		*this >> d.c0;
		if(d.c0 & 3 >= 1)
			*this >> d.c1;
		else if(d.c0 & 3 > 1)
			*this >> d.high;
		return d.res();
	}
};

class IMember
{
protected:
	bool m_bHasDeserialized;
	TES_FileStream* m_pStream;
	unsigned long long m_uPos;
	std::vector<IMember*> Members;
public:
	TYPE4 type;
	DWORD size;
public:
	void Access()
	{
		if(m_bHasDeserialized)
			return;
		m_bHasDeserialized = true;
		m_pStream->setPos(m_uPos);
		DeSerialize(*m_pStream);
	}
	void AccessAll()
	{
		Access();
		for(int i = 0; i < Members.size(); i++)
			Members[i]->AccessAll();
	}
	template<typename T> T* get(const TYPE4 name, const unsigned long long off = 0)
	{
		Access();
		unsigned long long q = 0;
		for(int i = 0; i < Members.size(); i++)
			if(Members[i]->IsOfType(name))
			{
				if(q == off)
					return (T*)Members[i];
				q++;
			}
		return 0;
	}
	IMember* operator[](const unsigned int i)
	{
		Access();
		return Members[i];
	}
	unsigned int Count()
	{
		Access();
		return Members.size();
	}
	IMember* getA(int num, ...)
	{
		va_list values;
		va_start ( values, num ) ;
		IMember* a = this;
		for(int i = 0; i < num; i++)
		{
			int q = va_arg ( values, int );
			a = a[0][q];
		}
		va_end ( values ) ;
		return a;
	}
protected:
	void Init(TES_FileStream* s, DWORD _size)
	{
		m_pStream = s;
		m_uPos = s->getPos();
		m_bHasDeserialized = false;
		size = _size;
		s->movePos(size);
	}
	virtual void DeSerialize(TES_FileStream& S) = 0;
	virtual bool IsOfType(TYPE4 t)
	{
		return t == type;
	}
};

class Field : public IMember
{
public:
	WORD size;
public:
	Field(TES_FileStream& S, TYPE4 _type)
	{
		type = _type;
		S >> size;
	}
protected:
	virtual void DeSerialize(TES_FileStream& S)
	{
	}
};

class BaseField : public Field
{
public:
	char* data;
public:
	BaseField(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		data = new char[size];
		S.Read(data, size);
	}
	BaseField(TES_FileStream& S, TYPE4 _type, unsigned int size)
		: Field(S, _type)
	{
		this->size = size;
		data = new char[size];
		S.Read(data, size);
	}
};

class Record : public IMember
{
public:
	DWORD flags;
	DWORD id;
	DWORD rev;
	WORD version;
	WORD unk;
public:
	Record(TES_FileStream& S, TYPE4 _type)
	{
		type = _type;
		DWORD s;
		S >> s;
		S >> flags;
		S >> id;
		S >> rev;
		S >> version;
		S >> unk;
		Init(&S, s);
	}
	void Decrypt()
	{
		if((flags & 0x00040000) == 0)
			return;
		m_pStream->setPos(m_uPos);
		DWORD cmpSize = size;
		m_pStream->operator>>(size);
		char* cmpData = new char[cmpSize], *uncData = new char[size];
		m_pStream->Read(cmpData, cmpSize);

		z_stream infstream;
		infstream.zalloc = Z_NULL;
		infstream.zfree = Z_NULL;
		infstream.opaque = Z_NULL;
		infstream.avail_in = 0;
		infstream.next_in = Z_NULL;
		inflateInit(&infstream);

		infstream.zfree = Z_NULL;
		infstream.opaque = Z_NULL;
		infstream.avail_in = cmpSize; // size of input
		infstream.next_in = (Bytef *)cmpData; // input char array
		infstream.avail_out = size; // size of output
		infstream.next_out = (Bytef *)uncData; // output char array

		inflate(&infstream, Z_NO_FLUSH);
		inflateEnd(&infstream);

		TES_FileStream* S2 = new TES_FileStream(uncData, size);
		unsigned long long end = S2->getPos() + size;
			do
			{
				if(S2->getPos() > end)
					throw 1;
				Members.push_back(CreateField(*S2, type, Members.size() > 0 ? (Field*)Members[Members.size() - 1] : 0));
			}
			while(S2->getPos() != end);
	}
protected:
	virtual void DeSerialize(TES_FileStream& S)
	{
		if(flags & 0x00040000)
		{
			S.movePos(size);
		}
		else
		{
			unsigned long long end = S.getPos() + size;
			do
			{
				if(S.getPos() > end)
					throw 1;
				Members.push_back(CreateField(S, type, Members.size() > 0 ? (Field*)Members[Members.size() - 1] : 0));
			}
			while(S.getPos() != end);
		}
	}
};

class Group : public IMember
{
public: 
	DWORD label;
	DWORD groupType;
	WORD stamp;
	WORD unk;
	WORD ver;
	WORD unk2;
	TYPE4 memberTypes;
public:
	Group(TES_FileStream& S, TYPE4 _type)
	{
		type = _type;
		DWORD s;
		S >> s;
		S >> label;
		S >> groupType;
		S >> stamp;
		S >> unk;
		S >> ver;
		S >> unk2;
		Init(&S, s - 24);
	}
	template<typename T> T* get(const DWORD id)
	{
		for(int i = 0; i < Members.size(); i++)
			if(dynamic_cast<Record*>(Members[i]) && dynamic_cast<Record*>(Members[i])->id == id)
				return dynamic_cast<Record*>(Members[i]);
		return 0;
	}
protected:
	virtual void DeSerialize(TES_FileStream& S)
	{
		unsigned long long end = S.getPos() + size;
		if(size)
		{
			do
			{
				Members.push_back(CreateRecord(S));
			}
			while(S.getPos() != end);
			memberTypes = Members[0]->type;
		}
	}
	virtual bool IsOfType(TYPE4 t)
	{
		return t == memberTypes;
	}
};

class CNAM : public Field
{
public:
	char* author;
public:
	CNAM(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		author = S.ReadZString();
	}
};

class SNAM : public Field
{
public:
	char* desc;
public:
	SNAM(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		desc = S.ReadZString();
	}
};

class MAST : public Field
{
public:
	char* master;
public:
	MAST(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		master = S.ReadZString();
	}
};

class EDID : public Field
{
public:
	char* editorID;
public:
	EDID(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		editorID = S.ReadZString();
	}
};

class OBND : public Field
{
public:
	short3 vmax, vmin;
public:
	OBND(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		S.Read((char*)&vmax, 12);
	}
	AABB box()
	{
		return AABB(make_float3(vmin.x, vmin.y, vmin.z), make_float3(vmax.x, vmax.y, vmax.z));
	}
};

class NAME : public Field
{
public:
	formid refid;
public:
	NAME(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		S >> refid;
	}
};

class MODL : public Field
{
public:
	char* model;
public:
	MODL(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		model = S.ReadZString();
	}
};

static float3 g_Offset = make_float3(0);
static float4x4 transMat = float4x4::RotateX(-PI/2);
class DATA : public Field
{
public:
	float3 pos;
	float3 rot;
public:
	DATA(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		S >> pos;
		S >> rot;
	}
	float4x4 getMat(float f = 1)
	{
		float4x4 r = float4x4::RotateZ(-rot.z) * float4x4::RotateY(-rot.y) * float4x4::RotateX(-rot.x);
		float4x4 q = float4x4::Scale(f) * r * float4x4::Translate(make_float3(pos.x, pos.y, pos.z));
		return q * transMat;
	}
};

class LIGH_DATA : public Field
{
public:
	int time;
	uint radius;
	uchar4 rgb;
	uint flags;
	float fallof;
	float fov;
	float nearv;
	float period;
	float intensity;
	float movement;
	uint value;
	float weight;
	LIGH_DATA(TES_FileStream& S, TYPE4 _type)
		: Field(S, _type)
	{
		S.Read(&time, 48);
	}
};

IMember* CreateField(TES_FileStream& S, TYPE4 record, Field* lastField)
{
	TYPE4 type;
	S.operator>>(*(int*)&type);
	if(record == "STAT" || record == "REFR")
	{
		RET(EDID)
		RET(OBND)
		RET(MODL)
		RET(NAME)
	}
	if(record == "REFR")
	{
		RET(DATA)
	}
	if(record == "LIGH")
	{
		RE("DATA", LIGH_DATA);
	}
	if(lastField && lastField->type == "XXXX")
		return new BaseField(S, type, *(int*)((BaseField*)lastField)->data);
	return new BaseField(S, type);
}

IMember* CreateRecord(TES_FileStream& S)
{
	TYPE4 type;
	S.operator>>(*(int*)&type);
	RE("GRUP", Group)
	return new Record(S, type);
}

std::vector<Group*> importESM(char* C)
{
	//F:\\Spiele\\Steam\\SteamApps\\common\\skyrim\\Data
	TES_FileStream* I = new TES_FileStream(C);
	//TES_FileStream I("Moria.esp");
	IMember* root = CreateRecord(*I);
	std::vector<Group*> groups;
	while(I->getPos() != I->getFileSize())
		groups.push_back((Group*)CreateRecord(*I));
	//cTimer T, T2;
	//T2.StartTimer();
	for(int i = 0; i < groups.size(); i++)
	{
		//T.StartTimer();
		groups[i]->Access();
		if(	   !(groups[i]->memberTypes == "GRUP") && !(groups[i]->memberTypes == "WRLD")
			&& !(groups[i]->memberTypes == "RACE") && !(groups[i]->memberTypes == "DIAL")
			&& !(groups[i]->memberTypes == "PACK") && !(groups[i]->memberTypes == "SCEN")
			&& !(groups[i]->memberTypes == "DLBR") && !(groups[i]->memberTypes == "QUST")
			&& !(groups[i]->memberTypes == "NAVI") && !(groups[i]->memberTypes == "FACT")
		  )
			groups[i]->AccessAll();
		/*
		else OutputDebugString("GRUP FOUND\n");
		double sec = T.EndTimer();
		char mes[1024];
		TYPE4 q = groups[i]->memberTypes;
		sprintf(mes, "Decoding %c%c%c%c took %f sec.\n", q.data[0], q.data[1], q.data[2], q.data[3], (float)sec);
		OutputDebugString(mes);*/
	}/*
	double sec = T2.EndTimer();
	char mes[1024];
	sprintf(mes, "Decoding took %f sec.\n", (float)sec);
	OutputDebugString(mes);*/
	return groups;
}

struct Hasher
{
	std::vector<Group*> groups;
	Hasher& operator()(Group* g)
	{
		groups.push_back(g);
		return *this;
	}
	Hasher(std::vector<Group*>& _groups)
		: groups(_groups)
	{
		
	}
	Hasher(){}
	Hasher& operator()(std::vector<Group*>& _groups, TYPE4 name)
	{
		for(int i = 0; i < _groups.size(); i++)
			if(_groups[i]->memberTypes == name)
			{
				_groups[i]->AccessAll();
				groups.push_back(_groups[i]);
				break;
			}
		return *this;
	}
	template<typename T> T* Find(DWORD id)
	{
		for(int i = 0; i < groups.size(); i++)
		{
			T* a = groups[i]->get<T>(id);
			if(a)
				return a;
		}
		return 0;
	}
};

struct vhData
{        
    float off;
	char data[33 * 33];
        
    template<typename T, bool useOff> void ComputeGrid(T* A)
    {
		float ver = off, hor = 0;
		for(int row = 0; row < 32; row++)
		{
			ver += data[row * 33];
			float p = ver + hor;
			for(int col = 1; col < 33; col++)
			{
				p += data[row * 33 + col];
				A[(row) * 32 + (col - 1)] = (T)(p * 8.0f) + g_Offset.y;
			}
		}
    }

	e_TerrainDataTransporter<32U>* getTransporter(char3* nrmls, int2 off)
	{
		e_TerrainDataTransporter<32>* r = new e_TerrainDataTransporter<32>(off);
		struct funct
		{
			float A[32 * 32];
			float3 D[32 * 32];
			float4 operator()(int x, int y)
			{
				x = clamp(x, 0, 32);
				y = clamp(y, 0, 32);
				return make_float4(D[y * 32 + x], A[y * 32 + x]);
			}
		};
		funct a;
		for(int row = 0; row < 32; row++)
			for(int col = 0; col < 32; col++)
			{
				char3* q = nrmls + row * 33 + col;
				float3 n = -transMat.TransformNormal(make_float3(q->x, q->y, q->z) / 127.0f);
				n.y *= -1.0f;
				a.D[row * 32 + col] = n;
			}
		ComputeGrid<float, true>(a.A);
		e_TerrainDataTransporter<32>::Create(a, r);
		return r;
	}

	void printf()
	{
		//float f[32 * 32];
		char msg[323];
		//ComputeGrid<float, false>(f);
		char* q = data;
		OutputDebugString("\n");
		for(int i = 0; i < 33; i++)
		{
			msg[0] = 0;
			for(int j = 0; j < 33; j++)
			{
				sprintf(msg, "%s%6d", msg, (int)q[i * 33 + j]);
			}
			OutputDebugString(msg);
			OutputDebugString("\n");
		}
	}
};

template<int L = 128, int W = 32> struct transporterCollection
{
	e_TerrainDataTransporter<W>* m_T[L][L];
	transporterCollection()
	{
		ZeroMemory(m_T, sizeof(m_T));
	}
	void operator()(e_TerrainDataTransporter<W>* T)
	{
		int x = T->m_sOffset.x + L / 2, y = T->m_sOffset.y + L / 2;
		m_T[x][y] = T;
	}
	float operator()(int x, int y)
	{
		int xo = x % W, yo = y % W;
		x /= W;
		y /= W;
		x += L / 2;
		y += L / 2;
		e_TerrainDataTransporter<W>* T = m_T[x][y];
		if(T)
			return T->operator()(xo, yo).w;
		else return 0;
	}
	float operator()(int xb, int yb, int xr, int yr)
	{
		xb += xr / W;
		yb += yr / W;
		xr = xr % W;
		yr = yr % W;
		e_TerrainDataTransporter<W>* T = m_T[xb][yb];
		if(T)
			return T->operator()(xr, yr).w;
		else return 0;
	}
	void IntegrateTransporters(e_Terrain* T)
	{
		int2 minloc, maxloc;
		getMinMaxLoc(minloc, maxloc);
		for(int yg = 0; yg < L; yg++)
			for(int xg = 0; xg < L; xg++)
				if(m_T[xg][yg])
				{
					int xb = (xg - L / 2) * 32, yb = (yg - L / 2) * 32;
					for(int xr = 0; xr < 32; xr += 2)
						for(int yr = 0; yr < 32; yr += 2)
						{
							int xa = xb + xr - minloc.x * 32, ya = yb + yr - minloc.y * 32;
							e_TerrainData_Leaf* l = T->getValAt(xa, ya);
#define G(a,b) this[0](xg, yg, xr + a, yr + b)
							l->setData(G(0,0), G(1, 0), G(2, 0), G(0, 1), G(1, 1), G(2, 1), G(0, 2), G(1, 2), G(2, 2));
#undef G
						}
				}
	}
	void getMinMaxLoc(int2& a_Min, int2& a_Max)
	{
		a_Min = make_int2(INT_MAX, INT_MAX);
		a_Max = make_int2(-INT_MAX, -INT_MAX);
		for(int yg = 0; yg < L; yg++)
			for(int xg = 0; xg < L; xg++)
				if(m_T[xg][yg] != 0)
				{
					int x = xg - L / 2, y = yg - L / 2;
					a_Min.x = MIN(a_Min.x, x);
					a_Min.y = MIN(a_Min.y, y);
					a_Max.x = MAX(a_Max.x, x + 1);
					a_Max.y = MAX(a_Max.y, y + 1);
				}
	}
};

void createNode(Hasher& H, e_DynamicScene* S, IMember* obj, IMember* cell)
{
	if(obj->type == "REFR")
	{
		NAME* name = obj->get<NAME>("NAME");
		IMember* f = H.Find<IMember>(name->refid);
		DATA* d = obj->get<DATA>("DATA");
		BaseField* scl = obj->get<BaseField>("XSCL");
		EDID* ed = f ? f->get<EDID>("EDID") : 0;
		float4x4 mat = d->getMat(scl ? *(float*)scl->data : 1.0f);

		char path[256];path[0]=0;
		int q0, q1;
#define COPY_PART_TO(from) q0 = strlen(path); q1 = strlen(from); memcpy(path + q0, from, q1); path[q0 + q1] = 0;
		COPY_PART_TO("Scenes\\Skyrim\\meshes\\");

		if(!f)
			OutputDebugString((char*)&name->refid);
		else if(f->type == "LIGH")
		{
			//COPY_PART_TO("marker_light.nif");
			path[0] = 0;
			OBND* bo = obj->get<OBND>("OBND");
			LIGH_DATA* d = f->get<LIGH_DATA>("DATA");
			float s = 50;
			AABB box = (bo ? bo->box() : AABB(make_float3(-s), make_float3(s))).Transform(mat);
			float3 col = !COLORREFToFloat4(d->rgb) * 3;
			S->creatLight(e_PointLight(box.Center(), col));
			//S->addSphereLight(box.minV + (box.maxV - box.minV) / 2.0f, length(box.maxV - box.minV) / 2.0f, col);
			//S->addDirectionalLight(box, make_float3(0,-1,0), col);
		}
		else if(f->type == "ARMO")
		{
			BaseField* mdl = f->get<BaseField>("MOD2");
			COPY_PART_TO(mdl->data);
		}
		else
		{
			MODL* m = f->get<MODL>("MODL");
			if(!m)
				return;
			std::string strA(m->model);
			std::transform(strA.begin(), strA.end(), strA.begin(), ::tolower);
			if(strstr(strA.c_str(), "effects\\") || strstr(strA.c_str(), "traps\\") || strstr(strA.c_str(), "sky\\") ||
			   strstr(strA.c_str(), "magic\\") || strstr(strA.c_str(), "markers\\") || strstr(strA.c_str(), "markerx") ||
			   strstr(strA.c_str(), "marker"))
				return;
			COPY_PART_TO(strA.c_str());	
		}

		if(strstr(path, "wrcastlebridge"))
			path[0] = path[0];

		if(path[0])
		{
			e_StreamReference(e_Node) N = S->CreateNode(path);
			S->SetNodeTransform(mat, N);
		}
	}
}

void ImportMap(e_DynamicScene* S, Record* cell, Group* grp, Hasher& H)
{
	cell->AccessAll();
	grp->AccessAll();
	for(int i = 0; i < grp->Count(); i++)
	{
		Group* g = (Group*)grp[0][i];
		for(int j = 0; j < g->Count(); j++)
			createNode(H, S, g[0][j], cell);
	}
}

void ImportMap2(e_DynamicScene* S, Group* grp, IMember* cell, Hasher& H)
{
	grp->AccessAll();
	for(int i = 0; i < grp->Count(); i++)
	{
		if(grp[0][i]->type == "GRUP")
			ImportMap2(S, (Group*)grp[0][i], cell, H);//grp[0][i - 1]
		else if(grp[0][i]->type == "REFR" || grp[0][i]->type == "LAND")
			createNode(H, S, grp[0][i], cell);
	}
}

void parseMaps(Group* g, IMember* cell, std::vector<std::pair<int2, IMember*>>& HA)
{
	IMember* l = cell;
	for(int i = 0; i < g->Count(); i++)
	{
		if(g[0][i]->type == "CELL")
		{
			l = g[0][i];
			((Record*)l)->Decrypt();
		}
		else if(g[0][i]->type == "GRUP")
			parseMaps((Group*)g[0][i], l, HA);
		else if(g[0][i]->type == "LAND")
		{
			int* xclc = (int*)cell->get<BaseField>("XCLC")->data;
			((Record*)g[0][i])->Decrypt();
			if(!g[0][i]->get<BaseField>("VHGT"))
				continue;
			HA.push_back(std::make_pair(*(int2*)xclc, g[0][i]));
		}
	}
}

enum PRINT_TYPE
{
	PT_HEIGHT,
	PT_NRMLS,
};
template<PRINT_TYPE PT> void printMap(e_Terrain* T, char* C)
{
	float2 sdxy = 2.0f * T->getKernelData().getsdxy();
	float2 minMax = make_float2(FLT_MAX,-FLT_MAX);
	int s = T->getSideSize(), s2 = s / 2;
	for(int i = 0; i < s2 * s2; i++)
	{
		e_TerrainData_Leaf* l = T->getValAt((i % s2) * 2, (i / s2) * 2);
		if(!l->hasValidHeight()) continue;
		float2 ra = l->getRange();
		minMax.x = fminf(minMax.x, ra.x);
		minMax.y = fmaxf(minMax.y, ra.y);
	}
	FILE *f = fopen(C, "w"); 
	fprintf(f, "P3\n%d %d\n%d\n", s, s, 255);
	for(int y = 0; y < s; y++)
		for(int x = 0; x < s; x++)
		{
			e_TerrainData_Leaf* l = T->getValAt(x, y);
			if(!l->hasValidHeight())
				fprintf(f,"%d %d %d \n", 255, 0, 0);
			else
			{
				int xo = x % 2, yo = y % 2;
				if(PT == PT_NRMLS)
				{
					float3 q2 = l->calcNormal(sdxy.x, sdxy.y) * 127.0f + make_float3(127.0f);
					uchar3 v = make_uchar3(q2.x, q2.y, q2.z);
					fprintf(f,"%d %d %d \n", v.x, v.y, v.z);
				}
				else if(PT == PT_HEIGHT)
				{
					float v0 = (l->H0 - minMax.x) / (minMax.y - minMax.x);
					int v = (int)(v0 * 255);
					fprintf(f,"%d %d %d \n", v, v, v);
				}
			}
		}
	fclose(f);
}

void printMapLevel(e_Terrain* T, unsigned int lvl, char* name)
{
	int s = T->getSideSize() * 2;
	uchar3* dat = new uchar3[s * s];
	T->printLevelMap(lvl, dat);
	FILE *f = fopen(name, "w"); 
	fprintf(f, "P3\n%d %d\n%d\n", s, s, 255);
	for(int y = 0; y < s; y++)
		for(uchar3* p = dat + y * s; p < dat + y * s + s; p++)
			fprintf(f,"%d %d %d \n", p->x, p->y, p->z);
	fclose(f);
}

void setCamera(e_Terrain* T, e_Camera* C, transporterCollection<128, 32>* H = 0)
{
	int2 minloc = make_int2(0,0), maxloc = make_int2(0,0);
	if(H)
		H->getMinMaxLoc(minloc, maxloc);
	int2 mid = (maxloc - minloc) * 16;
	float h = T->getValAt(mid.x, mid.y)->H0;
	float2 mi = make_float2(mid) * T->getKernelData().getsdxy();
	float3 p = make_float3(mi.x, h == -FLT_MAX ? 0 : h, mi.y);
	C->Set(p, p + make_float3(-1,0,1));
	float q = length(T->getKernelData().getFlatScale());
	C->setSpeed(q / 100.0f);
}

e_Terrain* createTerrain(transporterCollection<128, 32>& H, float2 minP, float2 maxP, e_Camera* C)
{
	int2 minloc, maxloc;
	H.getMinMaxLoc(minloc, maxloc);

	int w0 = maxloc.x - minloc.x, w = w0 * 32, h0 = maxloc.y - minloc.y, h2 = h0 * 32;
	char msg[255];
	ZeroMemory(msg, 255),
	sprintf(msg, "W : %d(%d), H : %d(%d)\n", w, w0, h2, h0);
	OutputDebugString(msg);

	int2 dif = (maxloc - minloc) * 32;
	int mdif = MAX(dif.x, dif.y), i0 = 0;
	while(pow2(i0) < mdif)
		i0++;
	if(i0 >= 13)
		throw 1;
	e_Terrain* T = new e_Terrain(i0 + 1, minP, maxP);
	H.IntegrateTransporters(T);
	T->updateFromTriangles();
	setCamera(T, C, &H);
	return T;
}

void load(char* c, e_DynamicScene* S, e_Camera* C)
{
	InputStream I(c);
	e_Terrain* T2 = new e_Terrain(I);
	S->setTerrain(T2);
	I.Close();
	setCamera(T2, C);
	//return T2;
}

void SkyrimImporter::ReadEsm(e_DynamicScene* S, e_Camera* C)
{
	std::vector<Group*> groups = importESM("Skyrim.esm");
	Hasher v_Hasher(groups);
	int j = 0;
	while(!(groups[j]->memberTypes == "WRLD"))
		j++;
	/*
	int lev0 = 0, lev1 = 2, lev2 = 0;
	IMember* lev0O = groups[j-1][0][lev0];
	IMember* lev1O = lev0O[0][lev1];
	IMember* cell = lev1O[0][2 * lev2], *group = lev1O[0][2 * lev2 + 1];
	ImportMap(S, (Record*)cell, (Group*)group, v_Hasher); return;*/

	//ImportMap2(S, (Group*)groups[j]->GET(1, 2, 3), 0, v_Hasher); return;
	
	int map = 0;//35
	char file[128];
	sprintf(file, "ter%d.bin", map);
	Record* wrld = (Record*)groups[j]->GET(2*map);
	wrld->Decrypt();
	float4 scl = *(float4*)wrld->get<BaseField>("ONAM")->data;
	BaseField* b0 = wrld->get<BaseField>("NAM0"), *b1 = wrld->get<BaseField>("NAM9");
	void* d0 = b0->data, *d1 = b1->data;
	float2 a0 = *(float2*)d0, a1 = *(float2*)d1;/*
	//a1 = a0 + make_float2(fmaxf(a1 - a0));
	//g_Offset = transMat * make_float3(scl.y, scl.z, scl.w);
	//g_Offset *= 7;g_Offset.y=10;
	a0 += make_float2(g_Offset.x, g_Offset.z);
	a1 += make_float2(g_Offset.x, g_Offset.z);
	ImportMap2(S, (Group*)groups[j][0][1 + map * 2], 0, v_Hasher); 
	//return load(file, S, C);
	return;	   */
	std::vector<std::pair<int2, IMember*>> dat;
	parseMaps((Group*)groups[j]->GET(2*map+1), 0, dat);
	transporterCollection<128, 32> H;
	for(int i = 0; i < dat.size(); i++)
	{
		vhData* Q = (vhData*)dat[i].second->get<BaseField>("VHGT")->data;
		e_TerrainDataTransporter<32>* Q2 = Q->getTransporter((char3*)dat[i].second->get<BaseField>("VNML")->data, dat[i].first);
		H(Q2);
	}
	e_Terrain* T = createTerrain(H, a0, a1, C);
	S->setTerrain(T);
	OutputStream A(file);
	T->Serialize(A);
	A.Close();
	printMap<PT_HEIGHT>(T, "1.ppm");
	//printMap<PT_NRMLS>(T, "2.ppm");
}