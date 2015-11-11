#include <StdAfx.h>
#include <Engine/e_Mesh.h>
#include <Engine/e_TriangleData.h>
#include <Engine/e_Material.h>
#include <Engine/e_TriIntersectorData.h>
#include "TangentSpaceHelper.h"
#include "Importer.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <Base/FileStream.h>
#include <map>

namespace CudaTracerLib {

typedef unsigned char       U8;
typedef unsigned short      U16;
typedef unsigned int        U32;
typedef signed char         S8;
typedef signed short        S16;
typedef signed int          S32;
typedef float               F32;
typedef double              F64;
typedef unsigned __int64    U64;
typedef signed __int64      S64;
typedef __w64 S32           SPTR;
typedef __w64 U32           UPTR;
template <class T> class Set
{
private:
	enum
	{
		BlockSize = 8,
		MinBytes = 32,
		MaxUsagePct = 60,
		ThrUsagePct = MaxUsagePct * 3 / 4
	};

	enum HashValue
	{
		Empty = -1,
		Removed = -2
	};

public:
	Set(void)                          { init(); }
	Set(const Set<T>& other)           { init(); set(other); }
	~Set(void)                          { reset(); }

	int                 getSize(void) const                    { return m_numItems; }
	bool                contains(const T& value) const          { return (findSlot(value) != -1); }
	const T*            search(const T& value) const          { int slot = findSlot(value); return (slot == -1) ? NULL : &m_values[slot]; }
	T*                  search(const T& value)                { int slot = findSlot(value); return (slot == -1) ? NULL : &m_values[slot]; }
	const T&            get(const T& value) const          { int slot = findSlot(value); FW_ASSERT(slot != -1); return m_values[slot]; }
	T&                  get(const T& value)                { int slot = findSlot(value); FW_ASSERT(slot != -1); return m_values[slot]; }

	void                clear(void)                          { m_numItems = 0; m_numNonEmpty = 0; memset(m_hashes, Empty, m_capacity * sizeof(S32)); }
	void                reset(void)                          { delete[] m_hashes; delete[] m_values; init(); }
	void                setCapacity(int numItems);
	void                compact(void)                          { setCapacity(m_numItems); }
	void                set(const Set<T>& other);

	T&                  add(const T& value)                { T& slot = addNoAssign(value); slot = value; return slot; }
	T&                  addNoAssign(const T& value);
	T*                  addNoAssign(const T* value);
	T&                  remove(const T& value);
	T                   replace(const T& value);

	int                 findSlot(const T& value) const;
	int                 firstSlot(void) const                    { return nextSlot(-1); }
	int                 nextSlot(int slot) const;
	const T&            getSlot(int slot) const                { FW_ASSERT(m_hashes[slot] >= 0); return m_values[slot]; }
	T&                  getSlot(int slot)                      { FW_ASSERT(m_hashes[slot] >= 0); return m_values[slot]; }

	Set<T>&             operator=   (const Set<T>& other)           { set(other); return *this; }
	const T&            operator[]  (const T& value) const          { return get(value); }
	T&                  operator[]  (const T& value)                { return get(value); }

private:
	void                init(void)                          { m_capacity = 0; m_numItems = 0; m_numNonEmpty = 0; m_hashes = NULL; m_values = NULL; }
	int                 findSlot(const T& value, S32 hashValue, bool needEmpty) const;
	void                rehash(int capacity);

private:
	S32                 m_capacity;
	S32                 m_numItems;
	S32                 m_numNonEmpty;
	S32*                m_hashes;
	T*                  m_values;
};

#define FW_HASH_MAGIC   (0x9e3779b9u)

// By Bob Jenkins, 1996. bob_jenkins@burtleburtle.net.
#define FW_JENKINS_MIX(a, b, c)   \
	a -= b; a -= c; a ^= (c>>13); \
	b -= c; b -= a; b ^= (a<<8);  \
	c -= a; c -= b; c ^= (b>>13); \
	a -= b; a -= c; a ^= (c>>12); \
	b -= c; b -= a; b ^= (a<<16); \
	c -= a; c -= b; c ^= (b>>5);  \
	a -= b; a -= c; a ^= (c>>3);  \
	b -= c; b -= a; b ^= (a<<10); \
	c -= a; c -= b; c ^= (b>>15);

inline bool equalsBuffer(const void* ptrA, const void* ptrB, int size)              { return (memcmp(ptrA, ptrB, size) == 0); }
U32 hashBufferAlign(const void* ptr, int size)
{
	CT_ASSERT(size >= 0);
	CT_ASSERT(ptr || !size);
	CT_ASSERT(((UPTR)ptr & 3) == 0);
	CT_ASSERT((size & 3) == 0);

	const U32*  src = (const U32*)ptr;
	U32         a = FW_HASH_MAGIC;
	U32         b = FW_HASH_MAGIC;
	U32         c = FW_HASH_MAGIC;

	while (size >= 12)
	{
		a += src[0];
		b += src[1];
		c += src[2];
		FW_JENKINS_MIX(a, b, c);
		src += 3;
		size -= 12;
	}

	switch (size)
	{
	case 8: b += src[1];
	case 4: a += src[0];
	case 0: break;
	}

	c += size;
	FW_JENKINS_MIX(a, b, c);
	return c;
}
U32         hashBuffer(const void* ptr, int size)
{
	CT_ASSERT(size >= 0);
	CT_ASSERT(ptr || !size);

	if ((((S32)(UPTR)ptr | size) & 3) == 0)
		return hashBufferAlign(ptr, size);

	const U8*   src = (const U8*)ptr;
	U32         a = FW_HASH_MAGIC;
	U32         b = FW_HASH_MAGIC;
	U32         c = FW_HASH_MAGIC;

	while (size >= 12)
	{
		a += src[0] + (src[1] << 8) + (src[2] << 16) + (src[3] << 24);
		b += src[4] + (src[5] << 8) + (src[6] << 16) + (src[7] << 24);
		c += src[8] + (src[9] << 8) + (src[10] << 16) + (src[11] << 24);
		FW_JENKINS_MIX(a, b, c);
		src += 12;
		size -= 12;
	}

	switch (size)
	{
	case 11: c += src[10] << 16;
	case 10: c += src[9] << 8;
	case 9:  c += src[8];
	case 8:  b += src[7] << 24;
	case 7:  b += src[6] << 16;
	case 6:  b += src[5] << 8;
	case 5:  b += src[4];
	case 4:  a += src[3] << 24;
	case 3:  a += src[2] << 16;
	case 2:  a += src[1] << 8;
	case 1:  a += src[0];
	case 0:  break;
	}

	c += size;
	FW_JENKINS_MIX(a, b, c);
	return c;
}
inline U32  hashBits(U32 a, U32 b = FW_HASH_MAGIC, U32 c = 0)                   { c += FW_HASH_MAGIC; FW_JENKINS_MIX(a, b, c); return c; }
inline U32  hashBits(U32 a, U32 b, U32 c, U32 d, U32 e = 0, U32 f = 0)          { c += FW_HASH_MAGIC; FW_JENKINS_MIX(a, b, c); a += d; b += e; c += f; FW_JENKINS_MIX(a, b, c); return c; }
template <class T>  inline bool equals(const T& a, const T& b)                { static_assert(false, "IMPL!"); }
template <class T>  inline U32  hash(const T& value)                        { static_assert(false, "IMPL!"); }

template <> inline bool equals<S8>(const S8& a, const S8& b)          { return (a == b); }
template <> inline bool equals<U8>(const U8& a, const U8& b)          { return (a == b); }
template <> inline bool equals<S16>(const S16& a, const S16& b)        { return (a == b); }
template <> inline bool equals<U16>(const U16& a, const U16& b)        { return (a == b); }
template <> inline bool equals<S32>(const S32& a, const S32& b)        { return (a == b); }
template <> inline bool equals<U32>(const U32& a, const U32& b)        { return (a == b); }
template <> inline bool equals<F32>(const F32& a, const F32& b)        { return (math::floatToBits(a) == math::floatToBits(b)); }
template <> inline bool equals<S64>(const S64& a, const S64& b)        { return (a == b); }
template <> inline bool equals<U64>(const U64& a, const U64& b)        { return (a == b); }

template <> inline U32  hash<S8>(const S8& value)                   { return hashBits(value); }
template <> inline U32  hash<U8>(const U8& value)                   { return hashBits(value); }
template <> inline U32  hash<S16>(const S16& value)                  { return hashBits(value); }
template <> inline U32  hash<U16>(const U16& value)                  { return hashBits(value); }
template <> inline U32  hash<S32>(const S32& value)                  { return hashBits(value); }
template <> inline U32  hash<U32>(const U32& value)                  { return hashBits(value); }
template <> inline U32  hash<F32>(const F32& value)                  { return hashBits(math::floatToBits(value)); }
template <> inline U32  hash<S64>(const S64& value)                  { return hashBits((U32)value, (U32)(value >> 32)); }
template <> inline U32  hash<U64>(const U64& value)                  { return hash<S64>((S64)value); }

//------------------------------------------------------------------------
// Specializations for compound types.
//------------------------------------------------------------------------

template <> inline bool equals<Vec2i>(const Vec2i& a, const Vec2i& b)    { return (a == b); }
template <> inline bool equals<Vec2f>(const Vec2f& a, const Vec2f& b)    { return (equals<F32>(a.x, b.x) && equals<F32>(a.y, b.y)); }
template <> inline bool equals<Vec3i>(const Vec3i& a, const Vec3i& b)    { return (a == b); }
template <> inline bool equals<Vec3f>(const Vec3f& a, const Vec3f& b)    { return (equals<F32>(a.x, b.x) && equals<F32>(a.y, b.y) && equals<F32>(a.z, b.z)); }
template <> inline bool equals<Vec4i>(const Vec4i& a, const Vec4i& b)    { return (a == b); }
template <> inline bool equals<Vec4f>(const Vec4f& a, const Vec4f& b)    { return (equals<F32>(a.x, b.x) && equals<F32>(a.y, b.y) && equals<F32>(a.z, b.z) && equals<F32>(a.w, b.w)); }

template <> inline U32  hash<Vec2i>(const Vec2i& value)                { return hashBits(value.x, value.y); }
template <> inline U32  hash<Vec2f>(const Vec2f& value)                { return hashBits(math::floatToBits(value.x), math::floatToBits(value.y)); }
template <> inline U32  hash<Vec3i>(const Vec3i& value)                { return hashBits(value.x, value.y, value.z); }
template <> inline U32  hash<Vec3f>(const Vec3f& value)                { return hashBits(math::floatToBits(value.x), math::floatToBits(value.y), math::floatToBits(value.z)); }
template <> inline U32  hash<Vec4i>(const Vec4i& value)                { return hashBits(value.x, value.y, value.z, value.w); }
template <> inline U32  hash<Vec4f>(const Vec4f& value)                { return hashBits(math::floatToBits(value.x), math::floatToBits(value.y), math::floatToBits(value.z), math::floatToBits(value.w)); }

template <class T, class TT> inline bool equals(TT* const& a, TT* const& b) { return (a == b); }
template <class T, class TT> inline U32 hash(TT* const& value) { return hashBits((U32)(UPTR)value); }

template <class T> int  Set<T>::findSlot(const T& value) const
{
	return findSlot(value, hash<T>(value) >> 1, false);
}

template <class T> void Set<T>::setCapacity(int numItems)
{
	int capacity = BlockSize;
	S64 limit = (S64)max(numItems, m_numItems, (MinBytes + (S32)sizeof(T) - 1) / (S32)sizeof(T)) * 100;
	while ((S64)capacity * MaxUsagePct < limit)
		capacity <<= 1;

	if (capacity != m_capacity)
		rehash(capacity);
}

template <class T> T& Set<T>::addNoAssign(const T& value)
{
	CT_ASSERT(!contains(value));

	// Empty => allocate.

	if (!m_capacity)
		setCapacity(0);

	// Exceeds MaxUsagePct => rehash.

	else if ((S64)m_numNonEmpty * 100 >= (S64)m_capacity * MaxUsagePct)
	{
		int cap = m_capacity;
		if ((S64)m_numItems * 100 >= (S64)cap * ThrUsagePct)
			cap <<= 1;
		rehash(cap);
	}

	// Find slot.

	S32 hashValue = hash<T>(value) >> 1;
	int slot = findSlot(value, hashValue, true);
	CT_ASSERT(m_hashes[slot] < 0);

	// Add item.

	m_numItems++;
	if (m_hashes[slot] == Empty)
		m_numNonEmpty++;

	m_hashes[slot] = hashValue;
	return m_values[slot];
}

template <class T> T* Set<T>::addNoAssign(const T* value)
{
	CT_ASSERT(!contains(*value));

	// Empty => allocate.

	if (!m_capacity)
		setCapacity(0);

	// Exceeds MaxUsagePct => rehash.

	else if ((S64)m_numNonEmpty * 100 >= (S64)m_capacity * MaxUsagePct)
	{
		int cap = m_capacity;
		if ((S64)m_numItems * 100 >= (S64)cap * ThrUsagePct)
			cap <<= 1;
		rehash(cap);
	}

	// Find slot.

	S32 hashValue = hash<T>(*value) >> 1;
	int slot = findSlot(*value, hashValue, true);
	CT_ASSERT(m_hashes[slot] < 0);

	// Add item.

	m_numItems++;
	if (m_hashes[slot] == Empty)
		m_numNonEmpty++;

	m_hashes[slot] = hashValue;
	return m_values + slot;
}

template <class T> int Set<T>::findSlot(const T& value, S32 hashValue, bool needEmpty) const
{
	CT_ASSERT(hashValue >= 0);
	if (!m_capacity)
		return -1;

	int blockMask = (m_capacity - 1) & -BlockSize;
	int firstSlot = hashValue;
	int firstBlock = firstSlot & blockMask;
	int blockStep = BlockSize * 3 + ((hashValue >> 17) & (-4 * BlockSize));

	int block = firstBlock;
	do
	{
		if (needEmpty)
		{
			for (int i = 0; i < BlockSize; i++)
			{
				int slot = block + ((firstSlot + i) & (BlockSize - 1));
				if (m_hashes[slot] < 0)
					return slot;
			}
		}
		else
		{
			for (int i = 0; i < BlockSize; i++)
			{
				int slot = block + ((firstSlot + i) & (BlockSize - 1));
				S32 slotHash = m_hashes[slot];

				if (slotHash == Empty)
					return -1;

				if (slotHash == hashValue && equals<T>(m_values[slot], value))
					return slot;
			}
		}

		block = (block + blockStep) & blockMask;
		blockStep += BlockSize * 4;
	} while (block != firstBlock);
	return -1;
}

template <class T> void Set<T>::rehash(int capacity)
{
	CT_ASSERT(capacity >= BlockSize);
	CT_ASSERT(capacity >= m_numItems);

	int oldCapacity = m_capacity;
	S32* oldHashes = m_hashes;
	T* oldValues = m_values;
	m_capacity = capacity;
	m_numNonEmpty = m_numItems;
	m_hashes = new S32[capacity];
	m_values = new T[capacity];

	memset(m_hashes, Empty, capacity * sizeof(S32));

	for (int i = 0; i < oldCapacity; i++)
	{
		S32 oldHash = oldHashes[i];
		if (oldHash < 0)
			continue;

		const T& oldValue = oldValues[i];
		int slot = findSlot(oldValue, oldHash, true);
		CT_ASSERT(m_hashes[slot] == Empty);

		m_hashes[slot] = oldHash;
		m_values[slot] = oldValue;
	}

	delete[] oldHashes;
	delete[] oldValues;
}

template <class K, class V> struct HashEntry
{
	K                   key;
	V                   value;
};

template <class K, class V> class Hash
{
public:
	typedef HashEntry<K, V> Entry;

public:
	Hash(void)                          {}
	Hash(const Hash<K, V>& other)       { set(other); }
	~Hash(void)                          {}

	const Set<Entry>&   getEntries(void) const                    { return m_entries; }
	Set<Entry>&         getEntries(void)                          { return m_entries; }
	int                 getSize(void) const                    { return m_entries.getSize(); }
	bool                contains(const K& key) const            { return m_entries.contains(keyEntry(key)); }
	const Entry*        searchEntry(const K& key) const            { return m_entries.search(keyEntry(key)); }
	Entry*              searchEntry(const K& key)                  { return m_entries.search(keyEntry(key)); }
	const K*            searchKey(const K& key) const            { const Entry* e = searchEntry(key); return (e) ? &e->key : NULL; }
	K*                  searchKey(const K& key)                  { Entry* e = searchEntry(key); return (e) ? &e->key : NULL; }
	const V*            search(const K& key) const            { const Entry* e = searchEntry(key); return (e) ? &e->value : NULL; }
	V*                  search(const K& key)                  { Entry* e = searchEntry(key); return (e) ? &e->value : NULL; }
	const Entry&        getEntry(const K& key) const            { return m_entries.get(keyEntry(key)); }
	Entry&              getEntry(const K& key)                  { return m_entries.get(keyEntry(key)); }
	const K&            getKey(const K& key) const            { return getEntry(key).key; }
	K&                  getKey(const K& key)                  { return getEntry(key).key; }
	const V&            get(const K& key) const            { return getEntry(key).value; }
	V&                  get(const K& key)                  { return getEntry(key).value; }

	void                clear(void)                          { m_entries.clear(); }
	void                reset(void)                          { m_entries.reset(); }
	void                setCapacity(int numItems)                  { m_entries.setCapacity(numItems); }
	void                compact(void)                          { m_entries.compact(); }
	void                set(const Hash<K, V>& other)       { m_entries.set(other.m_entries); }

	V&                  add(const K& key, const V& value)
	{
		Entry a0 = keyEntry(key, value);
		Entry* slot = m_entries.addNoAssign(&a0);
		slot->key = key; slot->value = value;
		return slot->value;
	}
	V&                  add(const K& key)                  { Entry& slot = m_entries.addNoAssign(keyEntry(key)); slot.key = key; return slot.value; }
	V&                  remove(const K& key)                  { return m_entries.remove(keyEntry(key)).value; }
	V                   replace(const K& key, const V& value)  { Entry e; e.key = key; e.value = value; return m_entries.replace(e).value; }

	int                 findSlot(const K& key) const            { return m_entries.findSlot(keyEntry(key)); }
	int                 firstSlot(void) const                    { return m_entries.firstSlot(); }
	int                 nextSlot(int slot) const                { return m_entries.nextSlot(slot); }
	const Entry&        getSlot(int slot) const                { return m_entries.getSlot(slot); }
	Entry&              getSlot(int slot)                      { return m_entries.getSlot(slot); }

	Hash<K, V>&         operator=   (const Hash<K, V>& other)       { set(other); return *this; }
	const V&            operator[]  (const K& key) const            { return get(key); }
	V&                  operator[]  (const K& key)                  { return get(key); }

private:
	static const Entry& keyEntry(const K& key)                  { return *(Entry*)&key; }
	static const Entry keyEntry(const K& key, const V& value)   { Entry E; E.key = key; E.value = value; return E; }

private:
	Set<Entry>          m_entries;
};

template <class T, class K, class V> inline bool equals(const HashEntry<K, V>& a, const HashEntry<K, V>& b) { return equals<K>(a.key, b.key); }
template <class T, class K, class V> inline U32 hash(const HashEntry<K, V>& value) { return hash<K>(value.key); }

bool parseSpace(const char*& ptr)
{
	while (*ptr == ' ' || *ptr == '\t')
		ptr++;
	return true;
}

bool parseChar(const char*& ptr, char chr)
{
	if (*ptr != chr)
		return false;
	ptr++;
	return true;
}

bool parseLiteral(const char*& ptr, const char* str)
{
	const char* tmp = ptr;

	while (*str && *tmp == *str)
	{
		tmp++;
		str++;
	}
	if (*str)
		return false;

	ptr = tmp;
	return true;
}

bool parseInt(const char*& ptr, int& value)
{
	const char* tmp = ptr;
	int v = 0;
	bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
	if (*tmp < '0' || *tmp > '9')
		return false;
	while (*tmp >= '0' && *tmp <= '9')
		v = v * 10 + *tmp++ - '0';

	value = (neg) ? -v : v;
	ptr = tmp;
	return true;
}

bool parseInt(const char*& ptr, long long& value)
{
	const char* tmp = ptr;
	long long v = 0;
	bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));
	if (*tmp < '0' || *tmp > '9')
		return false;
	while (*tmp >= '0' && *tmp <= '9')
		v = v * 10 + *tmp++ - '0';

	value = (neg) ? -v : v;
	ptr = tmp;
	return true;
}

bool parseHex(const char*& ptr, unsigned int& value)
{
	const char* tmp = ptr;
	unsigned int v = 0;
	for (;;)
	{
		if (*tmp >= '0' && *tmp <= '9')         v = v * 16 + *tmp++ - '0';
		else if (*tmp >= 'A' && *tmp <= 'F')    v = v * 16 + *tmp++ - 'A' + 10;
		else if (*tmp >= 'a' && *tmp <= 'f')    v = v * 16 + *tmp++ - 'a' + 10;
		else                                    break;
	}

	if (tmp == ptr)
		return false;

	value = v;
	ptr = tmp;
	return true;
}

bool parseFloat(const char*& ptr, float& value)
{
#define bitsToFloat(x) (*(float*)&x)
	const char* tmp = ptr;
	bool neg = (!parseChar(tmp, '+') && parseChar(tmp, '-'));

	float v = 0.0f;
	int numDigits = 0;
	while (*tmp >= '0' && *tmp <= '9')
	{
		v = v * 10.0f + (float)(*tmp++ - '0');
		numDigits++;
	}
	if (parseChar(tmp, '.'))
	{
		float scale = 1.0f;
		while (*tmp >= '0' && *tmp <= '9')
		{
			scale *= 0.1f;
			v += scale * (float)(*tmp++ - '0');
			numDigits++;
		}
	}
	if (!numDigits)
		return false;

	ptr = tmp;
	if (*ptr == '#')
	{
		unsigned int v = 0;
		if (parseLiteral(ptr, "#INF"))
			v = 0x7F800000;
		else if (parseLiteral(ptr, "#SNAN"))
			v = 0xFF800001;
		else if (parseLiteral(ptr, "#QNAN"))
			v = 0xFFC00001;
		else if (parseLiteral(ptr, "#IND"))
			v = 0xFFC00000;
		if (v)
		{
			v |= neg << 31;
			value = *(float*)&v;
			return true;
		}
		else return false;
	}

	int e = 0;
	if ((parseChar(tmp, 'e') || parseChar(tmp, 'E')) && parseInt(tmp, e))
	{
		ptr = tmp;
		if (e)
			v *= pow(10.0f, (float)e);
	}
	value = (neg) ? -v : v;
	return true;
#undef bitsToFloat
}

enum TextureType
{
	TextureType_Diffuse = 0,    // Diffuse color map.
	TextureType_Alpha,          // Alpha map (green = opacity).
	TextureType_Displacement,   // Displacement map (green = height).
	TextureType_Normal,         // Tangent-space normal map.
	TextureType_Environment,    // Environment map (spherical coordinates).

	TextureType_Max
};

struct Material
{
	std::string		Name;
	int				IlluminationModel;
	Vec4f          diffuse;
	Vec3f          specular;
	Vec3f			emission;
	float           glossiness;
	float           displacementCoef; // height = texture/255 * coef + bias
	float           displacementBias;
	float			IndexOfRefraction;
	Vec3f			Tf;
	std::string     textures[TextureType_Max];
	int				submesh;

	Material(void)
	{
		diffuse = Vec4f(0.75f, 0.75f, 0.75f, 1.0f);
		specular = Vec3f(0.5f);
		glossiness = 32.0f;
		displacementCoef = 1.0f;
		displacementBias = 0.0f;
		emission = Vec3f(0, 0, 0);
		submesh = -1;
		IlluminationModel = 2;
	}
};

struct TextureSpec
{
	std::string	              texture;
	float                     base;
	float                     gain;
};

struct MatHash
{
	std::vector<Material> vec;
	std::map<std::string, int> map;

	void add(const std::string& name, const Material& mat)
	{
		map[name] = (int)vec.size();
		vec.push_back(mat);
	}

	bool contains(const std::string& name)
	{
		return map.find(name) != map.end();
	}

	int searchi(const std::string& name)
	{
		std::map<std::string, int>::iterator it = map.find(name);
		return it == map.end() ? -1 : it->second;
	}
};

bool parseFloats(const char*& ptr, float* values, int num)
{
	const char* tmp = ptr;
	for (int i = 0; i < num; i++)
	{
		if (i)
			parseSpace(tmp);
		if (!parseFloat(tmp, values[i]))
			return false;
	}
	ptr = tmp;
	return true;
}

bool parseTexture(const char*& ptr, TextureSpec& value, const std::string& dirName)
{
	// Initialize result.

	std::string name;
	value.texture = "";
	value.base = 0.0f;
	value.gain = 1.0f;

	// Parse options.

	while (*ptr)
	{
		parseSpace(ptr);
		if ((parseLiteral(ptr, "-blendu ") || parseLiteral(ptr, "-blendv ") || parseLiteral(ptr, "-cc ") || parseLiteral(ptr, "-math::clamp ")) && parseSpace(ptr))
		{
			if (!parseLiteral(ptr, "on") && !parseLiteral(ptr, "off"))
				return false;
		}
		else if (parseLiteral(ptr, "-mm ") && parseSpace(ptr))
		{
			if (!parseFloat(ptr, value.base) || !parseSpace(ptr) || !parseFloat(ptr, value.gain))
				return false;
		}
		else if ((parseLiteral(ptr, "-o ") || parseLiteral(ptr, "-s ") || parseLiteral(ptr, "-t ")) && parseSpace(ptr))
		{
			float tmp[2];
			if (!parseFloats(ptr, tmp, 2))
				return false;
			parseSpace(ptr);
			parseFloat(ptr, tmp[0]);
		}
		else if ((parseLiteral(ptr, "-texres ") || parseLiteral(ptr, "-bm ")) && parseSpace(ptr))
		{
			float tmp;
			if (!parseFloat(ptr, tmp))
				return false;
		}
		else if (parseLiteral(ptr, "-type ") && parseSpace(ptr))
		{
			if (!parseLiteral(ptr, "sphere") &&
				!parseLiteral(ptr, "cube_top") && !parseLiteral(ptr, "cube_bottom") &&
				!parseLiteral(ptr, "cube_front") && !parseLiteral(ptr, "cube_back") &&
				!parseLiteral(ptr, "cube_left") && !parseLiteral(ptr, "cube_right"))
			{
				return false;
			}
		}
		else
		{
			if (*ptr == '-' || name.size())
				return false;
			while (*ptr && (*ptr != '-' || !boost::algorithm::ends_with(name, " ")))
				name += *ptr++;
		}
	}

	// Process file name.

	while (boost::algorithm::starts_with(name, "/"))
		name = name.substr(1);
	while (boost::algorithm::ends_with(name, " "))
		name = name.substr(0, name.size() - 1);

	// Zero-length file name => ignore.

	if (!name.size())
		return true;

	// Import texture.

	value.texture = dirName + '/' + name;

	return true;
}

struct SubMesh
{
	std::vector<Vec3i>   indices;
	Material        material;

	SubMesh()
	{
		material.submesh = -1234;
	}
};

struct ImportState
{
	std::vector<SubMesh> subMeshes;

	std::vector<Vec3f>            positions;
	std::vector<Vec2f>            texCoords;
	std::vector<Vec3f>            normals;

	Hash<Vec3i, int>        vertexHash;
	MatHash materialHash;

	std::vector<int>              vertexTmp;
	std::vector<Vec3i>            indexTmp;

	struct VertexPNT
	{
		Vec3f p;
		Vec2f t;
		Vec3f n;
	};
	std::vector<VertexPNT> vertices;

	int addVertex()
	{
		vertices.push_back(VertexPNT());
		return (int)vertices.size() - 1;
	}

	int addSubMesh()
	{
		subMeshes.push_back(SubMesh());
		return (int)subMeshes.size() - 1;
	}

	unsigned int numTriangles()
	{
		size_t n = 0;
		for (auto& s : subMeshes)
			n += s.indices.size();
		return (unsigned int)n;
	}
};

void loadMtl(ImportState& s, IInStream& mtlIn, const std::string& dirName)
{
	char ptrLast[256];
	Material* mat = NULL;
	std::string lineS;
	while (mtlIn.getline(lineS))
	{
		boost::algorithm::trim(lineS);
		const char* ptr = lineS.c_str();
		parseSpace(ptr);
		bool valid = false;

		if (!*ptr || parseLiteral(ptr, "#"))
		{
			valid = true;
		}
		else if (parseLiteral(ptr, "newmtl ") && parseSpace(ptr) && *ptr) // material name
		{
			if (mat != 0)
				s.materialHash.add(std::string(ptrLast), *mat);
			if (!s.materialHash.contains(std::string(ptr)))
			{
				mat = new Material();
				Platform::SetMemory(ptrLast, sizeof(ptrLast));
				memcpy(ptrLast, ptr, strlen(ptr));
				mat->Name = std::string(ptrLast);
			}
			valid = true;
		}
		else if (parseLiteral(ptr, "Ka ") && parseSpace(ptr)) // ambient color
		{
			float tmp[3];
			if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
				valid = true;
			else if (parseFloats(ptr, tmp, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Kd ") && parseSpace(ptr)) // diffuse color
		{
			if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
				valid = true;
			else if (parseFloats(ptr, (float*)&mat->diffuse, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Ks ") && parseSpace(ptr)) // specular color
		{
			if (parseLiteral(ptr, "spectral ") || parseLiteral(ptr, "xyz "))
				valid = true;
			else if (parseFloats(ptr, (float*)&mat->specular, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "d ") && parseSpace(ptr)) // alpha
		{
			if (parseFloat(ptr, mat->diffuse.w) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Ns ") && parseSpace(ptr)) // glossiness
		{
			if (parseFloat(ptr, mat->glossiness) && parseSpace(ptr) && !*ptr)
				valid = true;
			if (mat->glossiness <= 0.0f)
			{
				mat->glossiness = 1.0f;
				mat->specular = Vec3f(0);
			}
		}
		else if (parseLiteral(ptr, "map_Kd ")) // diffuse texture
		{
			TextureSpec tex;
			mat->textures[TextureType_Diffuse] = std::string(ptr);
			valid = parseTexture(ptr, tex, dirName);
		}
		else if (parseLiteral(ptr, "Ke "))
		{
			if (parseFloats(ptr, (float*)&mat->emission, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Tf "))
		{
			if (parseFloats(ptr, (float*)&mat->Tf, 3) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "Ni ") && parseSpace(ptr)) // alpha
		{
			if (parseFloat(ptr, mat->IndexOfRefraction) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "illum ") && parseSpace(ptr)) // alpha
		{
			if (parseInt(ptr, mat->IlluminationModel) && parseSpace(ptr) && !*ptr)
				valid = true;
		}
		else if (parseLiteral(ptr, "map_d ") || parseLiteral(ptr, "map_D ") || parseLiteral(ptr, "map_opacity ")) // alpha texture
		{
			TextureSpec tex;
			valid = parseTexture(ptr, tex, dirName);
			mat->textures[TextureType_Alpha] = tex.texture;
		}
		else if (parseLiteral(ptr, "disp ")) // displacement map
		{
			TextureSpec tex;
			valid = parseTexture(ptr, tex, dirName);
			mat->displacementCoef = tex.gain;
			mat->displacementBias = tex.base * tex.gain;
			mat->textures[TextureType_Displacement] = tex.texture;
		}
		else if (parseLiteral(ptr, "bump ") || parseLiteral(ptr, "map_bump ") || parseLiteral(ptr, "map_Bump ")) // bump map
		{
			TextureSpec tex;
			mat->displacementCoef = tex.gain;
			mat->displacementBias = tex.base * tex.gain;
			mat->textures[TextureType_Displacement] = std::string(ptr);
			valid = parseTexture(ptr, tex, dirName);
		}
		else if (parseLiteral(ptr, "refl ")) // environment map
		{
			TextureSpec tex;
			valid = parseTexture(ptr, tex, dirName);
			mat->textures[TextureType_Environment] = tex.texture;
		}
		else if (
			parseLiteral(ptr, "vp ") ||             // parameter space vertex
			parseLiteral(ptr, "Kf ") ||             // transmission color
			parseLiteral(ptr, "illum ") ||          // illumination model
			parseLiteral(ptr, "d -halo ") ||        // orientation-dependent alpha
			parseLiteral(ptr, "sharpness ") ||      // reflection sharpness
			parseLiteral(ptr, "Ni ") ||             // index of refraction
			parseLiteral(ptr, "map_Ks ") ||         // specular texture
			parseLiteral(ptr, "map_kS ") ||         // ???
			parseLiteral(ptr, "map_kA ") ||         // ???
			parseLiteral(ptr, "map_Ns ") ||         // glossiness texture
			parseLiteral(ptr, "map_aat ") ||        // texture antialiasing
			parseLiteral(ptr, "decal ") ||          // blended texture
			parseLiteral(ptr, "Km ") ||             // ???
			parseLiteral(ptr, "Tr ") ||             // ???
			parseLiteral(ptr, "Ke ") ||             // ???
			parseLiteral(ptr, "pointgroup ") ||     // ???
			parseLiteral(ptr, "pointdensity ") ||   // ???
			parseLiteral(ptr, "smooth") ||          // ???
			parseLiteral(ptr, "R "))                // ???
		{
			valid = true;
		}
	}
	if (mat != 0)
		s.materialHash.add(std::string(ptrLast), *mat);
}

static e_Texture CreateTexture(const char* p, const Spectrum& col)
{
	if (p && *p)
		return CreateTexture(p);
	else return CreateTexture(col);
}

template<typename T> void push(std::vector<T>& left, const std::vector<T>& right)
{
	std::move(right.begin(), right.end(), std::back_inserter(left));
}

void parse(ImportState& s, IInStream& in)
{
	std::string dirName = boost::filesystem::path(in.getFilePath()).parent_path().string();
	int submesh = -1;
	int defaultSubmesh = -1;
	std::string line;
	while (in.getline(line))
	{
		boost::algorithm::trim(line);
		const char* ptr = line.c_str();
		parseSpace(ptr);
		bool valid = false;

		if (!*ptr || parseLiteral(ptr, "#"))
		{
			valid = true;
		}
		else if (parseLiteral(ptr, "v ") && parseSpace(ptr)) // position vertex
		{
			Vec3f v;
			if (parseFloats(ptr, v.getPtr(), 3) && parseSpace(ptr) && !*ptr)
			{
				s.positions.push_back(v);
				valid = true;
			}
		}
		else if (parseLiteral(ptr, "vt ") && parseSpace(ptr)) // texture vertex
		{
			Vec2f v;
			if (parseFloats(ptr, v.getPtr(), 2) && parseSpace(ptr))
			{
				float dummy;
				while (parseFloat(ptr, dummy) && parseSpace(ptr));

				if (!*ptr)
				{
					s.texCoords.push_back(Vec2f(v.x, 1.0f - v.y));
					valid = true;
				}
			}
		}
		else if (parseLiteral(ptr, "vn ") && parseSpace(ptr)) // normal vertex
		{
			Vec3f v;
			if (parseFloats(ptr, v.getPtr(), 3) && parseSpace(ptr) && !*ptr)
			{
				s.normals.push_back(v);
				valid = true;
			}
		}
		else if (parseLiteral(ptr, "f ") && parseSpace(ptr)) // face
		{
			s.vertexTmp.clear();
			while (*ptr)
			{
				Vec3i ptn;
				if (!parseInt(ptr, ptn.x))
					break;
				for (int i = 1; i < 4 && parseLiteral(ptr, "/"); i++)
				{
					int tmp = 0;
					parseInt(ptr, tmp);
					if (i < 3)
						ptn[i] = tmp;
				}
				parseSpace(ptr);

				Vec3i size((int)s.positions.size(), (int)s.texCoords.size(), (int)s.normals.size());
				for (int i = 0; i < 3; i++)
				{
					if (ptn[i] < 0)
						ptn[i] += size[i];
					else
						ptn[i]--;

					if (ptn[i] < 0 || ptn[i] >= size[i])
						ptn[i] = -1;
				}

				int* idx = s.vertexHash.search(ptn);
				if (idx)
					s.vertexTmp.push_back(*idx);
				else
				{
					size_t vIdx = s.vertices.size();
					s.vertexTmp.push_back(s.vertexHash.add(ptn, (int)vIdx));
					s.vertices.push_back(ImportState::VertexPNT());
					ImportState::VertexPNT& v = s.vertices[vIdx];
					v.p = (ptn.x == -1) ? Vec3f(0.0f) : s.positions[ptn.x];
					v.t = (ptn.y == -1) ? Vec2f(0.0f) : s.texCoords[ptn.y];
					v.n = (ptn.z == -1) ? Vec3f(0.0f) : s.normals[ptn.z];
				}
			}
			if (!*ptr)
			{
				if (submesh == -1)
				{
					if (defaultSubmesh == -1)
						defaultSubmesh = s.addSubMesh();
					submesh = defaultSubmesh;
				}
				for (int i = 2; i < s.vertexTmp.size(); i++)
					s.indexTmp.push_back(Vec3i(s.vertexTmp[0], s.vertexTmp[i - 1], s.vertexTmp[i]));
				valid = true;
			}
		}
		else if (parseLiteral(ptr, "usemtl ") && parseSpace(ptr)) // material name
		{
			int mati = s.materialHash.searchi(std::string(ptr));
			if (submesh != -1)
			{
				push(s.subMeshes[submesh].indices, s.indexTmp);
				s.indexTmp.clear();
				submesh = -1;
			}
			if (mati != -1)
			{
				auto& mat = s.materialHash.vec[mati];
				if (mat.submesh == -1)
				{
					mat.submesh = s.addSubMesh();
					s.subMeshes[mat.submesh].material = mat;
				}
				submesh = mat.submesh;
				s.indexTmp.clear();
			}
			valid = true;
		}
		else if (parseLiteral(ptr, "mtllib ") && parseSpace(ptr) && *ptr) // material library
		{
			if (dirName.size())
			{
				boost::algorithm::trim(std::string(ptr));
				std::string fileName = dirName + "/" + ptr;
				MemInputStream mtlIn(fileName.c_str());
				loadMtl(s, mtlIn, dirName);
				mtlIn.Close();
			}
			valid = true;
		}
		else if (
			parseLiteral(ptr, "vp ") ||         // parameter space vertex
			parseLiteral(ptr, "deg ") ||        // degree
			parseLiteral(ptr, "bmat ") ||       // basis matrix
			parseLiteral(ptr, "step ") ||       // step size
			parseLiteral(ptr, "cstype ") ||     // curve/surface type
			parseLiteral(ptr, "p ") ||          // point
			parseLiteral(ptr, "l ") ||          // line
			parseLiteral(ptr, "curv ") ||       // curve
			parseLiteral(ptr, "curv2 ") ||      // 2d curve
			parseLiteral(ptr, "surf ") ||       // surface
			parseLiteral(ptr, "parm ") ||       // curve/surface parameters
			parseLiteral(ptr, "trim ") ||       // curve/surface outer trimming loop
			parseLiteral(ptr, "hole ") ||       // curve/surface inner trimming loop
			parseLiteral(ptr, "scrv ") ||       // curve/surface special curve
			parseLiteral(ptr, "sp ") ||         // curve/surface special point
			parseLiteral(ptr, "end ") ||        // curve/surface end statement
			parseLiteral(ptr, "con ") ||        // surface connect
			parseLiteral(ptr, "g ") ||          // group name
			parseLiteral(ptr, "s ") ||          // smoothing group
			parseLiteral(ptr, "mg ") ||         // merging group
			parseLiteral(ptr, "o ") ||          // object name
			parseLiteral(ptr, "bevel ") ||      // bevel interpolation
			parseLiteral(ptr, "c_interp ") ||   // color interpolation
			parseLiteral(ptr, "d_interp ") ||   // dissolve interpolation
			parseLiteral(ptr, "lod ") ||        // level of detail
			parseLiteral(ptr, "shadow_obj ") || // shadow casting
			parseLiteral(ptr, "trace_obj ") ||  // ray tracing
			parseLiteral(ptr, "ctech ") ||      // curve approximation technique
			parseLiteral(ptr, "stech ") ||      // surface approximation technique
			parseLiteral(ptr, "g"))             // ???
		{
			valid = true;
		}

#if WAVEFRONT_DEBUG
		if (!valid)
			setError("Invalid line %d in Wavefront OBJ: '%s'!", lineNum, line);
#endif
	}

	// Flush remaining indices.

	if (submesh != -1)
		push(s.subMeshes[submesh].indices, s.indexTmp);
}

void compileobj(IInStream& in, FileOutputStream& a_Out)
{
	ImportState state;
	parse(state, in);

	std::vector<e_MeshPartLight> lights;
	std::vector<e_KernelMaterial> matData;
	matData.reserve(state.materialHash.vec.size());
	for (size_t i = 0; i < state.materialHash.vec.size(); i++)
	{
		Material M = state.materialHash.vec[i];
		e_KernelMaterial mat(M.Name.c_str());
		float f = 0.0f;
		if (M.IlluminationModel == 2)
		{
			diffuse d;
			d.m_reflectance = CreateTexture(M.textures[0].c_str(), Spectrum(M.diffuse.x, M.diffuse.y, M.diffuse.z));
			mat.bsdf.SetData(d);
		}
		else if (M.IlluminationModel == 5)
		{
			mat.bsdf.SetData(conductor(Spectrum(0.0f), Spectrum(1.0f)));
		}
		else if (M.IlluminationModel == 7)
		{
			dielectric d;
			d.m_eta = M.IndexOfRefraction;
			d.m_invEta = 1.0f / M.IndexOfRefraction;
			d.m_specularReflectance = CreateTexture(0, Spectrum(M.specular.x, M.specular.y, M.specular.z));
			d.m_specularTransmittance = CreateTexture(0, Spectrum(M.Tf.x, M.Tf.y, M.Tf.z));
			mat.bsdf.SetData(d);
		}
		else if (M.IlluminationModel == 9)
		{
			dielectric d;
			d.m_eta = M.IndexOfRefraction;
			d.m_invEta = 1.0f / M.IndexOfRefraction;
			d.m_specularReflectance = CreateTexture(0, Spectrum(0.0f));
			d.m_specularTransmittance = CreateTexture(0, Spectrum(M.Tf.x, M.Tf.y, M.Tf.z));
			mat.bsdf.SetData(d);
		}
		if (M.textures[TextureType_Displacement].size())
		{
			mat.SetHeightMap(M.textures[TextureType_Displacement].c_str());
		}

		if (length(M.emission))
			lights.push_back(e_MeshPartLight(M.Name, Spectrum(M.emission.x, M.emission.y, M.emission.z)));
		matData.push_back(mat);
	}

	unsigned int m_numTriangles = (unsigned int)state.numTriangles();
	unsigned int m_numVertices = (unsigned int)state.vertices.size();
	e_TriangleData* triData = new e_TriangleData[m_numTriangles];
	Vec3f p[3];
	Vec3f n[3];
	Vec3f ta[3];
	Vec3f bi[3];
	Vec2f t[3];
	std::vector<Vec3f> positions, normals, tangents, bitangents;
	positions.resize(m_numVertices); normals.resize(m_numVertices); tangents.resize(m_numVertices); bitangents.resize(m_numVertices);
	std::vector<Vec2f> texCoords;
	texCoords.resize(m_numVertices);
	for (size_t i = 0; i < m_numVertices; i++)
	{
		auto& v = state.vertices[i];
		positions[i] = v.p;
		texCoords[i] = v.t;
		normals[i] = Vec3f(0.0f);
		tangents[i] = Vec3f(0.0f);
		bitangents[i] = Vec3f(0.0f);
	}
	std::vector<Vec3i> indices;
	indices.resize(state.numTriangles() * 3);
	size_t k = 0;
	for (size_t i = 0; i < state.subMeshes.size(); i++)
		for (size_t j = 0; j < state.subMeshes[i].indices.size(); j++)
			indices[k++] = state.subMeshes[i].indices[j];
#ifdef EXT_TRI
	ComputeTangentSpace(&positions[0], &texCoords[0], (unsigned int*)&indices[0], m_numVertices, m_numTriangles, &normals[0], &tangents[0], &bitangents[0], true);
#endif

	AABB box = AABB::Identity();
	unsigned int triCount = 0;
	for (unsigned int submesh = 0; submesh < state.subMeshes.size(); submesh++)
	{
		int matIndex = state.materialHash.searchi(state.subMeshes[submesh].material.Name);
		if (matIndex == -1)
			throw std::runtime_error(__FUNCTION__);
		for (size_t t_idx = 0; t_idx < state.subMeshes[submesh].indices.size(); t_idx++)
		{
			Vec3i& idx = state.subMeshes[submesh].indices[t_idx];
			for (int j = 0; j < 3; j++)
			{
				int l = idx[j];
				p[j] = positions[l];
				box = box.Extend(p[j]);
#ifdef EXT_TRI
				t[j] = texCoords[l];
				ta[j] = normalize(tangents[l]);
				bi[j] = normalize(bitangents[l]);
				n[j] = normalize(normals[l]);
#endif
			}
			triData[triCount++] = e_TriangleData(p, (unsigned char)matIndex, t, n, ta, bi);
		}
	}

	a_Out << box;
	a_Out << (unsigned int)lights.size();
	if (lights.size())
		a_Out.Write(&lights[0], lights.size() * sizeof(e_MeshPartLight));
	a_Out << m_numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * m_numTriangles);
	a_Out << (unsigned int)matData.size();
	a_Out.Write(&matData[0], sizeof(e_KernelMaterial) * (unsigned int)matData.size());
	ConstructBVH(&positions[0], (unsigned int*)&indices[0], m_numVertices, m_numTriangles * 3, a_Out);
	delete[] triData;
}

}