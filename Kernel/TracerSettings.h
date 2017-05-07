#pragma once
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <functional>
#include <boost/mpl/string.hpp>
#include <memory>
#include <algorithm>
#include <Base/EnumConverter.h>

namespace CudaTracerLib {

class IBaseParameterConstraint
{
public:
	virtual  ~IBaseParameterConstraint()
	{

	}
	virtual std::string Serialize() const = 0;
};

template<typename T> class IParameterConstraint : public IBaseParameterConstraint
{
public:
	virtual bool isValid(const T& obj) const = 0;
};

template<typename T> class IntervalParameterConstraint : public IParameterConstraint<T>
{
	T min, max;
public:
	IntervalParameterConstraint(const T& min, const T& max)
		: min(min), max(max)
	{

	}
	const T& getMin() const { return min; }
	const T& getMax() const { return max; }
	virtual std::string Serialize() const
	{
		std::ostringstream str;
		str << "Interval = {" << min << ", " << max << "}";
		return str.str();
	}
	virtual bool isValid(const T& obj) const
	{
		return min <= obj && obj <= max;
	}
};

template<typename T> class SetParameterConstraint : public IParameterConstraint<T>
{
	std::vector<T> elements;
public:
	SetParameterConstraint(std::initializer_list<T> vals)
		: elements(vals)
	{

	}
	SetParameterConstraint(const std::vector<T>& elements)
		: elements(elements)
	{

	}
	const std::vector<T>& getElements() const { return elements; }
	virtual std::string Serialize() const
	{
		std::ostringstream str;
		str << "Set = {";
		for (size_t i = 0; i < elements.size(); i++)
			str << elements[i] << (i ? ", " : "");
		return str.str();
	}
	virtual bool isValid(const T& obj) const
	{
		return std::find(elements.begin(), elements.end(), obj) != elements.end();
	}
};

template<typename T> class TracerParameter;
class ITracerParameter
{
protected:
	const IBaseParameterConstraint* constraint;
	ITracerParameter(const IBaseParameterConstraint* constraint)
		: constraint(constraint)
	{

	}
public:
	virtual ~ITracerParameter()
	{
		delete constraint;
	}
	virtual const IBaseParameterConstraint* getConstraint() const { return constraint; }
	template<typename T> const IParameterConstraint<T>* getConstraint() const { return dynamic_cast<const IParameterConstraint<T>*>(constraint); }
	template<typename T> TracerParameter<T>* As() { return dynamic_cast<TracerParameter<T>*>(this); }
	template<typename T> const TracerParameter<T>* As() const { return dynamic_cast<const TracerParameter<T>*>(this); }
	template<typename T> bool isOfType() const { return As<T>() != 0; }
};

template<typename T> class TracerParameter : public ITracerParameter
{
protected:
	T value;
	T defaultValue;
public:
	TracerParameter(const T& val, const IParameterConstraint<T>* cons)
		: ITracerParameter(cons), value(val), defaultValue(val)
	{

	}
	const T& getValue() const { return value; }
	void setValue(const T& val)
	{
		if (getConstraint<T>() && getConstraint<T>()->isValid(val))
			value = val;
		else;
	}
	const T& getDefaultValue() const { return defaultValue; }
};

class IEnumTracerParameter
{
public:
	virtual ~IEnumTracerParameter()
	{

	}
	virtual const std::string& getStringValue() const = 0;
	virtual void setStringValue(const std::string& strVal) = 0;
	virtual std::vector<std::string> getStringValues() const = 0;
};

template<typename T> class EnumTracerParameter : public TracerParameter<T>, public IEnumTracerParameter
{
	IParameterConstraint<T>* createConstraint() const
	{
		std::vector<T> e;
		EnumConverter<T>::enumerateEntries([&](T val, const std::string& strVal){ e.push_back(val); });
		auto c = new SetParameterConstraint<T>(e);
		return c;
	}
public:
	EnumTracerParameter(const T& val)
		: TracerParameter<T>(val, createConstraint())
	{

	}
	virtual ~EnumTracerParameter()
	{
	}

	virtual const std::string& getStringValue() const
	{
		return EnumConverter<T>::ToString(TracerParameter<T>::getValue());
	}

	virtual void setStringValue(const std::string& strVal)
	{
		T val = EnumConverter<T>::FromString(strVal);
		TracerParameter<T>::setValue(val);
	}

	virtual std::vector<std::string> getStringValues() const
	{
		std::vector<std::string> e;
		EnumConverter<T>::enumerateEntries([&](T val, const std::string& strVal){ e.push_back(strVal); });
		return e;
	}
};

template<typename T> TracerParameter<T>* CreateParameter(const T& val)
{
	return new TracerParameter<T>(val, 0);
}

template<typename T> TracerParameter<T>* CreateInterval(const T& val, const T& min, const T& max)
{
	return new TracerParameter<T>(val, new IntervalParameterConstraint<T>(min, max));
}

template<typename T> TracerParameter<T>* CreateSet(const T& val, std::initializer_list<T> vals)
{
	return new TracerParameter<T>(val, new SetParameterConstraint<T>(vals));
}

inline TracerParameter<bool>* CreateSetBool(bool val)
{
	return new TracerParameter<bool>(val, new SetParameterConstraint<bool>({ true, false }));
}

//typically this would be the ideal place to use a type as key but sadly nvcc will produce tons of warnings for multichar litearls (CUDA 7.5)
//and turning this off by passing arguments to cudafe doesn't work
template<typename T> struct TracerParameterKey
{
	const std::string name;

	TracerParameterKey(const std::string& name)
		: name(name)
	{

	}

	operator std::string () const
	{
		return name;
	}
};

#define PARAMETER_KEY(type, name) \
	struct KEY_##name : public TracerParameterKey<type> \
	{ \
		KEY_##name() \
			: TracerParameterKey(#name) \
		{ \
		} \
	};

class TracerParameterCollection
{
	std::map<std::string, ITracerParameter*> parameter;
	std::map<std::string, TracerParameterCollection*> collections;
	template<typename T, typename... Ts> void add(TracerParameter<T>* a, const std::string& name, Ts&&... rest)
	{
		add(a, name);
		add(rest...);
	}
	template<typename T> void add(TracerParameter<T>* a, const std::string& name)
	{
		parameter[name] = a;
	}
public:
	TracerParameterCollection()
	{

	}
	~TracerParameterCollection()
	{
		for (auto& i : parameter)
			delete i.second;
	}
	void iterate(const std::function<void(const std::string&, ITracerParameter*)>& f_para, const std::function<void(const std::string&, TracerParameterCollection*)>& f_coll) const
	{
		for (auto& i : parameter)
		{
			f_para(i.first, i.second);
		}
		for (auto& i : collections)
		{
			f_coll(i.first, i.second);
		}
	}
	ITracerParameter* operator[](const std::string& name) const
	{
		auto it = parameter.find(name);
		if (it == parameter.end())
			return 0;
		else return it->second;
	}
	template<typename T> TracerParameter<T>* get(const std::string& name) const
	{
		return dynamic_cast<TracerParameter<T>*>(operator[](name));
	}
	template<typename T> const T& getValue(const std::string& name) const
	{
		TracerParameter<T>* p = get<T>(name);
		if (p)
			return p->getValue();
		else throw std::runtime_error("Invalid access to parameter value!");
	}
	template<typename T> const T& getValue(const TracerParameterKey<T>& key) const
	{
		return getValue<T>(key.operator std::string());
	}
	template<typename T> void setValue(const TracerParameterKey<T>& key, const T& val)
	{
		TracerParameter<T>* p = get<T>(key.operator std::string());
		if (p)
			p->setValue(val);
		else throw std::runtime_error("Invalid access to parameter value!");
	}

	void addChildParameterCollection(const std::string& name, TracerParameterCollection* col)
	{
		if (collections.find(name) != collections.end())
			throw std::runtime_error("Trying to add collection with identical name!");
		collections[name] = col;
	}
	TracerParameterCollection* getCollection(const std::string& name)
	{
		auto it = collections.find(name);
		if (it == collections.end())
			return 0;
		else return it->second;
	}

	friend class InitHelper;
	template<typename T> class InitHelper
	{
		std::string lastName;
		TracerParameterCollection& settings;
		int state;// 1 -> set name waiting for property, 2 -> all done ready to be thrown away
	public:
		InitHelper(TracerParameterCollection& set, const std::string& name)
			: lastName(name), settings(set), state(1)
		{

		}

		~InitHelper()
		{
			if (state == 1)
				printf("Invalid initialization of tracer settings collection, forgot to pass a parameter? \nLast name was %s.", lastName.c_str());//invalid syntax
		}

		TracerParameterCollection& operator<<(ITracerParameter* para)
		{
			state = 2;
			settings.parameter[lastName] = para;
			return settings;
		}

		TracerParameterCollection& operator<<(T val)
		{
			state = 2;
			settings.parameter[lastName] = new EnumTracerParameter<T>(val);
			return settings;
		}
	};

	template<typename T> InitHelper<T> operator<<(const TracerParameterKey<T>& key)
	{
		return InitHelper<T>(*this, key);
	}

	/*template<typename T> TracerParameterCollection& add(const TracerParameterKey<T>& key, T val)
	{
		add(new EnumTracerParameter<T>(), key);
		return *this;
	}*/
};

class TracerArguments
{
	std::map<std::string, std::string> arguments;//name -> value
	void setParameterFromArgument(ITracerParameter* para, const std::string& value) const
	{
#define SET(type, conversion) if (para->isOfType<type>()) para->As<type>()->setValue(conversion);
		SET(bool, value[0] == 'T' || value[0] == 't' || value[0] == '1');
		SET(int, std::atoi(value.c_str()));
		SET(unsigned int, (unsigned int)std::atoll(value.c_str()));
		SET(float, (float)std::atof(value.c_str()));
#undef SET
	}
public:
	TracerArguments()
	{

	}
	void addArgument(const std::string& name, const std::string& value)
	{
		arguments[name] = value;
	}

	void setToParameters(TracerParameterCollection* parameters) const
	{
		for (auto& i : arguments)
		{
			ITracerParameter* para = parameters->operator[](i.first);
			if (para != 0)
				setParameterFromArgument(para, i.second);
		}
	}
};

}
