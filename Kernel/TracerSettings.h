#pragma once
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <functional>
#include <boost/mpl/string.hpp>
#include <memory>

namespace CudaTracerLib {

class IBaseParameterConstraint
{
public:
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
	template<typename... Us> SetParameterConstraint(Us... il)
		: elements(il...)
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
		return std::find(elements.begin(), elements.end(), obj) != elements.begin();
	}
};

template<typename T> class TracerParameter;
class ITracerParameter
{
protected:
	std::unique_ptr<const IBaseParameterConstraint> constraint;
	ITracerParameter(const IBaseParameterConstraint* constraint)
		: constraint(constraint)
	{

	}
public:
	virtual const IBaseParameterConstraint* getConstraint() const { return constraint.get(); }
	template<typename T> const IParameterConstraint<T>* getConstraint() const { return dynamic_cast<const IParameterConstraint<T>*>(constraint.get()); }
	template<typename T> TracerParameter<T>* As() { return dynamic_cast<TracerParameter<T>*>(this); }
	template<typename T> const TracerParameter<T>* As() const { return dynamic_cast<const TracerParameter<T>*>(this); }
	template<typename T> bool isOfType() const { return As<T>() != 0; }
};

template<typename T> class TracerParameter : public ITracerParameter
{
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

template<typename T> TracerParameter<T>* CreateParameter(const T& val)
{
	return new TracerParameter<T>(val, 0);
}

template<typename T> TracerParameter<T>* CreateInterval(const T& val, const T& min, const T& max)
{
	return new TracerParameter<T>(val, new IntervalParameterConstraint<T>(min, max));
}

template<typename T, typename... Ts> TracerParameter<T>* CreateSet(const T& val, Ts&&... il)
{
	return new TracerParameter<T>(val, new SetParameterConstraint<T>(il...));
}

inline TracerParameter<bool>* CreateSetBool(bool val)
{
	return new TracerParameter<bool>(val, new SetParameterConstraint<bool>(true, false));
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
	std::map<std::string, std::unique_ptr<ITracerParameter>> parameter;
	template<typename T, typename... Ts> void add(TracerParameter<T>* a, const std::string& name, Ts&&... rest)
	{
		add(a, name);
		add(rest...);
	}
	template<typename T> void add(TracerParameter<T>* a, const std::string& name)
	{
		parameter[name] = std::unique_ptr<ITracerParameter>(a);
	}
	std::string lastName;
public:
	template<typename... Ts> TracerParameterCollection(Ts&&... rest)
		: lastName("")
	{
		add(rest...);
	}
	template<> TracerParameterCollection()
		: lastName("")
	{

	}
	void iterate(std::function<void(const std::string&, ITracerParameter*)>& f) const
	{
		for (auto& i : parameter)
		{
			f(i.first, i.second.get());
		}
	}
	ITracerParameter* operator[](const std::string& name) const
	{
		auto it = parameter.find(name);
		if (it == parameter.end())
			return 0;
		else return it->second.get();
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
	friend TracerParameterCollection& operator<<(TracerParameterCollection& lhs, const std::string& name);
	friend TracerParameterCollection& operator<<(TracerParameterCollection& lhs, ITracerParameter* para);
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