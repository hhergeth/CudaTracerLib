#pragma once
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <functional>

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
	const IBaseParameterConstraint* constraint;
	ITracerParameter(const IBaseParameterConstraint* constraint)
		: constraint(constraint)
	{

	}
public:
	virtual const IBaseParameterConstraint* getConstraint() const { return constraint; }
	template<typename T> const IParameterConstraint<T>* getConstraint() const { return dynamic_cast<const IParameterConstraint<T>*>(constraint); }
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
		if (getConstraint<T>()->isValid(val))
			value = val;
		else;
	}
	const T& getDefaultValue() const { defaultValue; }
};

template<typename T> TracerParameter<T>* CreateInterval(const T& val, const T& min, const T& max)
{
	return new TracerParameter<T>(val, new IntervalParameterConstraint<T>(min, max));
}

template<typename T, typename... Ts> TracerParameter<T>* CreateSet(const T& val, Ts&&... il)
{
	return new TracerParameter<T>(val, new SetParameterConstraint<T>(il...));
}

class TracerParameterCollection
{
	std::map<std::string, ITracerParameter*> parameter;
	template<typename T, typename... Ts> void add(TracerParameter<T>* a, const std::string& name, Ts&&... rest)
	{
		add(a, name);
		add(rest...);
	}
	template<typename T> void add(TracerParameter<T>* a, const std::string& name)
	{
		parameter[name] = a;
	}
	std::string lastName;
public:
	TracerParameterCollection()
		: lastName("")
	{

	}
	friend TracerParameterCollection& operator<<(TracerParameterCollection& lhs, const std::string& name);
	friend TracerParameterCollection& operator<<(TracerParameterCollection& lhs, ITracerParameter* para);
	template<typename... Ts> TracerParameterCollection(Ts&&... rest)
	{
		add(rest...);
	}
	void iterate(std::function<void(const std::string&, ITracerParameter*)>& f) const
	{
		for (auto& i : parameter)
		{
			f(i.first, i.second);
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
};

TracerParameterCollection& operator<<(TracerParameterCollection& lhs, const std::string& name)
{
	if (lhs.lastName.size() != 0)
		throw std::runtime_error("Invalid state of TracerParameterCollection. Shift Parameter after name!");
	lhs.lastName = name;
	return lhs;
}

TracerParameterCollection& operator<<(TracerParameterCollection& lhs, ITracerParameter* para)
{
	if (lhs.lastName.size() == 0)
		throw std::runtime_error("Invalid state of TracerParameterCollection. Shift Parameter after name!");
	lhs.parameter[lhs.lastName] = para;
	lhs.lastName = "";
	return lhs;
}

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
	std::string lastVal;
public:
	TracerArguments()
		: lastVal("")
	{

	}
	void addArgument(const std::string& name, const std::string& value)
	{
		arguments[name] = value;
	}
	friend TracerArguments& operator<<(TracerArguments& lhs, const std::string& val);

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

TracerArguments& operator<<(TracerArguments& lhs, const std::string& val)
{
	if (lhs.lastVal.size() == 0)
		lhs.lastVal = val;
	else
	{
		lhs.addArgument(lhs.lastVal, val);
		lhs.lastVal = "";
	}
	return lhs;
}

}