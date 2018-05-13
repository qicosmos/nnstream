#pragma once

//CONSTEXPR STRING

template<size_t N, template<size_t...> class func, size_t... indices> struct do_foreach_range
{
	using type = typename do_foreach_range<N - 1, func, N - 1, indices...>::type;
};

template<template<size_t...> class func, size_t... indices> struct do_foreach_range<0, func, indices...>
{
	using type = typename func<indices...>::type;
};

template<char... cs> struct str
{
	static constexpr const char string[sizeof...(cs) + 1] = { cs..., '\0' };
	static constexpr const size_t value = sizeof...(cs);
};

template<char... cs> constexpr const char str<cs...>::string[];

template<typename str_type> struct builder//str_type is static class with string literal
{
	template<size_t... indices> struct do_foreach//will be func
	{
		//want to fetch the char of each index
		using type = str<str_type{}.chars[indices]...>;
	};
};

#define STR(s) #s
#define Rate(string_literal) CSTRING(STR(string_literal))

#define CSTRING(string_literal) []{ \
    struct const_str { const char* chars = string_literal; }; \
    return do_foreach_range<sizeof(string_literal) - 1, builder<const_str>::do_foreach>::type{}; }()
