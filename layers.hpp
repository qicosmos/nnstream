#pragma once
#include <sstream>
#include <boost/parameter.hpp>

namespace experimental {
	enum class net_type {
		sequential,
		recurrent
	};
	enum class pool_type {
		max,
		avg
	};
	constexpr pool_type max = pool_type::max;
	constexpr pool_type avg = pool_type::avg;

	enum class dimension_type
	{
		d1,
		d2,
		d3
	};

	constexpr dimension_type d1 = dimension_type::d1;
	constexpr dimension_type d2 = dimension_type::d2;
	constexpr dimension_type d3 = dimension_type::d3;

	enum class active_type {
		relu,
		tanh,
		sigmoid,
		elu,
	};

	enum class conv_padding {
		same,
		valid
	};
	constexpr const conv_padding same = conv_padding::same;
	constexpr const conv_padding valid = conv_padding::valid;

	enum class channel_sequence {
		first,
		last
	};
	constexpr const channel_sequence first = channel_sequence::first;
	constexpr const channel_sequence last = channel_sequence::last;

	enum class loss_type {
		svm, 
		sparse_cross_entropy, 
		softmax_sparse_cross_entropy, 
		eculidean,
		distance
	};
	constexpr const loss_type svm = loss_type::svm;
	constexpr const loss_type softmax_sparse_cross_entropy = loss_type::softmax_sparse_cross_entropy;
	constexpr const loss_type sparse_cross_entropy = loss_type::sparse_cross_entropy;
	constexpr const loss_type eculidean = loss_type::eculidean;
	constexpr const loss_type distance = loss_type::distance;

	struct shape {
		shape() = default;
		shape(size_t h) : h_(h), w_(h) {}
		shape(size_t h, size_t w) : h_(h), w_(w) {}
		shape(size_t c, size_t h, size_t w) :c_(c), h_(h), w_(w) {}

		std::string to_string() {
			std::stringstream ss;
			ss << "shape {\r\n";
			ss << "    dim: " << c_ << "\r\n";
			ss << "    dim: " << h_ << "\r\n";
			ss << "    dim: " << w_ << "\r\n";
			ss << "  }";
			return ss.str();
		}
		size_t c_;
		size_t h_;
		size_t w_;
	};
	
	//[batch, height, width, channels]
	struct cov_stride {
		cov_stride() = default;
		cov_stride(size_t h) : h_(h), w_(h) {}
		cov_stride(size_t h, size_t w) : h_(h), w_(w) {}
		cov_stride(size_t b, size_t c, size_t h, size_t w) : b_(h), c_(c), h_(h), w_(w) {}
		size_t b_;
		size_t c_;
		size_t h_;
		size_t w_;
	};

	struct conv_kernel {
		conv_kernel() = default;
		conv_kernel(size_t h) : h_(h), w_(h) {
			std::cout << 1 << std::endl;
		}
		conv_kernel(size_t h, size_t w) : h_(h), w_(w) {
			std::cout << 2 << std::endl;
		}
		
		size_t h_;
		size_t w_;
	};
	using pool_kernel = conv_kernel;

	struct conv_dilation_rate {
		conv_dilation_rate() {}
		conv_dilation_rate(size_t x):x_(x), y_(x) {}
		conv_dilation_rate(size_t x, size_t y) :x_(x), y_(y) {}
		size_t x_; //1d
		size_t y_; //2d
	};

	BOOST_PARAMETER_KEYWORD(tag, name);
	BOOST_PARAMETER_KEYWORD(tag, filters);
	BOOST_PARAMETER_KEYWORD(tag, padding);
	BOOST_PARAMETER_KEYWORD(tag, channel_pos);
	BOOST_PARAMETER_KEYWORD(tag, kernel);
	BOOST_PARAMETER_KEYWORD(tag, stride);
	BOOST_PARAMETER_KEYWORD(tag, dilation_rate);
	BOOST_PARAMETER_KEYWORD(tag, rate);
	BOOST_PARAMETER_KEYWORD(tag, use_bias);
	BOOST_PARAMETER_KEYWORD(tag, out_units);
	BOOST_PARAMETER_KEYWORD(tag, act_function);
	BOOST_PARAMETER_KEYWORD(tag, path);
	BOOST_PARAMETER_KEYWORD(tag, in_shape);

	using Tname = decltype(name);
	using Tfilters = decltype(filters);
	using Tpadding = decltype(padding);
	using Tchannel_pos = decltype(channel_pos);
	using Tkernel = decltype(kernel);
	using Tstride = decltype(stride);
	using Tdilation_rate = decltype(dilation_rate);
	using Tuse_bias = decltype(use_bias);
	using Tout_units = decltype(out_units);
	using Tact_function = decltype(act_function);
	using Tpath = decltype(path);
	using Tshape = decltype(in_shape);
	using Trate = decltype(rate);

	template <typename T, typename... Args>
	struct has_type;

	template <typename T, typename... Us>
	struct has_type<T, Us...> : std::disjunction<std::is_convertible<T, Us>...> {};

	class input {
	public:
		input() = default;
		input(std::string path) :path_(path) {}
		input(shape sp) :sp_(sp) {}
		input(std::string path, shape sp) :path_(path), sp_(sp) {}

		template <typename... ArgPack>
		input(ArgPack&&... args) {
			static_assert(has_type<Tpath, boost::parameter::keyword<ArgPack::key_type>...>::value, "lack of path");
			static_assert(has_type<Tshape, boost::parameter::keyword<ArgPack::key_type>...>::value, "lack of shape");
			(check(std::forward<ArgPack>(args)), ...);
		}

		std::string to_string() {
			std::stringstream ss;
			ss << "decode_ofrecord_conf {\r\n";
			ss << "  data_dir: \"" << path_<<"\"\r\n";
			ss << "  "<<sp_.to_string() << "\r\n";
			ss << "}\r\n";
			return ss.str();
		}

	private:
		template<typename ArgPack>
		constexpr void check(ArgPack&& arg) {
			using T = boost::parameter::keyword<ArgPack::key_type>;

			if constexpr(std::is_convertible_v<T, Tpath>) {
				path_ = arg[path];
			}
			else if constexpr(std::is_convertible_v<T, Tshape>) {
				sp_ = arg[in_shape];
			}
			else {
				static_assert(false);
			}
		}

	private:
		
		std::string path_;
		shape sp_;
	};

	struct normalization {
		std::string to_string() {
			return "";
		}
		input data;
	};

	template<dimension_type dt>
	class conv{
	public:
		template <typename... ArgPack>
		conv(ArgPack&&... args) {
			static_assert(has_type<Tkernel, boost::parameter::keyword<ArgPack::key_type>...>::value, "lack of kernel");
			static_assert(has_type<Tfilters, boost::parameter::keyword<ArgPack::key_type>...>::value, "lack of filters");
			(check(std::forward<ArgPack>(args)), ...);
		}

		std::string to_string() {
			std::stringstream ss;
			ss << "conv_2d_conf {\r\n";
			ss << "  filters: " << filter_num_ << "\r\n";
			ss << "  padding: " << (pd_== conv_padding::same?"SAME":"VALID") << "\r\n";
			ss << "  data_format: " << (pos_ == channel_sequence::first ? "channels_first" : "channels_last") << "\r\n";
			ss << "  kernel_size: " << kn_.h_ << "\r\n";
			ss << "  kernel_size: " << kn_.w_ << "\r\n";
			ss << "  strides: " << st_.h_ << "\r\n";
			ss << "  strides: " << st_.h_ << "\r\n";
			ss << "  dilation_rate: "<< rate_.x_<< "\r\n";
			ss << "  dilation_rate: " << rate_.y_ << "\r\n";
			ss << "  use_bias: " << use_bias_ << "\r\n";
			ss << "}";
			return ss.str();
		}

	private:
		template<typename ArgPack>
		constexpr void check(ArgPack&& arg) {
			using T = boost::parameter::keyword<ArgPack::key_type>;

			if constexpr(std::is_convertible_v<T, Tname>) {
				name_ = arg[name];
			}
			else if constexpr(std::is_convertible_v<T, Tfilters>) {
				filter_num_ = arg[filters];
			}
			else if constexpr(std::is_convertible_v<T, Tpadding>) {
				pd_ = arg[padding];
			}
			else if constexpr(std::is_convertible_v<T, Tchannel_pos>) {
				pos_ = arg[channel_pos];
			}
			else if constexpr(std::is_convertible_v<T, Tkernel>) {
				kn_ = arg[kernel];
			}
			else if constexpr(std::is_convertible_v<T, Tstride>) {
				st_ = arg[stride];
			}
			else if constexpr(std::is_convertible_v<T, Tdilation_rate>) {
				rate_ = arg[dilation_rate];
			}
			else if constexpr(std::is_convertible_v<T, Tuse_bias>) {
				use_bias_ = arg[use_bias];
			}
			else {
				static_assert(false);
			}
		}

		std::string name_;
		input data_;
		size_t filter_num_;
		conv_padding pd_;
		channel_sequence pos_;
		conv_kernel kn_;
		cov_stride st_ = { 1,1 };
		conv_dilation_rate rate_{ 1,1 };
		bool use_bias_ = true;
	};

	template<pool_type pt, dimension_type dt>
	class pooling {
	public:
		pooling() = default;
		template <typename... ArgPack>
		pooling(ArgPack... args) {
			static_assert(has_type<Tkernel, boost::parameter::keyword<ArgPack::key_type>...>::value, "lack of kernel");
			static_assert(has_type<Tstride, boost::parameter::keyword<ArgPack::key_type>...>::value, "lack of stride");
			(check(std::forward<ArgPack>(args)), ...);
		}

		std::string to_string() {
			return "";
		}

	private:
		template<typename ArgPack>
		constexpr void check(ArgPack arg) {
			using T = boost::parameter::keyword<ArgPack::key_type>;

			if constexpr(std::is_convertible_v<T, Tname>) {
				name_ = arg[name];
			}
			else if constexpr(std::is_convertible_v<T, Tpadding>) {
				pd_ = arg[padding];
			}
			else if constexpr(std::is_convertible_v<T, Tkernel>) {
				kn_ = arg[kernel];
			}
			else if constexpr(std::is_convertible_v<T, Tstride>) {
				st_ = arg[stride];
			}
			else {
				static_assert(false);
			}
		}

		std::string name_;
		conv_padding pd_;
		pool_kernel kn_;
		cov_stride st_;
	};

	template<active_type at>
	class activation {
	public:
		activation() = default;
		
		std::string to_string() {
			return "";
		}

	private:
		std::string name_;
	};

	template<typename T, size_t N>
	constexpr bool is_num() {
		for (size_t i = 2; i < N; i++)
		{
			char ch = T::string[i];
			if (T::string[i] > '9' || T::string[i] < '0')
				return false;
		}

		return true;
	}

	template<typename T>
	constexpr bool check_rate() {
		constexpr auto N = T::value;
		static_assert(T::string[0] == '0', "should less than 1");
		static_assert(T::string[1] == '.', "the rate is invalid");
		static_assert(is_num<T, N>(), "the rate must be number");
		return true;
	}

	template<>
	constexpr bool check_rate<str<'1', '.', '0'>>() { return true; }

	template<>
	constexpr bool check_rate<str<'1'>>() { return true; }

	struct dropout {
		dropout() = default;
		template <typename... ArgPack>
		dropout(ArgPack... args) {
			(check(std::forward<ArgPack>(args)), ...);
		}
		//template<typename T, typename=std::enable_if_t<check_rat(T{})>>
		//dropout(T t) : rate_(t.string) {
		//}

		std::string to_string() {
			return "";
		}

		template<typename ArgPack>
		constexpr void check(ArgPack arg) {
			//using T = boost::parameter::keyword<ArgPack::key_type>;
			constexpr bool r = check_rate<ArgPack>();
			if constexpr(r) {
				rate_ = arg.string;
			}
			else {
				static_assert(false);
			}
		}

		std::string rate_; //0.75
	};

	class full_connected {
	public:
		full_connected() = default;
		template <typename... ArgPack>
		full_connected(ArgPack&&... args) {
			static_assert(has_type<Tout_units, boost::parameter::keyword<ArgPack::key_type>...>::value, "lack of out_units");
			(check(std::forward<ArgPack>(args)), ...);
		}

		std::string to_string() {
			return "";
		}

	private:
		template<typename ArgPack>
		constexpr void check(ArgPack&& arg) {
			using T = boost::parameter::keyword<ArgPack::key_type>;

			if constexpr(std::is_convertible_v<T, Tname>) {
				name_ = arg[name];
			}
			else if constexpr(std::is_convertible_v<T, Tout_units>) {
				units_ = arg[out_units];
			}
			else if constexpr(std::is_convertible_v<T, Tuse_bias>) {
				use_bias_ = arg[use_bias];
			}
			else if constexpr(std::is_convertible_v<T, Tact_function>) {
				at_ = arg[act_function];
			}
			else {
				static_assert(false);
			}
		}

		std::string name_;
		size_t units_;
		bool use_bias_ = true;
		active_type at_ = active_type::relu;
	};

	BOOST_PARAMETER_KEYWORD(tag, loss);
	using Tloss = decltype(loss);
	class soft_max {
	public:
		soft_max() = default;
		soft_max(loss_type loss) : loss_(loss) {}
		template <typename... ArgPack>
		soft_max(ArgPack&&... args) {
			(check(std::forward<ArgPack>(args)), ...);
		}

		std::string to_string() {
			return "";
		}
	private:
		template<typename ArgPack>
		constexpr void check(ArgPack arg) {

			using T = boost::parameter::keyword<ArgPack::key_type>;

			if constexpr(std::is_convertible_v<T, Tname>) {
				name = arg[name];
			}
			else if constexpr(std::is_convertible_v<T, Tloss>) {
				loss_ = arg[loss];
			}
			else {
				static_assert(false);
			}
		}
		//softmax_label label;
		loss_type loss_ = loss_type::softmax_sparse_cross_entropy;
	};

	using conv_1d = conv<d1>;
	using conv_2d = conv<d2>;
	using conv_3d = conv<d3>;
	using max_pooling_1d = pooling<max, d1>;
	using max_pooling_2d = pooling<max, d2>;
	using max_pooling_3d = pooling<max, d3>;
	using avg_pooling_1d = pooling<avg, d1>;
	using avg_pooling_2d = pooling<avg, d2>;
	using avg_pooling_3d = pooling<avg, d3>;

	using relu = activation<active_type::relu>;
	using elu = activation<active_type::elu>;
	using sigmoid = activation<active_type::sigmoid>;
	using tanh = activation<active_type::tanh>;


	using Layer = std::variant<
		input, normalization, conv_1d, conv_2d, conv_3d, 
		max_pooling_1d, max_pooling_2d, max_pooling_3d, avg_pooling_1d, avg_pooling_2d, avg_pooling_3d,
		relu, elu, sigmoid, tanh,
		dropout, full_connected, soft_max
	>;
}