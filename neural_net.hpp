#pragma once
#include <array>
#include <vector>
#include <string>
#include "string_literal.hpp"
#include "layers.hpp"

namespace experimental {
	template<net_type nt>
	struct neural_net {

		void add(Layer layer) {
			layers_.push_back(std::move(layer));
		}

		std::string to_string() {
			std::string str;
			for (auto&& i : layers_) {
				std::visit([&str](auto&& arg) {
					str += arg.to_string();
				}, i);
			}
			return str;
		}

		void save(std::string path) {

		}

		std::vector<Layer> layers_;
	};

	template <net_type nt, typename Layer>
	neural_net<nt>& operator|(neural_net<nt> &n, Layer &&l) {
		n.add(std::forward<Layer>(l));
		return n;
	}
}