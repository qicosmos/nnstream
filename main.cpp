#include <iostream>
#include <tuple>
#include <variant>
#include <vector>
#include "neural_net.hpp"

using namespace experimental;

void test_net1() {
	using sequential_net = neural_net<net_type::sequential>;
	sequential_net net;
	net | input(path = "/data", in_shape = shape(1, 32, 32)) | normalization()
		| conv_2d(kernel = (5), filters = 10, padding = same)
		| max_pooling_2d(kernel = (2), stride = (1))
		| sigmoid()
		| full_connected(out_units = 64)
		| soft_max();

	net.save("net.prototxt");
	std::string str = net.to_string();
	std::cout << str << std::endl;
}

void test_net2() {
	max_pooling_2d max_pooling(kernel = (2), stride = (2));

	using sequential_net = neural_net<net_type::sequential>;
	sequential_net net;
	net | input(path = "/data", in_shape = shape(1, 28, 28))
		| conv_2d(kernel = (5), filters = 32, padding = same, channel_pos = first)
		| relu()
		| max_pooling_2d(kernel = (2), stride = (2)) //max_pooling
		| conv_2d(kernel = (5), filters = 64)
		| relu()
		| max_pooling_2d(kernel = (2), stride = (2)) //max_pooling
		| full_connected(out_units = 1024)
		| relu()
		| dropout(Rate("0.5"))
		| full_connected(out_units = 10)
		| soft_max(loss = softmax_sparse_cross_entropy);

	net.save("net.txt");
	std::string str = net.to_string();
}

int main() {
	test_net1();
	test_net2();

	return 0;
}