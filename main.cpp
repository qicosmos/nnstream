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

	std::string str = net.to_string();
	net.save("net.prototxt");

	std::cout << str << std::endl;
}

void test_net2() {
	max_pooling_2d max_pooling(kernel = (2), stride = (2));

	using sequential_net = neural_net<net_type::sequential>;
	sequential_net net;
	net | input(path = "/data", in_shape = shape(1, 28, 28))
		| conv_2d(kernel = (5), filters = 32, padding = same, channel_pos = first)
		| relu()
		| max_pooling_2d(kernel = (2), stride = (2))
		| conv_2d(kernel = (5), filters = 64)
		| relu()
		| max_pooling_2d(kernel = (2, 2), stride = (2, 2))
		| full_connected(out_units = 1024)
		| relu()
		| dropout(Rate(0.5))
		| full_connected(out_units = 10)
		| soft_max(loss = softmax_sparse_cross_entropy);

	net.save("net.txt");
	std::string str = net.to_string();
}

void check_params() {
	conv_2d conv1(kernel = (5), filters = 64); //ok
	//conv_2d conv2(filters = 32, padding = same);  //error: lack of kernel
	//conv_2d conv3(kernel = (5), use_bias = true); //error: lack of filters
	//conv_2d conv4(kernel = (5), filters = 64£¬ out_units = 10); //error: invalid argument

	//check_rate(Rate("0.2"));
	dropout p(Rate(0.5)); //ok
	//dropout p1(Rate(0.5a));//error: the rate must be number
	//dropout p2(Rate(1.5));//error: the rate should less than 1
	//dropout p3(Rate(00));//error: the rate is invalid
}

int main() {
	test_net1();
	test_net2();
	check_params();
	return 0;
}