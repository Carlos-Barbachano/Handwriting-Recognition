// Handwriting_Recognition.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <fstream>

std::vector<std::pair<cv::Mat, int>> loadBinary(const std::string& datapath, const std::string& labelpath);
int reverseInt(int i);

int main()
{


	std::vector<std::pair<cv::Mat, int>> trainImages = loadBinary("C:\\Users\\Alex\\Desktop\\Datasets\\MNIST\\Train_Images\\train-images.idx3-ubyte", "C:\\Users\\Alex\\Desktop\\Datasets\\MNIST\\Train_Images\\train-labels.idx1-ubyte");

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::POLY);
	svm->setDegree(3);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));


	cv::Mat trainingMat;
	cv::Mat labelMat;

	cv::Mat images;
	cv::Mat labels;

	for (int i = 0; i < trainImages.size(); i++)
	{

		cv::Mat image = (trainImages[i].first.reshape(1, 1));
		int label = trainImages[i].second;
		images.push_back(image);
		labels.push_back(label);

	}

	std::cout << "Training..." << std::endl;

	svm->train(images, cv::ml::ROW_SAMPLE, labels);

	std::vector<std::pair<cv::Mat, int>> test_images = loadBinary("C:\\Users\\Alex\\Desktop\\Datasets\\MNIST\\Test_Images\\t10k-images.idx3-ubyte", "C:\\Users\\Alex\\Desktop\\Datasets\\MNIST\\Test_Images\\t10k-labels.idx1-ubyte");

	int loop_index = 0;
	int correct = 0;

	while (loop_index < test_images.size()) {

		float prediction_value = svm->predict(test_images[loop_index].first.reshape(1, 1));

		int label_value = test_images[loop_index].second;

		if (label_value == prediction_value) {
			std::cout << "Correct." << std::endl;
			cv::imshow("test image", test_images[loop_index].first);
			std::cout << "label: " << label_value << std::endl;
			std::cout << "prediction: " << prediction_value << "\n\n" << std::endl;
			correct++;

		}
		else {
			std::cout << "Incorrect." << std::endl;
			cv::imshow("test image", test_images[loop_index].first);
			std::cout << "label: " << label_value << std::endl;
			std::cout << "prediction: " << prediction_value << "\n\n" << std::endl;
		}

		//		cv::waitKey(0);

		loop_index++;

	}

	std::cout << "Percentage correct: (" << correct << " / " << loop_index << ") = " << (correct / (float)loop_index) * 100 << "%" << std::endl;

}


std::vector<std::pair<cv::Mat, int>> loadBinary(const std::string & datapath, const std::string & labelpath) {
	std::vector<std::pair<cv::Mat, int>> dataset;
	std::ifstream datas(datapath, std::ios::binary);
	std::ifstream labels(labelpath, std::ios::binary);

	if (!datas.is_open() || !labels.is_open())
		throw std::runtime_error("binary files could not be loaded");

	int magic_number = 0; int number_of_images = 0; int r; int c;
	int n_rows = 0; int n_cols = 0; unsigned char temp = 0;

	// parse data header
	datas.read((char*)& magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	datas.read((char*)& number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	datas.read((char*)& n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	datas.read((char*)& n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);

	// parse label header - ignore
	int dummy;
	labels.read((char*)& dummy, sizeof(dummy));
	labels.read((char*)& dummy, sizeof(dummy));

	for (int i = 0; i < number_of_images; ++i) {
		cv::Mat img(n_rows, n_cols, CV_32FC1);

		for (r = 0; r < n_rows; ++r) {
			for (c = 0; c < n_cols; ++c) {
				datas.read((char*)& temp, sizeof(temp));
				img.at<float>(r, c) = 1.0 - ((float)temp) / 255.0; // inverse 0.255 values
			}
		}
		labels.read((char*)& temp, sizeof(temp));
		dataset.push_back(std::make_pair(img, (int)temp));
	}
	return dataset;
}

int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}