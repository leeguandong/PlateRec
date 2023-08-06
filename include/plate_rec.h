#pragma once
#ifndef PLATE_REC
#define PLATE_REC
#include <map>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>


using namespace cv;
using namespace std;
using namespace Ort;

class TextRecognizer {
public:
	TextRecognizer(string model_path, string classes_file,Ort::Env& env);
	string detect(Mat img);
	//Mat fourPointsTransform(const Mat& frame, Point2f vertices[4]);
	Mat get_rotate_crop_image_v1(const Mat& frame, Point2f vertices[4]);
	//Mat get_rotate_crop_image_v2(const Mat& frame, vector<vector<Point2f>> box);
	map<string, string> simcode; // 中文占两个字符 map<string,string> simcode

private:
	Mat preprocess(Mat img);
	void normalize_(Mat img);
	const int inp_width = 320;
	const int inp_height = 48;
	float max_wh_ratio;
	int imgW;
	vector<float> input_image_;
	vector<string> alphabet;
	int names_len;

//  	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "CRNN");
//     Ort::Env env;
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims;
	vector<vector<int64_t>> output_node_dims;	
};



#endif // !PLATE_REC
