#include "plate_rec.h"
#include "utils.h"

TextRecognizer::TextRecognizer(string model_path, string classes_file,Ort::Env& env) {
//     env = default_env();
//     static Ort::Env env;
	/*simcode["浙"] = 'ZJ-';
	simcode["粤"] = 'GD-';
	simcode["京"] = 'BJ-';
	simcode["津"] = 'TJ-';
	simcode["冀"] = 'HE-';
	simcode["晋"] = 'SX-';
	simcode["蒙"] = 'NM-';
	simcode["辽"] = 'LN-';
	simcode["黑"] = 'HLJ-';
	simcode["沪"] = 'SH-';
	simcode["吉"] = 'JL-';
	simcode["苏"] = 'JS-';
	simcode["皖"] = 'AH-';
	simcode["赣"] = 'JX-';
	simcode["鲁"] = 'SD-';
	simcode["豫"] = 'HA-';
	simcode["鄂"] = 'HB-';
	simcode["湘"] = 'HN-';
	simcode["桂"] = 'GX-';
	simcode["琼"] = 'HI-';
	simcode["渝"] = 'CQ-';
	simcode["川"] = 'SC-';
	simcode["贵"] = 'GZ-';
	simcode["云"] = 'YN-';
	simcode["藏"] = 'XZ-';
	simcode["陕"] = 'SN-';
	simcode["甘"] = 'GS-';
	simcode["青"] = 'QH-';
	simcode["宁"] = 'NX-';
	simcode["闽"] = 'FJ-';
	simcode["·"] = ' ';*/
	//simcode.insert(pair<string, string>('浙', 'ZJ-'));

	/*simcode = {
		{'浙', 'ZJ-'},
		{ '粤' , 'GD-' },
		{ '京' , 'BJ-' },
		{ '津' , 'TJ-' },
		{ '冀' , 'HE-' },
		{ '晋' , 'SX-' },
		{ '蒙' , 'NM-' },
		{ '辽' , 'LN-' },
		{ '黑' , 'HLJ-' },
		{ '沪' , 'SH-' },
		{ '吉' , 'JL-' },
		{ '苏' , 'JS-' },
		{ '皖' , 'AH-' },
		{ '赣' , 'JX-' },
		{ '鲁' , 'SD-' },
		{ '豫' , 'HA-' },
		{ '鄂', 'HB-' },
		{ '湘' , 'HN-' },
		{ '桂' , 'GX-' },
		{ '琼' , 'HI-' },
		{ '渝', 'CQ-' },
		{ '川' , 'SC-' },
		{ '贵' , 'GZ-' },
		{ '云' , 'YN-' },
		{ '藏' , 'XZ-' },
		{ '陕' , 'SN-' },
		{ '甘' , 'GS-' },
		{ '青' , 'QH-' },
		{ '宁' , 'NX-' },
		{ '闽' , 'FJ-' },
		{ '·' , ' ' },
	};*/

// 	wstring widestr = wstring(model_path.begin(), model_path.end());
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 1);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++) {
		input_names.push_back(ort_session->GetInputName(i, allocator));
		auto input_dims = ort_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++) {
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		auto output_dims = ort_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		output_node_dims.push_back(output_dims);
	}

	max_wh_ratio = (float)this->inp_width / (float)this->inp_height;
	imgW = int(this->inp_height * max_wh_ratio); //保证宽高比，高是48,

	ifstream ifs(classes_file.c_str());
	string line;
	while (getline(ifs, line)) {
		this->alphabet.push_back(line);
	}
	names_len = this->alphabet.size();
}


Mat TextRecognizer::get_rotate_crop_image_v1(const Mat& frame, Point2f vertices[4]) {
	int left = 10000;
	int right = 0;
	int top = 10000;
	int bottom = 0;
	for (int i = 0; i < 4; i++) {
		if (vertices[i].x < left) {
			left = int(vertices[i].x);
		}
		if (vertices[i].y < top) {
			top = int(vertices[i].y);
		}
		if (vertices[i].x > right) {
			right = int(vertices[i].x);
		}
		if (vertices[i].y > bottom) {
			bottom = int(vertices[i].y);
		}
//         cout<<"left:"<<left<<endl;
//         cout<<"right:"<<right<<endl;
//         cout<<"top:"<<top<<endl;
//         cout<<"bottom:"<<bottom<<endl;
	}

	Rect rect;
	rect.x = left;
	rect.y = top;
//     cout <<"######1"<<endl;
	rect.width = right - left;
	rect.height = bottom - top;
	if (rect.width == 0)
		rect.width = right - left + 1;
	if (rect.height == 0)
		rect.height = bottom - top + 1;
//     if (rect.width < 0)
//         rect.width = 1;
//     if (rect.height < 0)
//         rect.height = 1;
//     if (rect.x<0)
//         rect.x = 0;
//     if (rect.y<0)
//         rect.y = 0;
//     if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= frame.cols &&
//               0 <= rect.y && 0 <= rect.height && rect.y + rect.height <=frame.rows){
//         rect.width = 1;
//         rect.height = 1;
//     }    
//      cout<<"x:"<<rect.x<<"y:"<<rect.y<<"width:"<<rect.width<<"height:"<<rect.height<<endl;
//      cout<<"frame.width:"<<frame.cols<<"frame.height"<<frame.rows;
//     cout <<"######2"<<endl;
	Mat crop_plate = frame(rect);
//     cout <<"######3"<<endl;
	const Size outputSize = Size(rect.width, rect.height);
	Point2f targetVertices[4] = {
		Point(0,outputSize.height - 1),
		Point(0,0),Point(outputSize.width - 1,0),
		Point(outputSize.width - 1,outputSize.height - 1),
	};
	for (int i = 0; i < 4; i++) {
		vertices[i].x -= left;
		vertices[i].y -= top;
	}
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	Mat result;
	warpPerspective(crop_plate, result, rotationMatrix, outputSize);
	return result;
}


Mat TextRecognizer::preprocess(Mat img) {
	Mat dst;
	cvtColor(img, dst, COLOR_BGR2RGB);
	int h = img.rows;
	int w = img.cols;
	const float ratio = w / float(h);
	int resized_w = int(ceil((float)this->inp_height * ratio));
	if (ceil(this->inp_height * ratio) > imgW) {
		resized_w = imgW;
	}
	resize(dst, dst, Size(resized_w, this->inp_height), INTER_LINEAR);
	return dst;
}


void TextRecognizer::normalize_(Mat img) {
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(this->inp_height * imgW * img.channels());

	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < imgW; j++) {
				if (j < col) {
					float pix = img.ptr<uchar>(i)[j * 3 + c];
					this->input_image_[c * row * imgW + i * imgW + j] = (pix / 255.0 - 0.5) / 0.5;
				}
				else {
					this->input_image_[c * row * imgW + i * imgW + j] = 0;
				}
			}
		}
	}
}

string TextRecognizer::detect(Mat img) {
	Mat dst = this->preprocess(img);
	this->normalize_(dst);
	array<int64_t, 4> input_shape_{ 1,3,this->inp_height,this->imgW };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info,
		input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr },
		&input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	//int i = 0, j = 0;
	int h = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
	int w = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);
	int* preb_label = new int[w];//开辟内存中的w是变量
	for (int i = 0; i < w; i++) {
		int one_label_idx = 0;
		float max_data = -10000;
		for (int j = 0; j < h; j++) {
			float data_ = pdata[i * h + j];
			if (data_ > max_data) {
				max_data = data_;
				one_label_idx = j;
			}
		}
		preb_label[i] = one_label_idx;
	}

	vector<int> no_repeat_blank_label;
	for (size_t index = 1; index < w; ++index) {
		if (preb_label[index] != 0 && preb_label[index - 1] != preb_label[index]) {
			no_repeat_blank_label.push_back(preb_label[index] - 1);
		}
	}

	delete[] preb_label;
	int len_s = no_repeat_blank_label.size();
	string plate_text;
	for (int i = 0; i < len_s; i++) {
		plate_text += alphabet[no_repeat_blank_label[i]];
	}

	return plate_text;
}

