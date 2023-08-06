#include "plate_det.h"
#include "utils.h"

PlateDetector::PlateDetector(string model_path,Ort::Env& env) {
	this->binary_threshold = 0.3; // 概率图转二值图的阈值
	this->polygon_threshold = 0.6;  // 是否保留box的阈值
	this->unclipRatio = 1.5;
	this->maxCandidates = 1000;

// 	wstring widestr = wstring(model_path.begin(), model_path.end());
//     static Ort::Env env;
//     env = default_env();
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    ort_session = new Session(env, model_path.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++) {
		input_names.push_back(ort_session->GetInputName(i, allocator));
	}
	for (int i = 0; i < numOutputNodes; i++) {
		output_names.push_back(ort_session->GetOutputName(i, allocator));
	}
}


Mat PlateDetector::preprocess(Mat img) {
	Mat dst;
	cvtColor(img, dst, COLOR_BGR2RGB);

	int h = img.rows;
	int w = img.cols;
	float scale_h = 1;
	float scale_w = 1;
	if (h < w) {
		scale_h = (float)this->short_size / (float)h;
		float tar_w = (float)w * scale_h;
		tar_w = tar_w - (int)tar_w % 32; // 能被32整除,且大于32
		tar_w = max((float)32, tar_w);
		scale_w = tar_w / (float)w;
	}
	else {
		scale_w = (float)this->short_size / (float)w;
		float tar_h = (float)h * scale_w;
		tar_h = tar_h - (int)tar_h % 32;
		tar_h = max((float)32, tar_h);
		scale_h = tar_h / (float)h;
	}
	resize(dst, dst, Size(int(scale_w * dst.cols), int(scale_h * dst.rows)), INTER_LINEAR);
	return dst;
}

void PlateDetector::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - this->meanValues[c]) / this->normValues[c];
			}
		}
	}
}

vector<vector<Point2f>> PlateDetector::detect(Mat& img) {
	int h = img.rows;
	int w = img.cols;
	Mat dst = this->preprocess(img);
	this->normalize_(dst);
	array<int64_t, 4> input_shape_{ 1,3,dst.rows,dst.cols };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info,
		input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr },
		&input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	const float* floatArray = ort_outputs[0].GetTensorMutableData<float>();
	int outputCount = 1;
	auto dims = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape();
	for (int i = 0; i < dims.size(); i++) {
		int dim = dims.at(i);
		outputCount *= dim;
	}

	// 模型输出概率图，推理只使用概率图，经过阈值之后形成的二值图，训练的时候用阈值图进行二值化，但是推理只用概率图和固定阈值进行二值化
	Mat binary(dst.rows, dst.cols, CV_32FC1);
	memcpy(binary.data, floatArray, outputCount * sizeof(float));

	// threshold
	Mat bitmap;
	threshold(binary, bitmap, binary_threshold, 255, THRESH_BINARY);
	// scale ratio
	float scaleHeight = (float)(h) / (float)(binary.size[0]);
	float scaleWidth = (float)(w) / (float)(binary.size[1]);
	// find contours
	vector<vector<Point>> contours;
	bitmap.convertTo(bitmap, CV_8UC1);
	findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	// Candidate number limitation
	size_t numCandidate = min(contours.size(), (size_t)(maxCandidates > 0 ? maxCandidates : INT_MAX));
	vector<float> confidences;
	vector<vector<Point2f>> results;
	for (size_t i = 0; i < numCandidate; i++) {
		vector<Point>& contour = contours[i];

		// calculate text contour score
		if (contourScore(binary, contour) < polygon_threshold) {
			continue;
		}

		// rescale
		vector<Point> contourScaled;
		contourScaled.reserve(contour.size());
		for (size_t j = 0; j < contour.size(); j++) {
			contourScaled.push_back(Point(int(contour[j].x * scaleWidth),
				int(contour[j].y * scaleHeight)));
		}

		// unclip
		RotatedRect box = minAreaRect(contourScaled);

		// minArea() rect is not normalized,it may return rectangle with angle=-90 or height<width
		const float angle_threshold = 60;
		bool swap_size = false;
		if (box.size.width < box.size.height) { //水平方向的才是车牌
			swap_size = true;
		}
		else if (fabs(box.angle) >= angle_threshold) {
			swap_size = true;
		}
		if (swap_size) {
			swap(box.size.width, box.size.height);
			if (box.angle < 0)
				box.angle += 90;
			else if (box.angle > 0)
				box.angle -= 90;
		}

		Point2f vertex[4];
		box.points(vertex); // bl,tl,tr,br, returns 4 vertices of the rectangle
		vector<Point2f> approx;
		for (int j = 0; j < 4; j++) {
			approx.emplace_back(vertex[j]);
		}
		vector<Point2f> ploygon;
		unclip(approx, ploygon);
		results.push_back(ploygon);
	}
	confidences = vector<float>(contours.size(), 1.0f);
	return results;
}


float PlateDetector::contourScore(const Mat& binary, const vector<Point>& contour) {
	Rect rect = boundingRect(contour);
	int xmin = max(rect.x, 0);
	int xmax = min(rect.x + rect.width, binary.cols - 1);
	int ymin = max(rect.y, 0);
	int ymax = min(rect.y + rect.height, binary.rows - 1);

	Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
	Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
	vector<Point> roiContour;
	for (size_t i = 0; i < contour.size(); i++) {
		Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
		roiContour.push_back(pt);
	}
	vector<vector<Point>> roiContours = { roiContour };
	fillPoly(mask, roiContours, Scalar(1));
	float score = mean(binROI, mask).val[0];
	// 其实论文的motivation是这样子的，让训练的target和最终测试的target尽量一致，
	// 因为测试的时候使用的是后处理的二值mask来做评判，那么将可微分的二值化加入训练可以使得训练
	// 和测试过程更加贴近，从而获得一个好的效果
	return score;
}

void PlateDetector::unclip(const vector<Point2f>& inPoly, vector<Point2f>& outPoly) {
	float area = contourArea(inPoly);
	float length = arcLength(inPoly, true);
	float distance = area * unclipRatio / length;

	size_t numPoints = inPoly.size();
	vector<vector<Point2f>> newLines;
	for (size_t i = 0; i < numPoints; i++) {
		vector<Point2f> newLine;
		Point pt1 = inPoly[i];
		Point pt2 = inPoly[(i - 1) % numPoints];
		Point vec = pt1 - pt2;
		float unclipDis = (float)(distance / norm(vec));
		Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
		newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
		newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
		newLines.push_back(newLine);
	}

	size_t numLines = newLines.size();
	for (size_t i = 0; i < numLines; i++) {
		Point2f a = newLines[i][0];
		Point2f b = newLines[i][1];
		Point2f c = newLines[(i + 1) % numLines][0];
		Point2f d = newLines[(i + 1) % numLines][1];
		Point2f pt;
		Point2f v1 = b - a;
		Point2f v2 = d - c;
		float cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * (norm(v2)));

		if (fabs(cosAngle) > 0.7) {
			pt.x = (b.x + c.x) * 0.5;
			pt.y = (b.y + c.y) * 0.5;
		}
		else {
			float denom = a.x * (float)(d.y - c.y) + b.x * (float)(c.y - d.y) +
				d.x * (float)(b.y - a.y) + c.x * (float)(a.y - b.y);
			float num = a.x * (float)(d.y - c.y) + c.x * (float)(a.y - d.y) + d.x * (float)(c.y - a.y);
			float s = num / denom;

			pt.x = a.x + s * (b.x - a.x);
			pt.y = a.y + s * (b.y - a.y);
		}
		outPoly.push_back(pt);
	}
}


void PlateDetector::draw_pred(Mat& img, vector<vector<Point2f>>results) {
	for (int i = 0; i < results.size(); i++) {
		for (int j = 0; j < 4; j++) {
			circle(img, Point((int)results[i][j].x, (int)results[i][j].y), 2, Scalar(0, 0, 255), -1);
			if (j < 3) {
				line(img, Point((int)results[i][j].x, (int)results[i][j].y),
					Point((int)results[i][j + 1].x, (int)results[i][j + 1].y),
					Scalar(0, 255, 0));
			}
			else {
				line(img, Point((int)results[i][j].x, (int)results[i][j].y),
					Point((int)results[i][0].x, (int)results[i][0].y), Scalar(0, 255, 0));
			}
		}
	}
}





