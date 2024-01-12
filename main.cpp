#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

#define PI 3.1415926  
string title = "Not Found";

// rbg数值转HSV数值
void Rgb2Hsv(float R, float G, float B, float& H, float& S, float& V)
{
	// r,g,b values are from 0 to 1
	// h = [0,360], s = [0,1], v = [0,255]
	// if s == 0, then h = -1 (undefined)
	float min, max, delta, tmp;
	tmp = R > G ? G : R;
	min = tmp > B ? B : tmp;
	tmp = R > G ? R : G;
	max = tmp > B ? tmp : B;
	V = max; // v
	delta = max - min;
	if (max != 0)
		S = delta / max; // s
	else
	{
		// r = g = b = 0 // s = 0, v is undefined
		S = 0;
		H = 0;
		return;
	}
	if (delta == 0) {
		H = 0;
		return;
	}
	else if (R == max) {
		if (G >= B)
			H = (G - B) / delta; // between yellow & magenta
		else
			H = (G - B) / delta + 6.0;
	}
	else if (G == max)
		H = 2.0 + (B - R) / delta; // between cyan & yellow
	else if (B == max)
		H = 4.0 + (R - G) / delta; // between magenta & cyan
	H *= 60.0; // degrees
}


// 填充算法(漫水天填充)
void fillHole(const Mat srcBw, Mat& dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

	cv::floodFill(Temp, Point(0, 0), Scalar(255));

	Mat cutImg;
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

	dstBw = srcBw | (~cutImg);

}

//判断rect1与rect2是否有交集  
bool isInside(Rect rect1, Rect rect2)
{
	Rect t = rect1 & rect2;
	if (rect1.area() > rect2.area())
	{
		return false;
	}
	else
	{
		if (t.area() != 0)
			return true;
	}
}



int main()
{
	string originPath;
	cout << "输入图像的绝对地址：";
	cin >> originPath;
	Mat srcImg = imread(originPath, IMREAD_COLOR);// 载入检测图片
	if (srcImg.empty())
	{
		cout << "找不到相关图像,检查路径" << endl;
		return 0;
	}
	if (srcImg.empty())
	{
		cout << "找不到相关图像,检查路径" << endl;
		return 0;
	}

	// 限定图像长宽
	int width = srcImg.cols;//图像宽度  
	int height = srcImg.rows;//图像高度
	if (width > 1280 || height > 720)
	{
		float factor = min((float)1280 / width, (float)720 / height);
		cout << factor << endl;
		resize(srcImg, srcImg, Size(factor * width, factor * height));
		width *= factor;
		height *= factor;

	}
	//cout << "width=" << width << ",height=" << height << endl;
	//imshow("srcImg", srcImg);
	//waitKey(0);

	// 第一步:分割红色颜色色块
	Mat matRgb = Mat::zeros(srcImg.size(), CV_8UC1);
	int x, y; //循环  
	for (y = 0; y < height; y++)
		for (x = 0; x < width; x++)
		{
			// 获取BGR值  
			float B = srcImg.at<Vec3b>(y, x)[0];
			float G = srcImg.at<Vec3b>(y, x)[1];
			float R = srcImg.at<Vec3b>(y, x)[2];
			float H, S, V;

			Rgb2Hsv(R, G, B, H, S, V);
			//cout << "H=" << H << ",S=" << S<< ",V=" << V << endl;
			//红色范围  
			if ((H >= 135 * 2 && H <= 180 * 2 || H >= 0 && H <= 10 * 2) && S * 255 >= 16
				&& S * 255 <= 255 && V >= 46 && V <= 255)
			{
				matRgb.at<uchar>(y, x) = 255;
			}// if
		}// for
	//imshow("hsv", matRgb);
	waitKey(0);



	// 第二步:去噪相关处理
	medianBlur(matRgb, matRgb, 3);// 中值滤波
	medianBlur(matRgb, matRgb, 5);// 中值滤波
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
	erode(matRgb, matRgb, element);//腐蚀    
	dilate(matRgb, matRgb, element1);//膨胀   
	//imshow("dilate", matRgb);
	waitKey(0);

	// 第三步:填充
	fillHole(matRgb, matRgb);//填充   
	//imshow("fillHole", matRgb);
	waitKey(0);

	// 第四步:找轮廓
	vector<vector<Point>>contours; //轮廓    
	vector<Vec4i> hierarchy;//分层    
	findContours(matRgb, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE, Point(0, 0));//寻找轮廓    
	vector<vector<Point>> contours_poly(contours.size());  //近似后的轮廓点集    
	vector<Rect> boundRect(contours.size());  //包围点集的最小矩形vector

	// 第五步:找轮廓的最小外接矩形
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true); //对多边形曲线做适当近似，contours_poly[i]是输出的近似点集    
		boundRect[i] = boundingRect(Mat(contours_poly[i])); //计算并返回包围轮廓点集的最小矩形       
	}

	// 第六步:对提取出的轮廓进行去噪，筛选出交通标志
	Mat drawing = Mat::zeros(matRgb.size(), CV_8UC3);
	Mat imageContours1 = Mat::zeros(matRgb.size(), CV_8UC1); //最小外结圆画布
	vector<Mat> vec_roi;	// 存储筛选出的交通标志的图像信息
	vector<Rect> vec_rect;	// 存储交通标志相对于原图的roi区域
	for (int i = 0; i < contours.size(); i++)
	{
		Rect rect = boundRect[i];

		//1. 若轮廓矩形内部还包含着矩形，则将被包含的小矩形取消
		bool inside = false;
		for (int j = 0; j < contours.size(); j++)
		{
			Rect t = boundRect[j];
			if (rect == t)
				continue;
			else if (isInside(rect, t))
			{
				inside = true;
				break;
			}
		}// for
		if (inside)
			continue;

		//2.轮廓面积筛选      
		float Area = (float)rect.width * (float)rect.height;
		float dConArea = (float)contourArea(contours[i]);
		float dConLen = (float)arcLength(contours[i], 1);
		if (dConArea < 500)
			continue;

		//3.高宽比筛选
		float ratio = (float)rect.width / (float)rect.height;
		if (ratio > 1.3 || ratio < 0.4)
			continue;

		// 筛选完成,进行存储
		Mat roi = srcImg(Rect(boundRect[i].tl(), boundRect[i].br()));
		vec_roi.push_back(roi);
		vec_rect.push_back(Rect(boundRect[i].tl(), boundRect[i].br()));

	}

	// 第七步:载入模板的交通标志
	Mat template_srcimg = imread("C:\\Users\\suixi\\Desktop\\traffic\\real.jpg");
	cvtColor(template_srcimg, template_srcimg, COLOR_BGR2GRAY); //图像灰度化

	//第八步:遍历所有交通标志,进行相似度匹配
	Mat gray_template, gray_roi;
	for (int i = 0; i < vec_roi.size(); i++)
	{
		//创建一个模板副本
			template_srcimg.copyTo(gray_template);
		Mat tmp_roi = vec_roi[i].clone();
		//1. tmp_roi图像 resize为方形
		tmp_roi.resize(min(tmp_roi.rows, tmp_roi.cols), min(tmp_roi.rows, tmp_roi.cols));
		//2. tmp_roi图像灰度化
		cvtColor(tmp_roi, gray_roi, COLOR_BGR2GRAY);
		//3. 与模板图像统一尺寸
		int w = gray_template.cols, h = gray_template.rows;
		resize(gray_roi, gray_roi, cv::Size(w, h));
		//4. 标记最大内接圆
		vector<vector<bool>> enclosingcircle_flag;
		Point center(0.5 * w, 0.5 * h);
		for (int col = 0; col < w; col++)
		{
			vector<bool> col_flag;
			for (int row = 0; row < h; row++)
			{
				bool flag;
				if (((col - center.x) * (col - center.x) + (row - center.y) * (row - center.y)) < center.x * center.x) // 内接圆内
					flag = true;
				else
					flag = false;
				col_flag.push_back(flag);
			}
			enclosingcircle_flag.push_back(col_flag);
		}

		//5.高斯滤波
		cv::GaussianBlur(gray_roi, gray_roi, cv::Size(7, 7), 3, 3);
		cv::GaussianBlur(gray_roi, gray_roi, cv::Size(5, 5), 3, 3);
		cv::GaussianBlur(gray_template, gray_template, cv::Size(7, 7), 3, 3);
		cv::GaussianBlur(gray_template, gray_template, cv::Size(5, 5), 3, 3);

		//6.二值化
		// 与图像的灰度值均值作为二值化的阈值
		int gray_mean1 = 0, gray_mean2 = 0;
		for (int x = 0; x < w; x++)
			for (int y = 0; y < h; y++) {
				gray_mean1 += gray_roi.at<uchar>(y, x);
				gray_mean2 += gray_template.at<uchar>(y, x);
			}
		gray_mean1 /= (w * h);
		gray_mean2 /= (w * h);
		threshold(gray_roi, gray_roi, gray_mean1, 255, cv::THRESH_BINARY_INV);
		threshold(gray_template, gray_template, gray_mean2, 255, cv::THRESH_BINARY_INV);
		//imshow("gray_roi.jpg", gray_roi);
		imshow("目标图像灰度图", gray_template);

		//7. 去除圆环部分
/*
		核心思想：寻找红色圆环内的最大圆。通过观察二值化路标可以看到，圆环内为白色，红色圆环内边缘多为黑色。
最大外接圆像素一定是白色，我们通过不断地收缩最大外接圆，然后判断这一圈的圆环内像素点的阈值均值，如果依旧在红色圆环中，那么我们计算的圆环的灰度值均值一定是非常高的，还需要继续网往圆心内慢慢收缩。当某一次迭代后，这一圈的圆环内像素点的普遍降低非常厉害，说明我们收缩的圆环到达了红色圆环内边缘。

去除圆环的工作针对模板图和目标图同时进行。

w_r:表示我们收缩的圆环外圈半径，初始值就是图表的最大外接圆半径
n_r :表示我们收缩的圆环内圈半径
w_r - n_r:表示我们收缩的圆环的宽度。本代码中始终设置为2.若宽度设置越小，我们的迭代次数就会越多，计算越慢，但最后的定位越精确。宽度设置过大，每次缩进程度就会加大，减少迭代次数，但是圆环的范围扩大，最后红色圆环内边缘的定位就会越模糊。
*/

// w_r 代表收缩圆环的外半径，初始值为最大外接圆半径
// n_r 代表收缩圆环的内半径
		int w_r = 0.5 * w, n_r;
		if (w_r > 2)
			n_r = w_r - 2;// 收缩圆环的宽度设置为2	
		vector<Point> vec_p;
		while (n_r > 0)
		{
			vec_p.clear();
			int sum_roi = 0, sum_template = 0;
			for (int col = 0; col < w; col++)
			{
				for (int row = 0; row < h; row++)
				{
					if (enclosingcircle_flag[row][col] == false)
						continue; // 不处于内接圆,跳过
					// 判断该圆内的像素点距离圆心的距离是否在收缩圆环的外半径和内半径之间，
					// 若距离在收缩圆环的外半径和内半径之间，则表示该像素点在收缩圆环内
					if ((((col - center.x) * (col - center.x) + (row - center.y) * (row - center.y)) < w_r * w_r)
						&& (((col - center.x) * (col - center.x) + (row - center.y) * (row - center.y)) > n_r * n_r))
					{
						// 表示该像素点在收缩圆环内，统计灰度值
						vec_p.push_back(Point(col, row));
						sum_roi += gray_roi.at<uchar>(row, col);
						sum_template += gray_template.at<uchar>(row, col);
					}
				}
			}
			if (vec_p.size() == 0)
				break;

			int avg_roi = sum_roi / vec_p.size();
			int avg_template = sum_template / vec_p.size();
			// 判断收缩圆环是否到达了红色圆环内边缘
			if (avg_roi > 0.8 * 255 || avg_template > 0.8 * 255)
			{
				// 统计的像素值过大，说明目前的收缩圆环还在红色圆环内部
				for (int i = 0; i < vec_p.size(); i++)
				{
					// 该像素点的收缩圆环进行标记，不可能再成为红色圆环内边缘的像素，在后面的迭代中直接跳过，不必重复计算
					enclosingcircle_flag[vec_p[i].y][vec_p[i].x] = false;
				}
				// 收缩圆环进行进一步收缩，原来的收缩圆环内半径变为外半径，收缩圆环的内半径往内收缩2像素。
				w_r = n_r;
				n_r -= 2;
			}
			else // 收缩圆环像素灰度值在设置的阈值内，说明该收缩圆已经达到了红色圆环内边缘处，收缩圆环的半径可视为红色圆环内的半径，结束收缩
				break;
		}

#if 0
		// 比较两个图255像素点的交集与并集的比值
		float jiaoji = 0, bingji = 0;
		for (int x = 0; x < w; x++)
			for (int y = 0; y < h; y++)
			{
				if (enclosingcircle_flag[x][y] == false)
					continue; // 不处于内接圆,跳过
				if (gray_roi.at<uchar>(y, x) == 255 && gray_template.at<uchar>(y, x) == 255)	//交集
					jiaoji++;
				if (gray_roi.at<uchar>(y, x) == 255 || gray_template.at<uchar>(y, x) == 255)	//并集
					bingji++;
			}

		float score = jiaoji / bingji;
		float score_max = score;
# else
		//8. 采取滑动窗口的相似度匹配算法,产生一定范围内的偏移 ,最终取所有结果中的最优值作为匹配算法
		int offset_max = 10;
		float score_max = 0, jiaoji = 0, bingji = 0, score;
		for (int offset = 0; offset < offset_max; offset++)
		{
			for (int x = 0; x < w - offset; x++)
				for (int y = 0; y < h; y++)
				{
					if (enclosingcircle_flag[x][y] == false)
						continue; // 不处于内接圆,跳过
					if (gray_roi.at<uchar>(y, x + offset) == 255 && gray_template.at<uchar>(y, x) == 255)	//交集
						jiaoji++;
					if (gray_roi.at<uchar>(y, x + offset) == 255 || gray_template.at<uchar>(y, x) == 255)	//并集
						bingji++;
				}
			score = jiaoji / bingji;
			if (score > score_max)
				score_max = score;
			jiaoji = 0;
			bingji = 0;

			for (int x = 0; x < w; x++)
				for (int y = 0; y < h - offset; y++)
				{
					if (enclosingcircle_flag[x][y] == false)
						continue; // 不处于内接圆,跳过
					if (gray_roi.at<uchar>(y + offset, x) == 255 && gray_template.at<uchar>(y, x) == 255)	//交集
						jiaoji++;
					if (gray_roi.at<uchar>(y + offset, x) == 255 || gray_template.at<uchar>(y, x) == 255)	//并集
						bingji++;
				}
			score = jiaoji / bingji;
			if (score > score_max)
				score_max = score;
			jiaoji = 0;
			bingji = 0;
		}

#endif
		std::stringstream buf;
		buf.precision(3);//覆盖默认精度
		buf.setf(std::ios::fixed);//保留小数位
		buf << score_max;
		std::string str;
		str = buf.str();
		//putText(srcImg, str, Point(vec_rect[i].x, vec_rect[i].y), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 0), 2);
		//8. 相似度判断
		if (score_max > 0.45) // 判定通过
		{
			Scalar color = Scalar(128, 0, 128);  // 紫色
			rectangle(srcImg, vec_rect[i], color, 10, 8, 0); //相似度通过,画框
			title = "Annotated Image";
		}
		//else
		//{
		//	rectangle(srcImg, vec_rect[i], Scalar(0, 0, 255), 4, 8, 0); //相似度不通过,画红框
		//}
	}

    imshow(title, srcImg);//显示最终效果图
	waitKey(0);
	waitKey(0);

	system("pause");



	return 0;
}
