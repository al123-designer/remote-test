#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

#define PI 3.1415926  
string title = "Not Found";

// rbg��ֵתHSV��ֵ
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


// ����㷨(��ˮ�����)
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

//�ж�rect1��rect2�Ƿ��н���  
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
	cout << "����ͼ��ľ��Ե�ַ��";
	cin >> originPath;
	Mat srcImg = imread(originPath, IMREAD_COLOR);// ������ͼƬ
	if (srcImg.empty())
	{
		cout << "�Ҳ������ͼ��,���·��" << endl;
		return 0;
	}
	if (srcImg.empty())
	{
		cout << "�Ҳ������ͼ��,���·��" << endl;
		return 0;
	}

	// �޶�ͼ�񳤿�
	int width = srcImg.cols;//ͼ����  
	int height = srcImg.rows;//ͼ��߶�
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

	// ��һ��:�ָ��ɫ��ɫɫ��
	Mat matRgb = Mat::zeros(srcImg.size(), CV_8UC1);
	int x, y; //ѭ��  
	for (y = 0; y < height; y++)
		for (x = 0; x < width; x++)
		{
			// ��ȡBGRֵ  
			float B = srcImg.at<Vec3b>(y, x)[0];
			float G = srcImg.at<Vec3b>(y, x)[1];
			float R = srcImg.at<Vec3b>(y, x)[2];
			float H, S, V;

			Rgb2Hsv(R, G, B, H, S, V);
			//cout << "H=" << H << ",S=" << S<< ",V=" << V << endl;
			//��ɫ��Χ  
			if ((H >= 135 * 2 && H <= 180 * 2 || H >= 0 && H <= 10 * 2) && S * 255 >= 16
				&& S * 255 <= 255 && V >= 46 && V <= 255)
			{
				matRgb.at<uchar>(y, x) = 255;
			}// if
		}// for
	//imshow("hsv", matRgb);
	waitKey(0);



	// �ڶ���:ȥ����ش���
	medianBlur(matRgb, matRgb, 3);// ��ֵ�˲�
	medianBlur(matRgb, matRgb, 5);// ��ֵ�˲�
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
	erode(matRgb, matRgb, element);//��ʴ    
	dilate(matRgb, matRgb, element1);//����   
	//imshow("dilate", matRgb);
	waitKey(0);

	// ������:���
	fillHole(matRgb, matRgb);//���   
	//imshow("fillHole", matRgb);
	waitKey(0);

	// ���Ĳ�:������
	vector<vector<Point>>contours; //����    
	vector<Vec4i> hierarchy;//�ֲ�    
	findContours(matRgb, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE, Point(0, 0));//Ѱ������    
	vector<vector<Point>> contours_poly(contours.size());  //���ƺ�������㼯    
	vector<Rect> boundRect(contours.size());  //��Χ�㼯����С����vector

	// ���岽:����������С��Ӿ���
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true); //�Զ�����������ʵ����ƣ�contours_poly[i]������Ľ��Ƶ㼯    
		boundRect[i] = boundingRect(Mat(contours_poly[i])); //���㲢���ذ�Χ�����㼯����С����       
	}

	// ������:����ȡ������������ȥ�룬ɸѡ����ͨ��־
	Mat drawing = Mat::zeros(matRgb.size(), CV_8UC3);
	Mat imageContours1 = Mat::zeros(matRgb.size(), CV_8UC1); //��С���Բ����
	vector<Mat> vec_roi;	// �洢ɸѡ���Ľ�ͨ��־��ͼ����Ϣ
	vector<Rect> vec_rect;	// �洢��ͨ��־�����ԭͼ��roi����
	for (int i = 0; i < contours.size(); i++)
	{
		Rect rect = boundRect[i];

		//1. �����������ڲ��������ž��Σ��򽫱�������С����ȡ��
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

		//2.�������ɸѡ      
		float Area = (float)rect.width * (float)rect.height;
		float dConArea = (float)contourArea(contours[i]);
		float dConLen = (float)arcLength(contours[i], 1);
		if (dConArea < 500)
			continue;

		//3.�߿��ɸѡ
		float ratio = (float)rect.width / (float)rect.height;
		if (ratio > 1.3 || ratio < 0.4)
			continue;

		// ɸѡ���,���д洢
		Mat roi = srcImg(Rect(boundRect[i].tl(), boundRect[i].br()));
		vec_roi.push_back(roi);
		vec_rect.push_back(Rect(boundRect[i].tl(), boundRect[i].br()));

	}

	// ���߲�:����ģ��Ľ�ͨ��־
	Mat template_srcimg = imread("C:\\Users\\suixi\\Desktop\\traffic\\real.jpg");
	cvtColor(template_srcimg, template_srcimg, COLOR_BGR2GRAY); //ͼ��ҶȻ�

	//�ڰ˲�:�������н�ͨ��־,�������ƶ�ƥ��
	Mat gray_template, gray_roi;
	for (int i = 0; i < vec_roi.size(); i++)
	{
		//����һ��ģ�帱��
			template_srcimg.copyTo(gray_template);
		Mat tmp_roi = vec_roi[i].clone();
		//1. tmp_roiͼ�� resizeΪ����
		tmp_roi.resize(min(tmp_roi.rows, tmp_roi.cols), min(tmp_roi.rows, tmp_roi.cols));
		//2. tmp_roiͼ��ҶȻ�
		cvtColor(tmp_roi, gray_roi, COLOR_BGR2GRAY);
		//3. ��ģ��ͼ��ͳһ�ߴ�
		int w = gray_template.cols, h = gray_template.rows;
		resize(gray_roi, gray_roi, cv::Size(w, h));
		//4. �������ڽ�Բ
		vector<vector<bool>> enclosingcircle_flag;
		Point center(0.5 * w, 0.5 * h);
		for (int col = 0; col < w; col++)
		{
			vector<bool> col_flag;
			for (int row = 0; row < h; row++)
			{
				bool flag;
				if (((col - center.x) * (col - center.x) + (row - center.y) * (row - center.y)) < center.x * center.x) // �ڽ�Բ��
					flag = true;
				else
					flag = false;
				col_flag.push_back(flag);
			}
			enclosingcircle_flag.push_back(col_flag);
		}

		//5.��˹�˲�
		cv::GaussianBlur(gray_roi, gray_roi, cv::Size(7, 7), 3, 3);
		cv::GaussianBlur(gray_roi, gray_roi, cv::Size(5, 5), 3, 3);
		cv::GaussianBlur(gray_template, gray_template, cv::Size(7, 7), 3, 3);
		cv::GaussianBlur(gray_template, gray_template, cv::Size(5, 5), 3, 3);

		//6.��ֵ��
		// ��ͼ��ĻҶ�ֵ��ֵ��Ϊ��ֵ������ֵ
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
		imshow("Ŀ��ͼ��Ҷ�ͼ", gray_template);

		//7. ȥ��Բ������
/*
		����˼�룺Ѱ�Һ�ɫԲ���ڵ����Բ��ͨ���۲��ֵ��·����Կ�����Բ����Ϊ��ɫ����ɫԲ���ڱ�Ե��Ϊ��ɫ��
������Բ����һ���ǰ�ɫ������ͨ�����ϵ�����������Բ��Ȼ���ж���һȦ��Բ�������ص����ֵ��ֵ����������ں�ɫԲ���У���ô���Ǽ����Բ���ĻҶ�ֵ��ֵһ���Ƿǳ��ߵģ�����Ҫ��������Բ����������������ĳһ�ε�������һȦ��Բ�������ص���ձ齵�ͷǳ�������˵������������Բ�������˺�ɫԲ���ڱ�Ե��

ȥ��Բ���Ĺ������ģ��ͼ��Ŀ��ͼͬʱ���С�

w_r:��ʾ����������Բ����Ȧ�뾶����ʼֵ����ͼ���������Բ�뾶
n_r :��ʾ����������Բ����Ȧ�뾶
w_r - n_r:��ʾ����������Բ���Ŀ�ȡ���������ʼ������Ϊ2.���������ԽС�����ǵĵ��������ͻ�Խ�࣬����Խ���������Ķ�λԽ��ȷ��������ù���ÿ�������̶Ⱦͻ�Ӵ󣬼��ٵ�������������Բ���ķ�Χ��������ɫԲ���ڱ�Ե�Ķ�λ�ͻ�Խģ����
*/

// w_r ��������Բ������뾶����ʼֵΪ������Բ�뾶
// n_r ��������Բ�����ڰ뾶
		int w_r = 0.5 * w, n_r;
		if (w_r > 2)
			n_r = w_r - 2;// ����Բ���Ŀ������Ϊ2	
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
						continue; // �������ڽ�Բ,����
					// �жϸ�Բ�ڵ����ص����Բ�ĵľ����Ƿ�������Բ������뾶���ڰ뾶֮�䣬
					// ������������Բ������뾶���ڰ뾶֮�䣬���ʾ�����ص�������Բ����
					if ((((col - center.x) * (col - center.x) + (row - center.y) * (row - center.y)) < w_r * w_r)
						&& (((col - center.x) * (col - center.x) + (row - center.y) * (row - center.y)) > n_r * n_r))
					{
						// ��ʾ�����ص�������Բ���ڣ�ͳ�ƻҶ�ֵ
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
			// �ж�����Բ���Ƿ񵽴��˺�ɫԲ���ڱ�Ե
			if (avg_roi > 0.8 * 255 || avg_template > 0.8 * 255)
			{
				// ͳ�Ƶ�����ֵ����˵��Ŀǰ������Բ�����ں�ɫԲ���ڲ�
				for (int i = 0; i < vec_p.size(); i++)
				{
					// �����ص������Բ�����б�ǣ��������ٳ�Ϊ��ɫԲ���ڱ�Ե�����أ��ں���ĵ�����ֱ�������������ظ�����
					enclosingcircle_flag[vec_p[i].y][vec_p[i].x] = false;
				}
				// ����Բ�����н�һ��������ԭ��������Բ���ڰ뾶��Ϊ��뾶������Բ�����ڰ뾶��������2���ء�
				w_r = n_r;
				n_r -= 2;
			}
			else // ����Բ�����ػҶ�ֵ�����õ���ֵ�ڣ�˵��������Բ�Ѿ��ﵽ�˺�ɫԲ���ڱ�Ե��������Բ���İ뾶����Ϊ��ɫԲ���ڵİ뾶����������
				break;
		}

#if 0
		// �Ƚ�����ͼ255���ص�Ľ����벢���ı�ֵ
		float jiaoji = 0, bingji = 0;
		for (int x = 0; x < w; x++)
			for (int y = 0; y < h; y++)
			{
				if (enclosingcircle_flag[x][y] == false)
					continue; // �������ڽ�Բ,����
				if (gray_roi.at<uchar>(y, x) == 255 && gray_template.at<uchar>(y, x) == 255)	//����
					jiaoji++;
				if (gray_roi.at<uchar>(y, x) == 255 || gray_template.at<uchar>(y, x) == 255)	//����
					bingji++;
			}

		float score = jiaoji / bingji;
		float score_max = score;
# else
		//8. ��ȡ�������ڵ����ƶ�ƥ���㷨,����һ����Χ�ڵ�ƫ�� ,����ȡ���н���е�����ֵ��Ϊƥ���㷨
		int offset_max = 10;
		float score_max = 0, jiaoji = 0, bingji = 0, score;
		for (int offset = 0; offset < offset_max; offset++)
		{
			for (int x = 0; x < w - offset; x++)
				for (int y = 0; y < h; y++)
				{
					if (enclosingcircle_flag[x][y] == false)
						continue; // �������ڽ�Բ,����
					if (gray_roi.at<uchar>(y, x + offset) == 255 && gray_template.at<uchar>(y, x) == 255)	//����
						jiaoji++;
					if (gray_roi.at<uchar>(y, x + offset) == 255 || gray_template.at<uchar>(y, x) == 255)	//����
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
						continue; // �������ڽ�Բ,����
					if (gray_roi.at<uchar>(y + offset, x) == 255 && gray_template.at<uchar>(y, x) == 255)	//����
						jiaoji++;
					if (gray_roi.at<uchar>(y + offset, x) == 255 || gray_template.at<uchar>(y, x) == 255)	//����
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
		buf.precision(3);//����Ĭ�Ͼ���
		buf.setf(std::ios::fixed);//����С��λ
		buf << score_max;
		std::string str;
		str = buf.str();
		//putText(srcImg, str, Point(vec_rect[i].x, vec_rect[i].y), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 0), 2);
		//8. ���ƶ��ж�
		if (score_max > 0.45) // �ж�ͨ��
		{
			Scalar color = Scalar(128, 0, 128);  // ��ɫ
			rectangle(srcImg, vec_rect[i], color, 10, 8, 0); //���ƶ�ͨ��,����
			title = "Annotated Image";
		}
		//else
		//{
		//	rectangle(srcImg, vec_rect[i], Scalar(0, 0, 255), 4, 8, 0); //���ƶȲ�ͨ��,�����
		//}
	}

    imshow(title, srcImg);//��ʾ����Ч��ͼ
	waitKey(0);
	waitKey(0);

	system("pause");



	return 0;
}
