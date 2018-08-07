#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  
using namespace cv;

typedef struct EncodingResult {									//data struct 
	int geo; int avg; int error; double alpha; int x, y;		
}EncodingResult;

int** IntAlloc2(int width, int height) {						//alloc image
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int width, int height) {				//alloc free
	for (int i = 0; i<height; i++)
		free(image[i]);
	free(image);
}

int** ReadImage(const char* name, int* width, int* height) {	//image read
	Mat img = imread(name, IMREAD_GRAYSCALE);

	int** image = (int**)IntAlloc2(img.cols, img.rows);
	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);
	return(image);
}

void WriteImage(const char* name, int** image, int width, int height) {		//image write
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}

void ImageShow(const char* winname, int** image, int width, int height) {		//console image print
	Mat img(height, width, CV_8UC1);


	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}


void Contraction(int** image, int** image_out, int width, int height) {// 1/2 �̹��� ���
	for (int y = 0; y < height - 1; y += 2)
		for (int x = 0; x < width - 1; x += 2)
			image_out[y / 2][x / 2] = (image[y][x] + image[y][x + 1] + image[y + 1][x] + image[y + 1][x + 1]) / 4;
}

void IsoM_0(int** img_in, int width, int height, int** img_out) {// �̹��� ����(a->b copy)

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			img_out[y][x] = img_in[y][x];
}

void IsoM_1(int** img_in, int width, int height, int** img_out) {// �¿��Ī(y�� ��Ī)

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[i][(width - 1) - j];
}

void IsoM_2(int** img_in, int width, int height, int** img_out) {// ���ϴ�Ī(x�� ��Ī)

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[(height - 1) - i][j];
}

void IsoM_3(int** img_in, int width, int height, int** img_out) {// y=-x ��Ī(�������� ��Ī)

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[j][i];
}

void IsoM_4(int** img_in, int width, int height, int** img_out) {// y=x��Ī(������ ��Ī)

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[(height - 1) - j][(width - 1) - i];
}

void IsoM_5(int** img_in, int width, int height, int** img_out) {//90�� ȸ�� 

	if (width != height) {
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
				img_out[i][j] = img_in[(height - 1) - j][i];
	}else
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				img_out[i][j] = img_in[(width - 1) - j][i];
}

void IsoM_6(int** img_in, int width, int height, int** img_out) {//180�� ȸ��
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[(height - 1) - i][(width - 1) - j];
}

void IsoM_7(int** img_in, int width, int height, int** img_out) {//-90�� ȸ��
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[j][(height - 1) - i];
}

void Isometry(int num, int** img_in, int width, int height, int** img_out) {//Isom �����Լ�

	switch (num)
	{
	case 0:
		IsoM_0(img_in, width, height, img_out);break;
	case 1:
		IsoM_1(img_in, width, height, img_out);break;
	case 2:
		IsoM_2(img_in, width, height, img_out);break;
	case 3:
		IsoM_3(img_in, width, height, img_out);break;
	case 4:
		IsoM_4(img_in, width, height, img_out);break;
	case 5:
		IsoM_5(img_in, width, height, img_out);break;
	case 6:
		IsoM_6(img_in, width, height, img_out);break;
	case 7:
		IsoM_7(img_in, width, height, img_out);break;
	default:
		printf("Isom default", num);break;
	}
}

int ComputeAVG(int** image, int width, int height) {//�̹��� ��� ���
	int avg = 0;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			avg += image[j][i];
	return avg = avg / (width * height) + 0.5; //0.5�� �ݿø������ ���� �ּҰ� ��.
}

void ReadBlock(int** image, int x, int y, int dx, int dy, int** block) {//image(a)�� x,y�� ��ǥ�� ���ũ�⸸ŭ block(b)�� ����. 
	for (int i = 0; i < dy; i++)
		for (int j = 0; j < dx; j++)
			block[i][j] = image[y + i][x + j];
}

void WriteBlock(int** image, int x, int y, int dx, int dy, int** block) {//image(a)�� x,y�� ��ǥ���ٰ� block(b)�� ���ũ�⸸ŭ ����. 
	for (int i = 0; i < dy; i++)
		for (int j = 0; j < dx; j++)
			image[y + i][x + j] = block[i][j];
}

int ComputeError(int** block, int size_block, int** image, int width, int height, int x_temp, int y_temp) {//error�� ����
	int temp = 0;
	for (int y = 0; y < size_block; y++)
		for (int x = 0; x < size_block; x++)
			temp += abs(image[y][x] - block[y + y_temp][x + x_temp]);
	return temp;
}

void Find_AC(int** image, int size_x, int size_y, int block_avg) {//AC ��� ����
	for (int y = 0; y < size_y; y++)
		for (int x = 0; x < size_x; x++)
			image[y][x] = image[y][x] - block_avg;
}

void Copy_img(int** image, int width, int height, int** img_out) {//�̹��� ����
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			img_out[y][x] = image[y][x];
}

void AC_control(int** image, int width, int height, double alpha, int** temp) {//�̹����� alpha���ϰ�
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			temp[y][x] = (int)(alpha * image[y][x] + 0.5);//+0.5�� �� ����� �ս� ����.
}

EncodingResult TemplateMatchingWithDownSamplingPlusShuffle_StructEncoding(int** block, int bx, int by, int** image, int width, int height, double alpha) { //����ü ��� ���� �����ϴ� ���ڵ� �Լ�

	EncodingResult struct_Tmp;	//������ ����ü ����
	int error_min = INT_MAX;	//�ּҰ��� �񱳸� ���� int�ִ밪���� �ʱ�ȭ

	int** temp = (int**)IntAlloc2(bx * 2, by * 2);									//���������� 2�踸ŭ �о�� �޸� �Ҵ�
	int** domain = (int**)IntAlloc2(bx, by);										//��� ó���ϰ� ������ �̹��� �޸� �Ҵ�
	int** tmp_test = (int**)IntAlloc2(bx, by);										//isomó���ϰ� ������ �̹��� �޸� �Ҵ�
	int block_avg = ComputeAVG(block, bx, by);										//�о�� block�� ��հ� ���
	struct_Tmp.avg = block_avg;														//����� ����� ���������� �Ź� error�� �ּҰ��� �ٲ𶧸��� ����ü�� ������ �ʿ���� �����̹Ƿ�
	int** block_AC = (int**)IntAlloc2(bx, by);										//�о�� ���� ������� �����ص� ������� �̹���
	int** domain_AC = (int**)IntAlloc2(bx, by);										//��� ������ ���� �̹����� ������ �޸� �Ҵ�
	Copy_img(block, bx, by, block_AC);												//block�� ���� ������ block_AC�� ����								
	//ImageShow("test", block_AC, bx, by);		
	Find_AC(block_AC, bx, by, block_avg);											//AC��� ����
	
	// x,y��ǥ�� ���� �������� 2�踸ŭ ���� -> 1/2��� -> ��հ�� -> isom�ݺ��� ���� ������ �ּ��϶� alpha�� �������� ����
	for (int i = 0; i < height - (by * 2); i+=by) {
		for (int j = 0; j < width - (bx * 2); j+=bx) {
			ReadBlock(image, j, i, bx * 2, by * 2, temp);
			Contraction(temp, domain, bx * 2, by * 2);
			int domain_avg = ComputeAVG(domain, bx, by);							//domain�� ��հ� ����
			for (int n = 0; n < 8; n++) {											//isom�ݺ�
				Isometry(n, domain, bx, by, tmp_test);								
				Find_AC(tmp_test, bx, by, domain_avg);								//domain�� ��հ� ����
				for (double d = 0.3; d <= 1.0; d += 0.1) {							//alpha�� �ݺ� (double�� ����� 2.999999 ���ͽ��� - 1024bitȭ)
					AC_control(tmp_test, bx, by, d, domain_AC);						//error�� ���ϱ����� ���� ����
					int error = ComputeError(block_AC, bx, domain_AC, bx, by, 0, 0);//error�� ����
					if (error < error_min) {										//error���� �ּҰ��϶� ����ü�� �� ����.
						error_min = error;
						struct_Tmp.x = j;
						struct_Tmp.y = i;
						struct_Tmp.alpha = d;
						struct_Tmp.geo = n;
						struct_Tmp.error = error;
					}
				}
			}
		}
	}//	printf("%3d %3d %d %3d %.1lf %3d : %d\n", struct_Tmp.x, struct_Tmp.y, struct_Tmp.geo, struct_Tmp.avg, struct_Tmp.alpha, struct_Tmp.error);
	IntFree2(temp, bx * 2, by * 2);
	IntFree2(domain, bx, by);
	IntFree2(block_AC, bx, by);
	IntFree2(domain_AC, bx, by);
	IntFree2(tmp_test, bx, by);

	return struct_Tmp;
}

EncodingResult** ERAlloc2(int width, int height) {//���ڵ� ��� ����ü ���� �Ҵ��Լ�
	EncodingResult** tmp;
	tmp = (EncodingResult**)calloc(height, sizeof(EncodingResult*)); // calloc�� �⺻������ ���� 0���� �ʱ�ȭ�����ָ�, sizeof(int*)�� ũ���� height������ŭ �Ҵ�(1����)
	for (int i = 0; i<height; i++)
		tmp[i] = (EncodingResult*)calloc(width, sizeof(EncodingResult)); //calloc�� �⺻������ ���� 0���� �ʱ�ȭ�����ָ�, sizeof(int*)�� ũ���� width������ŭ �Ҵ�(2����)
	return(tmp);
}

void ERFree2(EncodingResult** image, int width, int height) {//���ڵ���� ����ü �����Ҵ������ �޸� ���� �Լ�
	for (int i = 0; i<height; i++)//�޸� ������ �Ҵ��� ��������
		free(image[i]);
	free(image);
}

bool WriteParameter(const char* name, EncodingResult** A, int x, int y) {//txt�� ����ü ������ ���Ϸ� ����
	FILE* fp = fopen(name, "w");
	if (fp == NULL) {
		printf("\n Failure in fopen!!"); return false;
	}
	for (int ty = 0; ty < y; ty++)
		for (int tx = 0; tx < x; tx++)
			fprintf(fp, "%d %d %d %d %f\n", A[ty][tx].x, A[ty][tx].y, A[ty][tx].geo, A[ty][tx].avg, A[ty][tx].alpha);
	fclose(fp);
	return true;
}

void main()
{
	int size = 16;//��� ������ ����
	int width, height;
	int** img_in = ReadImage("lena256x512.bmp", &width, &height);					//�̹��� �о��
	int** block_temp = (int**)IntAlloc2(size, size);								//�� �о�� temp �Ҵ�
		
	EncodingResult** en_result = ERAlloc2(width / size, height / size);				//���ڵ������ ������ ����ü ���� �Ҵ�
		
	for (int i = 0; i < height / size; i++)
		for (int j = 0; j < width / size; j++) {
			ReadBlock(img_in, size * j, size * i, size, size, block_temp);			//x,y�� ��ǥ�� size��ŭ�� ����� �о�ͼ� temp�� ����.
			en_result[i][j] = TemplateMatchingWithDownSamplingPlusShuffle_StructEncoding(block_temp, size, size, img_in, width, height, 1);//���ڵ� �Լ� ����(������� ����ü�� ����)
			printf(" < %3d , %3d > x��ǥ : %3d   y��ǥ : %3d  Error�� : %3d Isom�� : %3d  alpha�� : %.1lf  ��հ� : %3d \n", j * size, i *size, en_result[i][j].x, en_result[i][j].y, en_result[i][j].error, en_result[i][j].geo, en_result[i][j].alpha, en_result[i][j].avg); //����ó���� ������ �ð������� �������ؼ� ���
		}
	WriteParameter("encoding.txt", en_result, width / size, height / size);			//����ü�� ����� ������ txt�������� ����.
	//�޸������� �Ҵ��� ����
	ERFree2(en_result, width / size, height / size);
	IntFree2(block_temp, size, size);
	IntFree2(img_in, width, height);
}