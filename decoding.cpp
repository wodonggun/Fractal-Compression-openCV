#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  
using namespace cv;

typedef struct EncodingResult {//���ڵ���� ������ ����ü
	int geo; int avg; int error; double alpha; int x, y;
}EncodingResult;

int** IntAlloc2(int width, int height) {
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));//calloc�� �⺻������ ���� 0���� �ʱ�ȭ�����ָ�, sizeof(int*)�� ũ���� height������ŭ �Ҵ�
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));//calloc�� �⺻������ ���� 0���� �ʱ�ȭ�����ָ�, sizeof(int*)�� ũ���� width������ŭ �Ҵ�
	return(tmp);
}

void IntFree2(int** image, int width, int height) {
	for (int i = 0; i<height; i++)
		free(image[i]);
	free(image);
}

int** ReadImage(const char* name, int* width, int* height) {
	Mat img = imread(name, IMREAD_GRAYSCALE);

	int** image = (int**)IntAlloc2(img.cols, img.rows);
	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);
	return(image);
}

void WriteImage(const char* name, int** image, int width, int height) {
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}

void ImageShow(const char* winname, int** image, int width, int height) {
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
	}
	else
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
		IsoM_0(img_in, width, height, img_out); break;
	case 1:
		IsoM_1(img_in, width, height, img_out); break;
	case 2:
		IsoM_2(img_in, width, height, img_out); break;
	case 3:
		IsoM_3(img_in, width, height, img_out); break;
	case 4:
		IsoM_4(img_in, width, height, img_out); break;
	case 5:
		IsoM_5(img_in, width, height, img_out); break;
	case 6:
		IsoM_6(img_in, width, height, img_out); break;
	case 7:
		IsoM_7(img_in, width, height, img_out); break;
	default:
		printf("Isom default", num); break;
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

void AC_control(int** image, int width, int height, double alpha, int** temp) {//�̹������� ���� ����
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			temp[y][x] = (int)(alpha * image[y][x] + 0.5);
}

bool ReadParameter(const char* name, EncodingResult** A, int width, int height) {
	FILE* fp = fopen(name, "r");

	if (fp == NULL) { printf("\n Failure in fopen!!");return false;
	}
	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
			fscanf(fp, "%d%d%d%d%lf", &(A[j][i].x), &(A[j][i].y), &(A[j][i].geo), &(A[j][i].avg), &(A[j][i].alpha));
	
	fclose(fp);
	return true;
}

EncodingResult** ERAlloc2(int width, int height) {
	EncodingResult** tmp;
	tmp = (EncodingResult**)calloc(height, sizeof(EncodingResult*));
	for (int i = 0; i<height; i++)
		tmp[i] = (EncodingResult*)calloc(width, sizeof(EncodingResult));
	return(tmp);
}

void ERFree2(EncodingResult** image, int width, int height) {
	for (int i = 0; i<height; i++)
		free(image[i]);
	free(image);
}

void Decoding(EncodingResult** en_Result, int** image_dec, int width, int height, int size_x, int size_y) //���ڵ� �Լ�
{
	int** block = IntAlloc2(size_x * 2, size_y * 2);
	int** block_contract_tmp = IntAlloc2(size_x, size_y);
	int** block_contract_tmp_aftercontrol = IntAlloc2(size_x, size_y);
	int** image_dec_tmp = IntAlloc2(width, height);
	//x,y��ǥ���� size�� 2��ũ�⸸ŭ �о��->�̹��� ���->������̹����� ��ձ��ϰ� �� �̹������� ��ջ��� -> ����� isom���� -> ����� alpha �����ְ� -> ��� �����ְ�-> dec�� �׷���.
	for (int i = 0; i < height / size_y; i++) {
		for (int j = 0; j < width / size_x; j++) {
			ReadBlock(image_dec, en_Result[i][j].x, en_Result[i][j].y, size_x * 2, size_y * 2, block);							//x,y��ǥ�� ���ũ���� 2�踸ŭ �о��
			Contraction(block, block_contract_tmp, size_x * 2, size_y * 2);														// �̹���1/2 ���
			int b_avg = ComputeAVG(block_contract_tmp, size_x, size_y);															//��հ� ���
			Find_AC(block_contract_tmp, size_x, size_y, b_avg);																	//��հ� ����
			Isometry(en_Result[i][j].geo, block_contract_tmp, size_x, size_y, block_contract_tmp_aftercontrol);					//����� isom(geo)����
			AC_control(block_contract_tmp_aftercontrol, size_x, size_y, en_Result[i][j].alpha, block_contract_tmp_aftercontrol);//alpha ������.
			Find_AC(block_contract_tmp_aftercontrol, size_x, size_y, -en_Result[i][j].avg);										//����� ��հ��� ������
			WriteBlock(image_dec_tmp, j * size_x, i * size_y, size_x, size_y, block_contract_tmp_aftercontrol);					//img_dec_tmp�� x,y�� ��ǥ�� ���ũ�⸸ŭ ó�����̹����� ����.
		}
	}
	Copy_img(image_dec_tmp, width, height, image_dec);				//image_tmp(���ڵ�ó���� �̹���)�� image_dec(����� �̹���)�� ����.
	//�޸� ����
	IntFree2(block, size_x * 2, size_y * 2);
	IntFree2(block_contract_tmp, size_x, size_y);
	IntFree2(block_contract_tmp_aftercontrol, size_x, size_y);
	IntFree2(image_dec_tmp, width, height);
}

void main()//���ڵ� �����Լ�
{
	int size = 16;																//��� ������ ����
	int width, height;
	int** img_in = ReadImage("lena256x512.bmp", &width, &height);				//�̹��� �о��
	int** image_dec = IntAlloc2(width, height);									//���ڵ� ��ȯ�Ҷ����� ����� �̹��� �Ҵ�
	EncodingResult** en_result = ERAlloc2(width / size, height / size);			//���ڵ��� ������ �������ִ� ����ü ���� �Ҵ�
	ReadParameter("encoding.txt", en_result, width / size, height / size);		//���ڵ� �� ������ �������ִ� txt���� �о��

	for (int i = 0; i< height; i++)											
		for (int j = 0; j < width; j++)
			image_dec[i][j] = 128;												//image_dec ����Ʈ �� 128�� �ʱ�ȭ

	for (int i = 0; i < 5; i++){
		printf("====== %d��° ���ڵ� ���� ======\n", i);						//0��° �̹����� ������ ȭ��
		ImageShow("���ڵ� �̹���", image_dec, width, height);					//0������ �̹������� 4������ �̹������� ���������� ���ڵ�(ó������ ȸ���� ��ȭ��)
		Decoding(en_result, image_dec, width, height, size, size);				//���ڵ� ����(���ڵ��� ������ �������ִ� ����ü, ���ڵ�������������̹���, ���� , ���� , ���x������, ���y������)
	}
	ImageShow("����", img_in, width, height);									//�̹��� ���� ���
	//�޸������� �Ҵ��� ����
	IntFree2(image_dec, width, height);
	IntFree2(img_in, width, height);
	ERFree2(en_result, width / size, height / size);
}