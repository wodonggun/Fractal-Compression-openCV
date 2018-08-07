#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  
using namespace cv;

typedef struct EncodingResult {//인코딩결과 저장할 구조체
	int geo; int avg; int error; double alpha; int x, y;
}EncodingResult;

int** IntAlloc2(int width, int height) {
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));//calloc은 기본적으로 값을 0으로 초기화시켜주며, sizeof(int*)의 크기의 height갯수만큼 할당
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));//calloc은 기본적으로 값을 0으로 초기화시켜주며, sizeof(int*)의 크기의 width갯수만큼 할당
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

void Contraction(int** image, int** image_out, int width, int height) {// 1/2 이미지 축소
	for (int y = 0; y < height - 1; y += 2)
		for (int x = 0; x < width - 1; x += 2)
			image_out[y / 2][x / 2] = (image[y][x] + image[y][x + 1] + image[y + 1][x] + image[y + 1][x + 1]) / 4;
}

void IsoM_0(int** img_in, int width, int height, int** img_out) {// 이미지 복사(a->b copy)
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			img_out[y][x] = img_in[y][x];
}

void IsoM_1(int** img_in, int width, int height, int** img_out) {// 좌우대칭(y축 대칭)
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[i][(width - 1) - j];
}

void IsoM_2(int** img_in, int width, int height, int** img_out) {// 상하대칭(x축 대칭)
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[(height - 1) - i][j];
}

void IsoM_3(int** img_in, int width, int height, int** img_out) {// y=-x 대칭(역슬래쉬 대칭)
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[j][i];
}

void IsoM_4(int** img_in, int width, int height, int** img_out) {// y=x대칭(슬래쉬 대칭)
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[(height - 1) - j][(width - 1) - i];
}

void IsoM_5(int** img_in, int width, int height, int** img_out) {//90도 회전 
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

void IsoM_6(int** img_in, int width, int height, int** img_out) {//180도 회전
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[(height - 1) - i][(width - 1) - j];
}

void IsoM_7(int** img_in, int width, int height, int** img_out) {//-90도 회전
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_out[i][j] = img_in[j][(height - 1) - i];
}

void Isometry(int num, int** img_in, int width, int height, int** img_out) {//Isom 선택함수

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

int ComputeAVG(int** image, int width, int height) {//이미지 평균 계산
	int avg = 0;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			avg += image[j][i];
	return avg = avg / (width * height) + 0.5; //0.5를 반올림해줘야 오차 최소가 됨.
}

void ReadBlock(int** image, int x, int y, int dx, int dy, int** block) {//image(a)의 x,y의 좌표에 블록크기만큼 block(b)에 씌움. 
	for (int i = 0; i < dy; i++)
		for (int j = 0; j < dx; j++)
			block[i][j] = image[y + i][x + j];
}

void WriteBlock(int** image, int x, int y, int dx, int dy, int** block) {//image(a)의 x,y의 좌표에다가 block(b)를 블록크기만큼 씌움. 
	for (int i = 0; i < dy; i++)
		for (int j = 0; j < dx; j++)
			image[y + i][x + j] = block[i][j];
}

int ComputeError(int** block, int size_block, int** image, int width, int height, int x_temp, int y_temp) {//error값 검출
	int temp = 0;
	for (int y = 0; y < size_block; y++)
		for (int x = 0; x < size_block; x++)
			temp += abs(image[y][x] - block[y + y_temp][x + x_temp]);
	return temp;
}

void Find_AC(int** image, int size_x, int size_y, int block_avg) {//AC 평균 제거
	for (int y = 0; y < size_y; y++)
		for (int x = 0; x < size_x; x++)
				image[y][x] = image[y][x] - block_avg;
}

void Copy_img(int** image, int width, int height, int** img_out) {//이미지 복사
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			img_out[y][x] = image[y][x];
}

void AC_control(int** image, int width, int height, double alpha, int** temp) {//이미지값에 알파 곱함
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

void Decoding(EncodingResult** en_Result, int** image_dec, int width, int height, int size_x, int size_y) //디코딩 함수
{
	int** block = IntAlloc2(size_x * 2, size_y * 2);
	int** block_contract_tmp = IntAlloc2(size_x, size_y);
	int** block_contract_tmp_aftercontrol = IntAlloc2(size_x, size_y);
	int** image_dec_tmp = IntAlloc2(width, height);
	//x,y좌표에서 size의 2배크기만큼 읽어옴->이미지 축소->축소한이미지의 평균구하고 그 이미지에서 평균빼줌 -> 저장된 isom진행 -> 저장된 alpha 곱해주고 -> 평균 더해주고-> dec에 그려줌.
	for (int i = 0; i < height / size_y; i++) {
		for (int j = 0; j < width / size_x; j++) {
			ReadBlock(image_dec, en_Result[i][j].x, en_Result[i][j].y, size_x * 2, size_y * 2, block);							//x,y좌표의 블록크기의 2배만큼 읽어옴
			Contraction(block, block_contract_tmp, size_x * 2, size_y * 2);														// 이미지1/2 축소
			int b_avg = ComputeAVG(block_contract_tmp, size_x, size_y);															//평균값 계산
			Find_AC(block_contract_tmp, size_x, size_y, b_avg);																	//평균값 빼줌
			Isometry(en_Result[i][j].geo, block_contract_tmp, size_x, size_y, block_contract_tmp_aftercontrol);					//저장된 isom(geo)진행
			AC_control(block_contract_tmp_aftercontrol, size_x, size_y, en_Result[i][j].alpha, block_contract_tmp_aftercontrol);//alpha 곱해줌.
			Find_AC(block_contract_tmp_aftercontrol, size_x, size_y, -en_Result[i][j].avg);										//저장된 평균값을 더해줌
			WriteBlock(image_dec_tmp, j * size_x, i * size_y, size_x, size_y, block_contract_tmp_aftercontrol);					//img_dec_tmp의 x,y의 좌표에 블록크기만큼 처리된이미지를 씌움.
		}
	}
	Copy_img(image_dec_tmp, width, height, image_dec);				//image_tmp(디코딩처리한 이미지)를 image_dec(출력할 이미지)에 복사.
	//메모리 해제
	IntFree2(block, size_x * 2, size_y * 2);
	IntFree2(block_contract_tmp, size_x, size_y);
	IntFree2(block_contract_tmp_aftercontrol, size_x, size_y);
	IntFree2(image_dec_tmp, width, height);
}

void main()//디코딩 메인함수
{
	int size = 16;																//블록 사이즈 결정
	int width, height;
	int** img_in = ReadImage("lena256x512.bmp", &width, &height);				//이미지 읽어옴
	int** image_dec = IntAlloc2(width, height);									//디코딩 변환할때마다 출력할 이미지 할당
	EncodingResult** en_result = ERAlloc2(width / size, height / size);			//디코딩할 정보를 가지고있는 구조체 동적 할당
	ReadParameter("encoding.txt", en_result, width / size, height / size);		//디코딩 할 정보를 가지고있는 txt파일 읽어옴

	for (int i = 0; i< height; i++)											
		for (int j = 0; j < width; j++)
			image_dec[i][j] = 128;												//image_dec 디폴트 값 128로 초기화

	for (int i = 0; i < 5; i++){
		printf("====== %d번째 디코딩 진행 ======\n", i);						//0번째 이미지는 무색의 화면
		ImageShow("디코딩 이미지", image_dec, width, height);					//0번돌린 이미지부터 4번돌린 이미지까지 순차적으로 디코딩(처음에는 회색의 도화지)
		Decoding(en_result, image_dec, width, height, size, size);				//디코딩 실행(디코딩할 정보를 가지고있는 구조체, 디코딩결과값저장할이미지, 가로 , 새로 , 블록x사이즈, 블록y사이즈)
	}
	ImageShow("원본", img_in, width, height);									//이미지 원본 출력
	//메모리해제는 할당의 역순
	IntFree2(image_dec, width, height);
	IntFree2(img_in, width, height);
	ERFree2(en_result, width / size, height / size);
}