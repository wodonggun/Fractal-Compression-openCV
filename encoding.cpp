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
	}else
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

void AC_control(int** image, int width, int height, double alpha, int** temp) {//이미지에 alpha곱하고
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			temp[y][x] = (int)(alpha * image[y][x] + 0.5);//+0.5를 꼭 해줘야 손실 줄임.
}

EncodingResult TemplateMatchingWithDownSamplingPlusShuffle_StructEncoding(int** block, int bx, int by, int** image, int width, int height, double alpha) { //구조체 결과 값을 리턴하는 인코딩 함수

	EncodingResult struct_Tmp;	//리턴할 구조체 선언
	int error_min = INT_MAX;	//최소값은 비교를 위한 int최대값으로 초기화

	int** temp = (int**)IntAlloc2(bx * 2, by * 2);									//블럭사이즈의 2배만큼 읽어올 메모리 할당
	int** domain = (int**)IntAlloc2(bx, by);										//축소 처리하고 저장할 이미지 메모리 할당
	int** tmp_test = (int**)IntAlloc2(bx, by);										//isom처리하고 저장할 이미지 메모리 할당
	int block_avg = ComputeAVG(block, bx, by);										//읽어온 block의 평균값 계산
	struct_Tmp.avg = block_avg;														//블록의 평균은 일정하지만 매번 error의 최소값이 바뀔때마다 구조체에 넣으면 필요없는 연산이므로
	int** block_AC = (int**)IntAlloc2(bx, by);										//읽어온 블럭을 마음대로 수정해도 상관없는 이미지
	int** domain_AC = (int**)IntAlloc2(bx, by);										//모든 과정을 끝낸 이미지를 저장할 메모리 할당
	Copy_img(block, bx, by, block_AC);												//block에 대한 정보를 block_AC에 복사								
	//ImageShow("test", block_AC, bx, by);		
	Find_AC(block_AC, bx, by, block_avg);											//AC평균 제거
	
	// x,y좌표의 블럭을 사이즈의 2배만큼 읽음 -> 1/2축소 -> 평균계산 -> isom반복을 통해 에러가 최소일때 alpha와 정보들을 저장
	for (int i = 0; i < height - (by * 2); i+=by) {
		for (int j = 0; j < width - (bx * 2); j+=bx) {
			ReadBlock(image, j, i, bx * 2, by * 2, temp);
			Contraction(temp, domain, bx * 2, by * 2);
			int domain_avg = ComputeAVG(domain, bx, by);							//domain의 평균값 저장
			for (int n = 0; n < 8; n++) {											//isom반복
				Isometry(n, domain, bx, by, tmp_test);								
				Find_AC(tmp_test, bx, by, domain_avg);								//domain의 평균값 빼줌
				for (double d = 0.3; d <= 1.0; d += 0.1) {							//alpha값 반복 (double형 연산시 2.999999 부터시작 - 1024bit화)
					AC_control(tmp_test, bx, by, d, domain_AC);						//error값 구하기전에 알파 곱함
					int error = ComputeError(block_AC, bx, domain_AC, bx, by, 0, 0);//error값 추출
					if (error < error_min) {										//error값이 최소값일때 구조체에 값 넣음.
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

EncodingResult** ERAlloc2(int width, int height) {//인코딩 결과 구조체 동적 할당함수
	EncodingResult** tmp;
	tmp = (EncodingResult**)calloc(height, sizeof(EncodingResult*)); // calloc은 기본적으로 값을 0으로 초기화시켜주며, sizeof(int*)의 크기의 height갯수만큼 할당(1차원)
	for (int i = 0; i<height; i++)
		tmp[i] = (EncodingResult*)calloc(width, sizeof(EncodingResult)); //calloc은 기본적으로 값을 0으로 초기화시켜주며, sizeof(int*)의 크기의 width갯수만큼 할당(2차원)
	return(tmp);
}

void ERFree2(EncodingResult** image, int width, int height) {//인코딩결과 구조체 동적할당받은거 메모리 해제 함수
	for (int i = 0; i<height; i++)//메모리 해제는 할당의 역순으로
		free(image[i]);
	free(image);
}

bool WriteParameter(const char* name, EncodingResult** A, int x, int y) {//txt로 구조체 정보를 파일로 저장
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
	int size = 16;//블록 사이즈 결정
	int width, height;
	int** img_in = ReadImage("lena256x512.bmp", &width, &height);					//이미지 읽어옴
	int** block_temp = (int**)IntAlloc2(size, size);								//블럭 읽어올 temp 할당
		
	EncodingResult** en_result = ERAlloc2(width / size, height / size);				//인코딩결과를 저장할 구조체 동적 할당
		
	for (int i = 0; i < height / size; i++)
		for (int j = 0; j < width / size; j++) {
			ReadBlock(img_in, size * j, size * i, size, size, block_temp);			//x,y의 좌표의 size만큼의 블록을 읽어와서 temp에 저장.
			en_result[i][j] = TemplateMatchingWithDownSamplingPlusShuffle_StructEncoding(block_temp, size, size, img_in, width, height, 1);//인코딩 함수 실행(순서대로 구조체에 저장)
			printf(" < %3d , %3d > x좌표 : %3d   y좌표 : %3d  Error값 : %3d Isom값 : %3d  alpha값 : %.1lf  평균값 : %3d \n", j * size, i *size, en_result[i][j].x, en_result[i][j].y, en_result[i][j].error, en_result[i][j].geo, en_result[i][j].alpha, en_result[i][j].avg); //영상처리된 정보를 시각적으로 보기위해서 출력
		}
	WriteParameter("encoding.txt", en_result, width / size, height / size);			//구조체에 저장된 정보를 txt형식으로 저장.
	//메모리해제는 할당의 역순
	ERFree2(en_result, width / size, height / size);
	IntFree2(block_temp, size, size);
	IntFree2(img_in, width, height);
}