// reconApp.cpp : Defines the entry point for the console application.
//
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <stdexcept>
#include <cerrno>
#include <set>

#include "stdafx.h"
#include "Recon_API.h"

int main(int argc, char** argv)
{
	std::string path_image;
	std::string path_scene;
	std::string match_mode;
	std::string mvs_type;
	std::cout << "Input image directory: ";
	std::getline(std::cin, path_image);

	std::cout << "Input scene directory: ";
	std::getline(std::cin, path_scene);

	std::cout << "matching mode(f-cv_akaze+cv_sift, a-cv_akaze, s-cv_sift, m-mve_feature matching): ";
	std::getline(std::cin, match_mode);
	int nMatchMode = 0;
	if (match_mode == "f")
	{
		nMatchMode = 0;//cv_akaze plus cv_sift feature matching
	}
	if (match_mode == "a")
	{
		nMatchMode = 1;//cv_akaze feature matching
	}
	if (match_mode == "s")
	{
		nMatchMode = 2;//cv_sift feature matching
	}
	if (match_mode == "m")
	{
		nMatchMode = 3;//mve feature matching
		std::cout << "mve match_mode: " << std::endl;
	}
	std::cout << "mvs type(m-mvs, s-smvs): ";
	std::getline(std::cin, mvs_type);
	int nMvsType = 0;//mvs
	if (mvs_type == "s")
		nMvsType = 1;//smvs
	if (mvs_type == "m")
		nMvsType = 0;//mvs
	std::cout << "match mode:" << match_mode.c_str() << "," << "mvs type:" << mvs_type.c_str() << std::endl;
	std::cout << "match mode: " << nMatchMode << "," << "n mvs type:" << nMvsType << std::endl;
	recon_all(path_image, path_scene, nMatchMode, nMvsType);
	return 0;
}

