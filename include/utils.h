#pragma once
#ifndef UTILS
#define UTILS

#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#elif LINUX
#include <stdarg.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

using namespace std;

bool PathExists(const std::string& path);
void MkDir(const std::string& path);
std::string DirName(const std::string& filepath);
void MkDirs(const std::string& path);

// static Ort::Env env={};

#endif // !UTILS



