#include "utils.h"

bool PathExists(const std::string& path) {
#ifdef _WIN32
	struct _stat buffer;
	return (_stat(path.c_str(), &buffer) == 0);
#else
	struct stat buffer;
	return (stat(path.c_str(), &buffer) == 0);
#endif  // !_WIN32
}

//bool staticPathExists(const std::string& path) {
//	PathExists(path);
//}

void MkDir(const std::string& path) {
	if (PathExists(path)) return;
	int ret = 0;
#ifdef _WIN32
	ret = _mkdir(path.c_str());
#else
	ret = mkdir(path.c_str(), 0755);
#endif  // !_WIN32
	if (ret != 0) {
		std::string path_error(path);
		path_error += " mkdir failed!";
		throw std::runtime_error(path_error);
		//cout << path_error << endl;
	}
}

//void staticMkDir(const std::string& path) {
//	MkDir(path);
//}

std::string DirName(const std::string& filepath) {
	auto pos = filepath.rfind(OS_PATH_SEP);
	if (pos == std::string::npos) {
		return "";
	}
	return filepath.substr(0, pos);
}

//std::string staticDirName(const std::string& filepath) {
//	DirName(filepath);
//}

void MkDirs(const std::string& path) {
	if (path.empty()) return;
	if (PathExists(path)) return;

	MkDirs(DirName(path));
	MkDir(path);
}

//void staticM

