# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ivms/local_disk/plate_rec_linux

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ivms/local_disk/plate_rec_linux/build

# Include any dependencies generated for this target.
include CMakeFiles/plate_rec.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/plate_rec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/plate_rec.dir/flags.make

CMakeFiles/plate_rec.dir/src/plate_det.cpp.o: CMakeFiles/plate_rec.dir/flags.make
CMakeFiles/plate_rec.dir/src/plate_det.cpp.o: ../src/plate_det.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ivms/local_disk/plate_rec_linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/plate_rec.dir/src/plate_det.cpp.o"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plate_rec.dir/src/plate_det.cpp.o -c /home/ivms/local_disk/plate_rec_linux/src/plate_det.cpp

CMakeFiles/plate_rec.dir/src/plate_det.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plate_rec.dir/src/plate_det.cpp.i"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ivms/local_disk/plate_rec_linux/src/plate_det.cpp > CMakeFiles/plate_rec.dir/src/plate_det.cpp.i

CMakeFiles/plate_rec.dir/src/plate_det.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plate_rec.dir/src/plate_det.cpp.s"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ivms/local_disk/plate_rec_linux/src/plate_det.cpp -o CMakeFiles/plate_rec.dir/src/plate_det.cpp.s

CMakeFiles/plate_rec.dir/src/plate_rec.cpp.o: CMakeFiles/plate_rec.dir/flags.make
CMakeFiles/plate_rec.dir/src/plate_rec.cpp.o: ../src/plate_rec.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ivms/local_disk/plate_rec_linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/plate_rec.dir/src/plate_rec.cpp.o"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plate_rec.dir/src/plate_rec.cpp.o -c /home/ivms/local_disk/plate_rec_linux/src/plate_rec.cpp

CMakeFiles/plate_rec.dir/src/plate_rec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plate_rec.dir/src/plate_rec.cpp.i"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ivms/local_disk/plate_rec_linux/src/plate_rec.cpp > CMakeFiles/plate_rec.dir/src/plate_rec.cpp.i

CMakeFiles/plate_rec.dir/src/plate_rec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plate_rec.dir/src/plate_rec.cpp.s"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ivms/local_disk/plate_rec_linux/src/plate_rec.cpp -o CMakeFiles/plate_rec.dir/src/plate_rec.cpp.s

CMakeFiles/plate_rec.dir/src/utils.cpp.o: CMakeFiles/plate_rec.dir/flags.make
CMakeFiles/plate_rec.dir/src/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ivms/local_disk/plate_rec_linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/plate_rec.dir/src/utils.cpp.o"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plate_rec.dir/src/utils.cpp.o -c /home/ivms/local_disk/plate_rec_linux/src/utils.cpp

CMakeFiles/plate_rec.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plate_rec.dir/src/utils.cpp.i"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ivms/local_disk/plate_rec_linux/src/utils.cpp > CMakeFiles/plate_rec.dir/src/utils.cpp.i

CMakeFiles/plate_rec.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plate_rec.dir/src/utils.cpp.s"
	/usr/local/bin/x86_64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ivms/local_disk/plate_rec_linux/src/utils.cpp -o CMakeFiles/plate_rec.dir/src/utils.cpp.s

# Object files for target plate_rec
plate_rec_OBJECTS = \
"CMakeFiles/plate_rec.dir/src/plate_det.cpp.o" \
"CMakeFiles/plate_rec.dir/src/plate_rec.cpp.o" \
"CMakeFiles/plate_rec.dir/src/utils.cpp.o"

# External object files for target plate_rec
plate_rec_EXTERNAL_OBJECTS =

../dependencies/lib/libplate_rec.so: CMakeFiles/plate_rec.dir/src/plate_det.cpp.o
../dependencies/lib/libplate_rec.so: CMakeFiles/plate_rec.dir/src/plate_rec.cpp.o
../dependencies/lib/libplate_rec.so: CMakeFiles/plate_rec.dir/src/utils.cpp.o
../dependencies/lib/libplate_rec.so: CMakeFiles/plate_rec.dir/build.make
../dependencies/lib/libplate_rec.so: CMakeFiles/plate_rec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ivms/local_disk/plate_rec_linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../dependencies/lib/libplate_rec.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/plate_rec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/plate_rec.dir/build: ../dependencies/lib/libplate_rec.so

.PHONY : CMakeFiles/plate_rec.dir/build

CMakeFiles/plate_rec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/plate_rec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/plate_rec.dir/clean

CMakeFiles/plate_rec.dir/depend:
	cd /home/ivms/local_disk/plate_rec_linux/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ivms/local_disk/plate_rec_linux /home/ivms/local_disk/plate_rec_linux /home/ivms/local_disk/plate_rec_linux/build /home/ivms/local_disk/plate_rec_linux/build /home/ivms/local_disk/plate_rec_linux/build/CMakeFiles/plate_rec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/plate_rec.dir/depend
