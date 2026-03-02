FFMPEG_DIR = /home/dengyufei/ffmpeg6
OPENCV_DIR = /home/dengyufei/opencv4

CXX = g++
CXXFLAGS = -O3 -std=c++20 -Wno-deprecated-enum-enum-conversion -Wno-deprecated-declarations

INCLUDES = \
    `pkg-config --cflags dv-processing eigen3` \
    -Ilibav-cpp \
    -I$(FFMPEG_DIR)/include \
    -I$(OPENCV_DIR)/include

LIBS = \
    -L$(FFMPEG_DIR)/lib \
    -lavfilter -lswresample -lswscale -lavformat -lavcodec -lavutil \
    -llz4 -lzstd -lz -lm -lpthread \
    `pkg-config --libs dv-processing eigen3` \
    -L$(OPENCV_DIR)/lib \
    -lopencv_imgproc -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio

all: edi3

edi3: edi3.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) edi3.cpp -o edi3 $(LIBS)

clean:
	rm -f edi3







# all: build

# build: edi3 accumulator_test aedat_player mv_aedat_player

# edi3: edi3.cpp
# 	g++ -o edi3 -O3 `pkg-config --libs --cflags dv-processing opencv4 eigen3 libswscale libswresample libavutil libavcodec libavformat libavfilter` -Ilibav-cpp -std=c++20 -Wno-deprecated-enum-enum-conversion -Wno-deprecated-declarations edi3.cpp



mv_aedat_player: mv_aedat_player.cpp
	g++ -o mv_aedat_player -O3 `pkg-config --libs --cflags dv-processing opencv4 eigen3` -std=c++20 mv_aedat_player.cpp

aedat_player: aedat_player.cpp
	g++ -o aedat_player -g `pkg-config --libs --cflags dv-processing opencv4 eigen3` -std=c++20 aedat_player.cpp

accumulator_test: accumulator_test.cpp
	g++ -o accumulator_test -g accumulator_test.cpp

run: build
	./edi3

convert_jump: edi3
	./edi3 -p 0.7 -n -0.7 -a 46 -b 50 --t0=1703154373201968 -r 25 -e 12e-3 -s srgb -c 0 -o 0707_0.mp4 -i ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4
	./edi3 -p 0.7 -n -0.7 -a 46 -b 50 --t0=1703154373201968 -r 25 -e 12e-3 -s srgb -c 1 -o 0707_1.mp4 -i ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4
	./edi3 -p 0.7 -n -0.7 -a 46 -b 50 --t0=1703154373201968 -r 25 -e 12e-3 -s srgb -c 2 -o 0707_2.mp4 -i ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4
	./edi3 -p 0.7 -n -0.7 -a 46 -b 50 --t0=1703154373201968 -r 25 -e 12e-3 -s srgb -c 3 -o 0707_3.mp4 -i ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4
	./edi3 -p 0.7 -n -0.7 -a 46 -b 50 --t0=1703154373201968 -r 25 -e 12e-3 -s srgb -c 4 -o 0707_4.mp4 -i ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4
	./edi3 -p 0.7 -n -0.7 -a 46 -b 50 --t0=1703154373201968 -r 25 -e 12e-3 -s srgb -c 5 -o 0707_5.mp4 -i ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4

convert_calib: edi3
	./edi3 -p 0.7 -n -0.7 -a 0 -b 333 --t0=1703153613661205 -r 25 -e 12e-3 -s srgb -c 0 -o calib_0707_0.mp4 -i ../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4
	./edi3 -p 0.7 -n -0.7 -a 0 -b 333 --t0=1703153613661205 -r 25 -e 12e-3 -s srgb -c 1 -o calib_0707_1.mp4 -i ../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4
	./edi3 -p 0.7 -n -0.7 -a 0 -b 333 --t0=1703153613661205 -r 25 -e 12e-3 -s srgb -c 2 -o calib_0707_2.mp4 -i ../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4
	./edi3 -p 0.7 -n -0.7 -a 0 -b 333 --t0=1703153613661205 -r 25 -e 12e-3 -s srgb -c 3 -o calib_0707_3.mp4 -i ../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4
	./edi3 -p 0.7 -n -0.7 -a 0 -b 333 --t0=1703153613661205 -r 25 -e 12e-3 -s srgb -c 4 -o calib_0707_4.mp4 -i ../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4
	./edi3 -p 0.7 -n -0.7 -a 0 -b 333 --t0=1703153613661205 -r 25 -e 12e-3 -s srgb -c 5 -o calib_0707_5.mp4 -i ../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4