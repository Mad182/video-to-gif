CXX = g++
CXXFLAGS = -O2 -Wall -Wextra -std=c++17
LDFLAGS = $(shell pkg-config --libs libavformat libavcodec libavutil libswscale)
CPPFLAGS = $(shell pkg-config --cflags libavformat libavcodec libavutil libswscale)
TARGETS = video-to-gif

all: $(TARGETS)

video-to-gif: video-to-gif.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)

install: $(TARGETS)
	install -m 755 video-to-gif /usr/local/bin/video-to-gif

clean:
	rm -f $(TARGETS)

.PHONY: all clean install
