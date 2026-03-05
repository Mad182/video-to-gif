# Build stage
FROM alpine:3.21 AS build
RUN apk add --no-cache g++ make pkgconf ffmpeg-dev
WORKDIR /src
COPY Makefile video-to-gif.cpp ./
RUN make

# Runtime stage
FROM alpine:3.21
RUN apk add --no-cache ffmpeg-libavformat ffmpeg-libavcodec ffmpeg-libavutil ffmpeg-libswscale
COPY --from=build /src/video-to-gif /usr/local/bin/video-to-gif
ENTRYPOINT ["video-to-gif"]
